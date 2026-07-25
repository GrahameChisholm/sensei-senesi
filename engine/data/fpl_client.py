"""Official FPL API client — player master data, prices, fixtures, gameweek history (1.1).

No API key required, no rate limits of concern (per BUILD_PLAN 1.1). A plain ``httpx.Client`` is
injected so tests can swap in an ``httpx.MockTransport`` instead of hitting the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

BASE_URL = "https://fantasy.premierleague.com/api"
DEFAULT_TIMEOUT = 15.0


class FPLClientError(RuntimeError):
    """Raised when the FPL API returns something the client can't use."""


@dataclass
class FPLClient:
    """Thin wrapper over the official (unofficial-but-public) FPL API.

    Endpoints covered map directly onto BUILD_PLAN 1.1's ingestion list: player master data,
    prices, ownership, positions, ICT components, defensive-contribution stats, fixtures, team
    news/injury flags, and full gameweek-by-gameweek history.
    """

    base_url: str = BASE_URL
    client: httpx.Client = field(default_factory=lambda: httpx.Client(timeout=DEFAULT_TIMEOUT))

    def __enter__(self) -> FPLClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def close(self) -> None:
        self.client.close()

    def _get(self, path: str) -> Any:
        try:
            response = self.client.get(f"{self.base_url}{path}")
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise FPLClientError(f"FPL API request failed for {path}: {exc}") from exc
        try:
            return response.json()
        except ValueError as exc:
            raise FPLClientError(f"FPL API returned non-JSON body for {path}") from exc

    def get_bootstrap_static(self) -> dict[str, Any]:
        """Player master data, teams, positions ("element_types"), and gameweek ("events") metadata.

        This single endpoint carries prices, ownership, ICT components, and the
        defensive-contribution raw counters (``clearances_blocks_interceptions``, ``tackles``,
        ``recoveries``, ``defensive_contribution``) needed by 2.5.
        """
        return self._get("/bootstrap-static/")

    def get_fixtures(self, event: int | None = None) -> list[dict[str, Any]]:
        """All fixtures, or just one gameweek's if ``event`` is given.

        Each fixture carries the FPL's own difficulty rating (``team_h_difficulty`` /
        ``team_a_difficulty``) which the engine deliberately does not use as a model input (see
        BUILD_PLAN 2.2) but which is convenient for validation / display.
        """
        path = "/fixtures/" if event is None else f"/fixtures/?event={event}"
        return self._get(path)

    def get_element_summary(self, player_id: int) -> dict[str, Any]:
        """Per-player detail: ``fixtures`` (upcoming), ``history`` (this season, per gameweek),
        and ``history_past`` (previous seasons, aggregated) — the full gameweek-by-gameweek
        history BUILD_PLAN 1.1 asks for.
        """
        return self._get(f"/element-summary/{player_id}/")

    def iter_element_summaries(self, player_ids: list[int]) -> dict[int, dict[str, Any]]:
        """Convenience batch fetch. FPL has no bulk per-player-history endpoint, so this is one
        request per player — callers doing a full-league pull should expect ~500-700 requests.
        """
        return {player_id: self.get_element_summary(player_id) for player_id in player_ids}


def _drop_nested_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop any column holding raw dicts/lists (e.g. events' ``overrides``/``chip_plays``,
    elements' ``scout_risks``) — display-only nested fields the engine has no use for, and which
    an empty-struct value (``{}``) makes Parquet unable to write at all.
    """
    nested_columns = [
        col for col in df.columns if df[col].apply(lambda v: isinstance(v, (dict, list))).any()
    ]
    return df.drop(columns=nested_columns)


def bootstrap_to_dataframes(bootstrap: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Flatten the handful of ``bootstrap-static`` list fields the engine actually needs into
    tabular form, ready for :func:`engine.data.snapshots.capture_snapshot`."""
    return {
        "elements": _drop_nested_columns(pd.DataFrame(bootstrap["elements"])),
        "teams": _drop_nested_columns(pd.DataFrame(bootstrap["teams"])),
        "element_types": _drop_nested_columns(pd.DataFrame(bootstrap["element_types"])),
        "events": _drop_nested_columns(pd.DataFrame(bootstrap["events"])),
    }


def fixtures_to_dataframe(fixtures: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(fixtures)
