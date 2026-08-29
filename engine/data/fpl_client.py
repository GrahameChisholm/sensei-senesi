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

    # --- Manager (entry) endpoints (A3 / SEASON_SIMULATOR.md) -----------------------------------
    # Public data, no auth required, but each needs the caller's own FPL entry/team id.

    def get_entry(self, entry_id: int) -> dict[str, Any]:
        """Manager profile: name, and — critically — ``last_deadline_bank``/``last_deadline_value``
        (both tenths of a million, matching :func:`features.team_state.compute_sell_price`'s own
        convention), the bank/squad-value snapshot as of the most recent deadline."""
        return self._get(f"/entry/{entry_id}/")

    def get_entry_picks(self, entry_id: int, gameweek: int) -> dict[str, Any]:
        """This manager's squad for one gameweek: ``picks`` (15 entries, each carrying ``element``,
        ``position`` — 1-11 is the starting XI in formation-slot order, 12-15 the bench, **not** a
        real football position — ``multiplier``, ``is_captain``, ``is_vice_captain``),
        ``active_chip``, and ``entry_history`` (that gameweek's bank/value/points snapshot)."""
        return self._get(f"/entry/{entry_id}/event/{gameweek}/picks/")

    def get_entry_transfers(self, entry_id: int) -> list[dict[str, Any]]:
        """Every transfer this manager has made this season, each carrying
        ``element_in``/``element_in_cost``/``element_out``/``element_out_cost`` (tenths of a
        million) and the ``event`` it happened in. **Does not cover the initial pre-GW1 squad** —
        FPL tracks that as a separate "team pick" action, not a transfer, so a player who has been
        in the squad since GW1 and never been transferred has no record here at all (see
        ``engine.data.team_state_builder``'s own docstring for how that gap is handled)."""
        return self._get(f"/entry/{entry_id}/transfers/")

    def get_entry_history(self, entry_id: int) -> dict[str, Any]:
        """This manager's full season history: ``current`` (one row per gameweek — points, rank,
        bank, value, transfers made and their cost), ``past`` (previous seasons' final totals), and
        ``chips`` (name + event of every chip already played this season)."""
        return self._get(f"/entry/{entry_id}/history/")

    def get_league_standings(self, league_id: int, page: int = 1) -> dict[str, Any]:
        """One page (50 entries) of a classic mini-league's standings (MINI_LEAGUE_PLAN M16):
        ``league`` (name and metadata) plus ``standings`` (``has_next`` and ``results``, each entry
        carrying ``entry`` (the manager's real FPL entry/team ID -- **not** ``id``, an unrelated
        per-row identifier some standings responses also carry that is not a usable entry ID at
        all), ``entry_name``, ``player_name``, ``rank``, ``total``, ``event_total``). A caller
        wanting the whole league pages through this itself, stopping once ``has_next`` is false or
        its own rival cap is reached -- this method is a plain one-request wrapper like every
        other endpoint on this client, not a batch-fetch convenience."""
        return self._get(f"/leagues-classic/{league_id}/standings/?page_standings={page}")


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
