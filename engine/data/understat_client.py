"""Understat client — player and team xG/xA/npxG per match, back to 2014/15 (1.1).

Understat's player/team/league pages are now client-rendered; the browser fetches JSON from a
small set of internal endpoints rather than embedding data in the HTML. This client talks to
those same JSON endpoints directly instead of scraping HTML:

- ``GET /getLeagueData/{league}/{season}`` -> season-aggregate stats for every player in the
  league, per-match team-level xG/xGA/npxG/npxGA history for every team, and a per-fixture
  home/away xG list. This is the source for the team-level history the clean-sheet model (2.4)
  needs.
- ``GET /getPlayerData/{understat_player_id}`` -> full match-by-match history for one player
  (goals, xG, assists, xA, minutes, date, season) — the source for the per-90 rate stats the
  goals/assists models (2.2/2.3) need.

Both endpoints return plain JSON (gzip-encoded) rather than the escaped ``JSON.parse('...')``
blobs older scraping guides describe — no HTML parsing is needed here. Understat has no
documented rate limit, but a browser-like ``User-Agent`` is required or requests are served a
stripped page/empty body.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import httpx
import pandas as pd

BASE_URL = "https://understat.com"
DEFAULT_TIMEOUT = 15.0

# Understat serves a bot-safe stripped response without a browser-like User-Agent.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}

EARLIEST_SEASON = 2014  # 2014/15 — Understat's first covered season.


class UnderstatClientError(RuntimeError):
    """Raised when Understat returns something the client can't use."""


@dataclass
class UnderstatClient:
    base_url: str = BASE_URL
    client: httpx.Client = field(
        default_factory=lambda: httpx.Client(timeout=DEFAULT_TIMEOUT, headers=DEFAULT_HEADERS)
    )

    def __enter__(self) -> UnderstatClient:
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
            raise UnderstatClientError(f"Understat request failed for {path}: {exc}") from exc
        try:
            data = response.json()
        except ValueError as exc:
            raise UnderstatClientError(f"Understat returned non-JSON body for {path}") from exc
        if not data:
            raise UnderstatClientError(f"Understat returned an empty body for {path}")
        return data

    def get_league_data(self, season: int, league: str = "EPL") -> dict[str, Any]:
        """One season of EPL data: ``teams`` (per-team match history), ``players`` (season
        aggregates), ``dates`` (per-fixture home/away xG).

        ``season`` is the year the season *started* in (e.g. 2024 for 2024/25), matching
        Understat's own convention.
        """
        data = self._get(f"/getLeagueData/{league}/{season}")
        for key in ("teams", "players", "dates"):
            if key not in data:
                raise UnderstatClientError(f"Understat league data missing '{key}' key")
        return data

    def get_player_data(self, understat_player_id: int) -> dict[str, Any]:
        """Full match-by-match history for one player, keyed by Understat's own player id (not
        the FPL element id — see crosswalk.py)."""
        data = self._get(f"/getPlayerData/{understat_player_id}")
        if "matches" not in data:
            raise UnderstatClientError("Understat player data missing 'matches' key")
        return data


def league_data_to_dataframes(league_data: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Flatten a ``get_league_data`` payload into tabular form, ready for
    :func:`engine.data.snapshots.capture_snapshot`.

    - ``players``: one row per player, season-aggregate xG/xA/npxG/time.
    - ``teams_history``: one row per team per match, carrying the per-match team-level
      xG/xGA/npxG/npxGA the clean-sheet model (2.4) needs — flattened out of the
      ``teams[team_id]["history"]`` nesting with ``team_id``/``team_title`` columns attached.
    - ``dates``: one row per league fixture, home/away team + xG.
    """
    players_df = pd.DataFrame(league_data["players"])

    history_rows: list[dict[str, Any]] = []
    for team_id, team in league_data["teams"].items():
        for match in team["history"]:
            history_rows.append({"team_id": team_id, "team_title": team["title"], **match})
    teams_history_df = pd.DataFrame(history_rows)

    dates_df = pd.json_normalize(league_data["dates"], sep="_")

    return {"players": players_df, "teams_history": teams_history_df, "dates": dates_df}


def player_data_to_dataframe(player_data: dict[str, Any]) -> pd.DataFrame:
    """One row per match for a single player — the match-by-match history the goals/assists
    models (2.2/2.3) build their per-90 EWMA rates from."""
    return pd.DataFrame(player_data["matches"])
