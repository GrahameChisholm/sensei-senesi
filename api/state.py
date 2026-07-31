"""In-memory application state the API's endpoints read from (BUILD_PLAN Phase 5.1: "the API
mostly serves precomputed results"). Populated once (currently by ``api.demo_data.load_demo_state``;
eventually by the Phase 6 weekly refresh job) and read by every request — endpoints never fetch
or compute projections themselves, only rank/compare what's already here via ``features/``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from engine.projections import PlayerHorizonProjection
from features.fixtures import TeamFixture, TeamRates
from features.team_state import MyTeamState


@dataclass
class AppState:
    my_team: MyTeamState
    projections: dict[int, PlayerHorizonProjection]
    team_id_by_player: dict[int, int]
    buy_prices: dict[int, int]
    fixtures: list[TeamFixture]
    team_rates: dict[int, TeamRates]
    league_avg_xg_per_90: float
    league_avg_xga_per_90: float
    horizon_gameweeks: list[int]
    # A5: display name per player, for Player Search -- everything else in this state is keyed by
    # player_id alone, since every other feature (captaincy, transfers, chips) only ever needs to
    # rank/compare, never display a name. Defaults to empty so every existing caller that predates
    # this field is unaffected; a player missing from it just has no display name available yet.
    player_names: dict[int, str] = field(default_factory=dict)
    # A6: when this state's projections were actually generated -- the snapshot's own
    # `captured_at` for a real weekly refresh, `None` for synthetic demo data (there is no real
    # "as of" time for numbers that were never actually computed from live data; the UI's "data as
    # of" banner shows an honest "demo data" label rather than fabricating a timestamp for it).
    generated_at: datetime | None = None


_state: AppState | None = None


def get_state() -> AppState:
    """FastAPI dependency — returns the process-wide state, loading the demo dataset on first
    use. Tests override this via ``app.dependency_overrides[get_state]`` to inject a fixture
    state instead."""
    global _state
    if _state is None:
        from api.demo_data import load_demo_state

        _state = load_demo_state()
    return _state


def set_state(state: AppState) -> None:
    """Replace the process-wide state — what the Phase 6 weekly job would call after each
    refresh."""
    global _state
    _state = state
