"""In-memory application state the API's endpoints read from (BUILD_PLAN Phase 5.1: "the API
mostly serves precomputed results"). Populated once (currently by ``api.demo_data.load_demo_state``;
eventually by the Phase 6 weekly refresh job) and read by every request — endpoints never fetch
or compute projections themselves, only rank/compare what's already here via ``features/``.
"""

from __future__ import annotations

from dataclasses import dataclass

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
