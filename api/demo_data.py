"""Deterministic synthetic state for running/testing the API without a live data pipeline.

BUILD_PLAN Phase 6 ("weekly: refresh data, capture the pre-deadline snapshot, regenerate
projections... log predictions") is what would normally populate :class:`api.state.AppState` for
real. That job isn't wired up yet — Phase 1's snapshot store is empty in this environment — so
this module builds a small, internally-consistent stand-in: 4 teams, a legal 15-player squad, a
wider candidate pool, and a 5-gameweek fixture schedule (including one deliberate blank and one
deliberate double, so Free Hit/Wildcard's evaluators have something real to react to).

Every number here is arithmetic on ``player_id``/``gameweek``, not randomness — the same call
always returns the same state, so manual testing and automated API tests both see stable figures.
Swap :func:`load_demo_state` for whatever the real weekly job persists once Phase 6 exists.
"""

from __future__ import annotations

from api.state import AppState
from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection, PlayerHorizonProjection
from engine.rates import league_average_rate
from features.fixtures import TeamFixture, TeamRates
from features.team_state import MyTeamState, SquadPlayer

HORIZON_GAMEWEEKS = (1, 2, 3, 4, 5)

# One player (id 5) is deliberately modelled as unlikely to feature at all, to exercise
# transfers.py's "forced sell" detection in a running demo.
_INJURED_PLAYER_ID = 5

_HEALTHY_MINUTES = MinutesDistribution(
    p_zero=0.05,
    p_1_to_59=0.10,
    p_60_plus=0.85,
    expected_minutes_given_1_to_59=30.0,
    expected_minutes_given_60_plus=88.0,
)
_INJURED_MINUTES = MinutesDistribution(
    p_zero=0.95,
    p_1_to_59=0.05,
    p_60_plus=0.0,
    expected_minutes_given_1_to_59=10.0,
    expected_minutes_given_60_plus=0.0,
)

_TEAM_RATES: dict[int, TeamRates] = {
    1: TeamRates(home_xg_per_90=1.8, away_xg_per_90=1.5, home_xga_per_90=1.1, away_xga_per_90=1.3),
    2: TeamRates(home_xg_per_90=0.9, away_xg_per_90=0.7, home_xga_per_90=1.8, away_xga_per_90=2.0),
    3: TeamRates(home_xg_per_90=1.3, away_xg_per_90=1.1, home_xga_per_90=1.3, away_xga_per_90=1.5),
    4: TeamRates(home_xg_per_90=0.8, away_xg_per_90=0.6, home_xga_per_90=0.9, away_xga_per_90=1.0),
}

# (gameweek, home_team_id, away_team_id). Deliberately: GW4 has only one fixture (teams 3 and 4
# blank), and GW5 gives team 1 a double (playing both team 3 and team 4) while team 2 blanks.
_MATCHES = [
    (1, 1, 2),
    (1, 3, 4),
    (2, 2, 3),
    (2, 4, 1),
    (3, 1, 3),
    (3, 2, 4),
    (4, 1, 2),
    (5, 1, 3),
    (5, 1, 4),
]

_SQUAD_POSITIONS = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
# Cycled to give the wider pool more DEF/MID depth than GK/FWD, roughly mirroring real squads.
_POOL_POSITION_CYCLE = ["DEF", "MID", "MID", "DEF", "FWD", "GK", "MID", "DEF"]

_BASE_POINTS_BY_POSITION = {"GK": 3.5, "DEF": 4.0, "MID": 4.5, "FWD": 4.5}


def _team_for_player(player_id: int) -> int:
    return ((player_id - 1) % 4) + 1


def _expected_points(player_id: int, position: str, gameweek: int) -> float:
    if player_id == _INJURED_PLAYER_ID:
        return 0.2
    return _BASE_POINTS_BY_POSITION[position] + ((player_id + gameweek) % 5) * 0.5


def _horizon_projection(player_id: int, position: str) -> PlayerHorizonProjection:
    minutes = _INJURED_MINUTES if player_id == _INJURED_PLAYER_ID else _HEALTHY_MINUTES
    gameweeks = {
        gw: PlayerGameweekProjection(
            player_id=player_id,
            position=position,
            gameweek=gw,
            minutes=minutes,
            breakdown=ComponentBreakdown(
                appearance=_expected_points(player_id, position, gw),
                goals=0.0,
                assists=0.0,
                clean_sheet=0.0,
                goals_conceded=0.0,
                defensive_contribution=0.0,
                saves=0.0,
                bonus=0.0,
                cards=0.0,
                penalty_misses=0.0,
            ),
        )
        for gw in HORIZON_GAMEWEEKS
    }
    return PlayerHorizonProjection(player_id=player_id, position=position, gameweeks=gameweeks)


def _build_fixtures() -> list[TeamFixture]:
    fixtures = []
    for gameweek, home_id, away_id in _MATCHES:
        fixtures.append(
            TeamFixture(team_id=home_id, opponent_id=away_id, gameweek=gameweek, is_home=True)
        )
        fixtures.append(
            TeamFixture(team_id=away_id, opponent_id=home_id, gameweek=gameweek, is_home=False)
        )
    return fixtures


def load_demo_state() -> AppState:
    """Build the full synthetic :class:`~api.state.AppState` described in this module's
    docstring."""
    squad_ids = list(range(1, 16))
    pool_ids = list(range(16, 41))

    projections: dict[int, PlayerHorizonProjection] = {}
    team_id_by_player: dict[int, int] = {}

    squad_players = []
    for i, player_id in enumerate(squad_ids):
        position = _SQUAD_POSITIONS[i]
        purchase, current = (45, 55) if player_id == 1 else (50, 50)
        squad_players.append(
            SquadPlayer(
                player_id=player_id,
                position=position,
                purchase_price=purchase,
                current_price=current,
            )
        )
        projections[player_id] = _horizon_projection(player_id, position)
        team_id_by_player[player_id] = _team_for_player(player_id)

    buy_prices: dict[int, int] = {}
    for i, player_id in enumerate(pool_ids):
        position = _POOL_POSITION_CYCLE[i % len(_POOL_POSITION_CYCLE)]
        projections[player_id] = _horizon_projection(player_id, position)
        team_id_by_player[player_id] = _team_for_player(player_id)
        buy_prices[player_id] = 40 + (player_id % 20)

    my_team = MyTeamState(
        squad=tuple(squad_players),
        starting_xi=tuple(squad_ids[:11]),
        bench_order=tuple(squad_ids[11:]),
        captain_id=squad_ids[0],
        vice_captain_id=squad_ids[1],
        bank=20,
        free_transfers=1,
        chips_remaining=frozenset({"wildcard", "free_hit", "triple_captain", "bench_boost"}),
    )

    team_xg_overall = {
        team_id: (rates.home_xg_per_90 + rates.away_xg_per_90) / 2
        for team_id, rates in _TEAM_RATES.items()
    }
    team_xga_overall = {
        team_id: (rates.home_xga_per_90 + rates.away_xga_per_90) / 2
        for team_id, rates in _TEAM_RATES.items()
    }

    return AppState(
        my_team=my_team,
        projections=projections,
        team_id_by_player=team_id_by_player,
        buy_prices=buy_prices,
        fixtures=_build_fixtures(),
        team_rates=dict(_TEAM_RATES),
        league_avg_xg_per_90=league_average_rate(team_xg_overall),
        league_avg_xga_per_90=league_average_rate(team_xga_overall),
        horizon_gameweeks=list(HORIZON_GAMEWEEKS),
    )
