"""Tests for features/captaincy.py — full-pool captaincy ranking (BUILD_PLAN Phase 4)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection
from engine.simulate import PlayerSimulationSummary
from features.captaincy import rank_captaincy_pool
from features.team_state import MyTeamState, SquadPlayer

MINUTES = MinutesDistribution(
    p_zero=0.05,
    p_1_to_59=0.10,
    p_60_plus=0.85,
    expected_minutes_given_1_to_59=30.0,
    expected_minutes_given_60_plus=88.0,
)


def _breakdown(
    total_bias: float = 0.0, goals: float = 0.0, bonus: float = 0.0
) -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=1.8 + total_bias,
        goals=goals,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=bonus,
        cards=0.0,
        penalty_misses=0.0,
    )


def _projection(
    player_id: int,
    position: str = "FWD",
    gameweek: int = 1,
    breakdown: ComponentBreakdown | None = None,
    floor: float | None = None,
    ceiling: float | None = None,
    prob_big_haul: float = 0.1,
) -> PlayerGameweekProjection:
    breakdown = breakdown or _breakdown()
    simulation = None
    if floor is not None and ceiling is not None:
        simulation = PlayerSimulationSummary(
            player_id=player_id,
            mean=breakdown.total,
            median=breakdown.total,
            floor=floor,
            ceiling=ceiling,
            prob_big_haul=prob_big_haul,
            raw_points=np.array([breakdown.total]),
        )
    return PlayerGameweekProjection(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        minutes=MINUTES,
        breakdown=breakdown,
        simulation=simulation,
    )


def _squad_player(player_id: int, position: str = "MID") -> SquadPlayer:
    return SquadPlayer(player_id=player_id, position=position, price=50)


def _team(owned_ids: list[int], starting_ids: list[int]) -> MyTeamState:
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad = tuple(
        _squad_player(pid, positions[i % len(positions)]) for i, pid in enumerate(owned_ids)
    )
    bench = tuple(pid for pid in owned_ids if pid not in starting_ids)
    return MyTeamState(
        squad=squad,
        starting_xi=tuple(starting_ids),
        bench_order=bench,
        captain_id=starting_ids[0],
        vice_captain_id=starting_ids[1],
    )


def test_rank_captaincy_pool_rejects_empty_projections():
    team = _team(list(range(1, 16)), list(range(1, 12)))
    with pytest.raises(ValueError):
        rank_captaincy_pool(team, [])


def test_ranked_pool_sorted_by_expected_points_descending():
    team = _team(list(range(1, 16)), list(range(1, 12)))
    projections = [
        _projection(1, breakdown=_breakdown(goals=2.0)),  # highest EV
        _projection(100, breakdown=_breakdown(goals=0.0)),  # not owned at all
        _projection(2, breakdown=_breakdown(goals=1.0)),
    ]
    result = rank_captaincy_pool(team, projections)
    assert [o.player_id for o in result.ranked_pool] == [1, 2, 100]


def test_ownership_and_eligibility_flags():
    team = _team(list(range(1, 16)), list(range(1, 12)))
    projections = [
        _projection(1),  # owned + starting
        _projection(13),  # owned but benched (13,14,15 are bench)
        _projection(999),  # not owned at all
    ]
    result = rank_captaincy_pool(team, projections)
    by_id = {o.player_id: o for o in result.ranked_pool}
    assert by_id[1].is_owned and by_id[1].is_eligible
    assert by_id[13].is_owned and not by_id[13].is_eligible
    assert not by_id[999].is_owned and not by_id[999].is_eligible


def test_top_ev_pick_restricted_to_eligible_players():
    team = _team(list(range(1, 16)), list(range(1, 12)))
    projections = [
        _projection(999, breakdown=_breakdown(goals=10.0)),  # highest EV but not owned
        _projection(1, breakdown=_breakdown(goals=1.0)),  # eligible, lower EV
    ]
    result = rank_captaincy_pool(team, projections)
    assert result.ranked_pool[0].player_id == 999  # still tops the full pool ranking
    assert result.top_ev_pick.player_id == 1  # but the eligible pick is the starting-XI player


def test_safe_and_punt_picks_use_floor_and_ceiling():
    team = _team(list(range(1, 16)), list(range(1, 12)))
    projections = [
        _projection(1, floor=2.0, ceiling=6.0),
        _projection(2, floor=4.0, ceiling=15.0),
    ]
    result = rank_captaincy_pool(team, projections)
    assert result.safe_pick.player_id == 2  # higher floor
    assert result.punt_pick.player_id == 2  # higher ceiling


def test_safe_and_punt_picks_none_when_no_eligible_player_has_simulation():
    team = _team(list(range(1, 16)), list(range(1, 12)))
    projections = [_projection(1, floor=None, ceiling=None)]
    result = rank_captaincy_pool(team, projections)
    assert result.top_ev_pick is not None
    assert result.safe_pick is None
    assert result.punt_pick is None


def test_reasoning_mentions_top_components_and_ev():
    projection = _projection(1, breakdown=_breakdown(goals=3.0, bonus=1.0), floor=2.0, ceiling=8.0)
    team = _team(list(range(1, 16)), list(range(1, 12)))
    result = rank_captaincy_pool(team, [projection])
    reasoning = result.ranked_pool[0].reasoning
    assert "goals" in reasoning
    assert "floor" in reasoning
    assert "ceiling" in reasoning
