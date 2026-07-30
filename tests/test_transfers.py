"""Tests for features/transfers.py — greedy one-swap-at-a-time transfer comparator (BUILD_PLAN
Phase 4)."""

from __future__ import annotations

import pytest

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection, PlayerHorizonProjection
from features.team_state import MyTeamState, SquadPlayer
from features.transfers import (
    TRANSFER_HIT_COST,
    evaluate_transfer,
    find_transfer_candidates,
)

HEALTHY_MINUTES = MinutesDistribution(
    p_zero=0.05,
    p_1_to_59=0.10,
    p_60_plus=0.85,
    expected_minutes_given_1_to_59=30.0,
    expected_minutes_given_60_plus=88.0,
)
INJURED_MINUTES = MinutesDistribution(
    p_zero=0.95,
    p_1_to_59=0.05,
    p_60_plus=0.0,
    expected_minutes_given_1_to_59=10.0,
    expected_minutes_given_60_plus=0.0,
)


def _breakdown(total: float) -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=total,
        goals=0.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )


def _horizon(
    player_id: int,
    position: str,
    per_gw_points: list[float],
    minutes: MinutesDistribution = HEALTHY_MINUTES,
) -> PlayerHorizonProjection:
    gameweeks = {
        gw: PlayerGameweekProjection(
            player_id=player_id,
            position=position,
            gameweek=gw,
            minutes=minutes,
            breakdown=_breakdown(points),
        )
        for gw, points in enumerate(per_gw_points, start=1)
    }
    return PlayerHorizonProjection(player_id=player_id, position=position, gameweeks=gameweeks)


def _squad_player(player_id: int, position: str, purchase: int = 50, current: int = 50):
    return SquadPlayer(
        player_id=player_id, position=position, purchase_price=purchase, current_price=current
    )


def _team(free_transfers: int = 1, bank: int = 10) -> MyTeamState:
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad = tuple(_squad_player(i, pos) for i, pos in enumerate(positions, start=1))
    return MyTeamState(
        squad=squad,
        starting_xi=tuple(range(1, 12)),
        bench_order=tuple(range(12, 16)),
        captain_id=1,
        vice_captain_id=2,
        bank=bank,
        free_transfers=free_transfers,
        chips_remaining=frozenset(),
    )


# --- evaluate_transfer -----------------------------------------------------------------------


def test_evaluate_transfer_rejects_position_mismatch():
    team = _team()
    sold = _horizon(3, "DEF", [4.0] * 5)
    bought = _horizon(100, "MID", [5.0] * 5)
    with pytest.raises(ValueError):
        evaluate_transfer(team, 3, 100, 55, sold, bought)


def test_evaluate_transfer_no_hit_when_free_transfer_available():
    team = _team(free_transfers=1)
    sold = _horizon(3, "DEF", [4.0] * 5)  # 20.0 total
    bought = _horizon(100, "DEF", [5.0] * 5)  # 25.0 total
    candidate = evaluate_transfer(team, 3, 100, 55, sold, bought)
    assert candidate.hit_cost == 0
    assert candidate.points_gain == pytest.approx(5.0)
    assert candidate.net_points_gain == pytest.approx(5.0)


def test_evaluate_transfer_applies_hit_when_no_free_transfer():
    team = _team(free_transfers=0)
    sold = _horizon(3, "DEF", [4.0] * 5)
    bought = _horizon(100, "DEF", [5.0] * 5)
    candidate = evaluate_transfer(team, 3, 100, 55, sold, bought)
    assert candidate.hit_cost == TRANSFER_HIT_COST
    assert candidate.net_points_gain == pytest.approx(5.0 - TRANSFER_HIT_COST)


def test_evaluate_transfer_uses_sell_price_not_current_price():
    team = _team()  # player 3 has purchase=current=50 -> sell_price=50
    sold = _horizon(3, "DEF", [4.0] * 5)
    bought = _horizon(100, "DEF", [5.0] * 5)
    candidate = evaluate_transfer(
        team, 3, 100, buy_price=60, sold_horizon=sold, bought_horizon=bought
    )
    assert candidate.sell_price == 50
    assert candidate.net_spend == 10


def test_evaluate_transfer_flags_forced_when_sold_player_likely_unavailable():
    team = _team()
    sold = _horizon(3, "DEF", [0.0] * 5, minutes=INJURED_MINUTES)
    bought = _horizon(100, "DEF", [4.0] * 5)
    candidate = evaluate_transfer(team, 3, 100, 55, sold, bought)
    assert candidate.is_forced
    assert "forced" in candidate.reasoning


def test_evaluate_transfer_not_forced_for_healthy_player():
    team = _team()
    sold = _horizon(3, "DEF", [4.0] * 5)
    bought = _horizon(100, "DEF", [4.5] * 5)
    candidate = evaluate_transfer(team, 3, 100, 55, sold, bought)
    assert not candidate.is_forced


# --- find_transfer_candidates ----------------------------------------------------------------


def test_find_transfer_candidates_filters_by_position_and_ownership():
    team = _team(bank=100)
    current = {3: _horizon(3, "DEF", [4.0] * 5)}
    pool = {
        100: _horizon(100, "MID", [10.0] * 5),  # wrong position, excluded
        3: _horizon(3, "DEF", [4.0] * 5),  # already owned, excluded
        101: _horizon(101, "DEF", [6.0] * 5),  # valid candidate
    }
    plan = find_transfer_candidates(team, current, pool, buy_prices={100: 50, 101: 50})
    assert {c.buy_player_id for c in plan.affordable_candidates} == {101}


def test_find_transfer_candidates_excludes_unaffordable_buys():
    team = _team(bank=0)  # player 3 sell_price=50, bank=0 -> budget=50
    current = {3: _horizon(3, "DEF", [4.0] * 5)}
    pool = {101: _horizon(101, "DEF", [8.0] * 5)}
    plan = find_transfer_candidates(team, current, pool, buy_prices={101: 51})
    assert plan.affordable_candidates == ()
    assert plan.recommended is None


def test_find_transfer_candidates_recommends_best_net_positive_upgrade():
    team = _team(bank=100)
    current = {3: _horizon(3, "DEF", [4.0] * 5)}  # 20.0 total
    pool = {
        101: _horizon(101, "DEF", [4.2] * 5),  # marginal upgrade: +1.0
        102: _horizon(102, "DEF", [8.0] * 5),  # big upgrade: +20.0
    }
    plan = find_transfer_candidates(team, current, pool, buy_prices={101: 50, 102: 50})
    assert plan.recommended.buy_player_id == 102


def test_find_transfer_candidates_no_recommendation_when_nothing_is_an_upgrade():
    team = _team(bank=100)
    current = {3: _horizon(3, "DEF", [8.0] * 5)}
    pool = {101: _horizon(101, "DEF", [4.0] * 5)}
    plan = find_transfer_candidates(team, current, pool, buy_prices={101: 50})
    assert plan.recommended is None
    assert len(plan.affordable_candidates) == 1


def test_find_transfer_candidates_prioritises_forced_sell_even_if_not_top_net_gain():
    team = _team(bank=100)
    current = {
        3: _horizon(3, "DEF", [0.0] * 5, minutes=INJURED_MINUTES),  # forced: must be moved on
        4: _horizon(4, "DEF", [4.0] * 5),  # healthy, big optional upgrade available
    }
    pool = {
        101: _horizon(101, "DEF", [2.0] * 5),  # small forced-replacement upgrade for player 3
        102: _horizon(102, "DEF", [9.0] * 5),  # huge optional upgrade for player 4
    }
    plan = find_transfer_candidates(team, current, pool, buy_prices={101: 50, 102: 50})
    # Even though replacing player 4 nets far more points, the forced sale takes priority.
    assert plan.recommended.sell_player_id == 3
    assert plan.recommended.is_forced
