"""Tests for features/chips.py — per-chip value-now-vs-waiting evaluators (BUILD_PLAN Phase 4)."""

from __future__ import annotations

import pytest

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection, PlayerHorizonProjection
from features.chips import (
    FREE_HIT_BLOCKED_GAMEWEEKS,
    bench_points_by_gameweek,
    best_eligible_points_by_gameweek,
    blank_exposure_by_gameweek,
    evaluate_bench_boost,
    evaluate_free_hit,
    evaluate_triple_captain,
    evaluate_wildcard,
)
from features.fixtures import TeamFixture
from features.team_state import MyTeamState, SquadPlayer

MINUTES = MinutesDistribution(
    p_zero=0.05,
    p_1_to_59=0.10,
    p_60_plus=0.85,
    expected_minutes_given_1_to_59=30.0,
    expected_minutes_given_60_plus=88.0,
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


def _horizon(player_id: int, position: str, per_gw_points: dict[int, float]):
    gameweeks = {
        gw: PlayerGameweekProjection(
            player_id=player_id,
            position=position,
            gameweek=gw,
            minutes=MINUTES,
            breakdown=_breakdown(points),
        )
        for gw, points in per_gw_points.items()
    }
    return PlayerHorizonProjection(player_id=player_id, position=position, gameweeks=gameweeks)


def _squad_player(player_id: int, position: str) -> SquadPlayer:
    return SquadPlayer(player_id=player_id, position=position, purchase_price=50, current_price=50)


def _team() -> MyTeamState:
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    squad = tuple(_squad_player(i, pos) for i, pos in enumerate(positions, start=1))
    return MyTeamState(
        squad=squad,
        starting_xi=tuple(range(1, 12)),
        bench_order=tuple(range(12, 16)),
        captain_id=1,
        vice_captain_id=2,
        bank=0,
        free_transfers=1,
        chips_remaining=frozenset(),
    )


# --- bench boost ---------------------------------------------------------------------------


def test_bench_points_by_gameweek_sums_only_bench_players():
    team = _team()
    projections = {
        1: _horizon(1, "GK", {1: 5.0}),  # starting XI, excluded
        12: _horizon(12, "GK", {1: 2.0, 2: 3.0}),
        13: _horizon(13, "DEF", {1: 1.0, 2: 1.0}),
    }
    totals = bench_points_by_gameweek(team, projections)
    assert totals == {1: 3.0, 2: 4.0}


def test_evaluate_bench_boost_recommends_play_now_at_peak_week():
    team = _team()
    projections = {12: _horizon(12, "GK", {1: 2.0, 2: 8.0})}
    result = evaluate_bench_boost(team, projections, target_gameweek=2)
    assert result.recommendation == "play_now"
    assert result.best_gameweek == 2


def test_evaluate_bench_boost_recommends_wait_when_better_week_exists():
    team = _team()
    projections = {12: _horizon(12, "GK", {1: 2.0, 2: 8.0})}
    result = evaluate_bench_boost(team, projections, target_gameweek=1)
    assert result.recommendation == "wait"
    assert result.best_gameweek == 2
    assert result.best_value == pytest.approx(8.0)


# --- triple captain -------------------------------------------------------------------------


def test_best_eligible_points_by_gameweek_takes_max_across_starting_xi():
    team = _team()
    projections = {
        1: _horizon(1, "GK", {1: 3.0, 2: 4.0}),
        2: _horizon(2, "DEF", {1: 9.0, 2: 2.0}),
        13: _horizon(13, "DEF", {1: 100.0, 2: 100.0}),  # bench, excluded
    }
    bests = best_eligible_points_by_gameweek(team, projections)
    assert bests == {1: 9.0, 2: 4.0}


def test_evaluate_triple_captain_picks_peak_gameweek():
    team = _team()
    projections = {1: _horizon(1, "GK", {1: 5.0, 2: 12.0, 3: 6.0})}
    result = evaluate_triple_captain(team, projections, target_gameweek=1)
    assert result.recommendation == "wait"
    assert result.best_gameweek == 2
    assert result.best_value == pytest.approx(12.0)


# --- free hit --------------------------------------------------------------------------------


def test_blank_exposure_by_gameweek_counts_squad_players_on_blanking_teams():
    team = _team()
    team_id_by_player = dict.fromkeys(range(1, 16), 10)  # whole squad on team 10
    fixtures = [TeamFixture(team_id=10, opponent_id=99, gameweek=2, is_home=True)]
    counts = blank_exposure_by_gameweek(team, team_id_by_player, fixtures, gameweeks=[1, 2])
    assert counts == {1: 15, 2: 0}  # GW1 blank for the whole squad, GW2 has a fixture


def test_evaluate_free_hit_rejects_blocked_gameweek():
    team = _team()
    with pytest.raises(ValueError):
        evaluate_free_hit(
            team, {}, [], gameweeks=[1, 2], target_gameweek=FREE_HIT_BLOCKED_GAMEWEEKS[0]
        )


def test_evaluate_free_hit_recommends_worst_blank_gameweek():
    team = _team()
    team_id_by_player = dict.fromkeys(range(1, 16), 10)
    fixtures = [
        TeamFixture(team_id=10, opponent_id=99, gameweek=2, is_home=True),
        TeamFixture(team_id=10, opponent_id=98, gameweek=3, is_home=True),
    ]
    result = evaluate_free_hit(
        team, team_id_by_player, fixtures, gameweeks=[2, 3, 4], target_gameweek=2
    )
    assert result.recommendation == "wait"
    assert result.best_gameweek == 4  # GW4 has no fixture at all for the squad's team


# --- wildcard --------------------------------------------------------------------------------


def test_evaluate_wildcard_recommends_hold_below_threshold():
    team = _team()
    current = {3: _horizon(3, "DEF", {1: 4.0, 2: 4.0})}
    pool = {101: _horizon(101, "DEF", {1: 4.5, 2: 4.5})}  # +1.0 total, below the materiality bar
    result = evaluate_wildcard(team, current, pool, buy_prices={101: 50})
    assert result.recommendation == "hold"


def test_evaluate_wildcard_recommends_play_now_above_threshold():
    team = _team()
    current = {3: _horizon(3, "DEF", {1: 2.0, 2: 2.0})}
    pool = {101: _horizon(101, "DEF", {1: 8.0, 2: 8.0})}  # +12.0 total, well above the bar
    result = evaluate_wildcard(team, current, pool, buy_prices={101: 50})
    assert result.recommendation == "play_now"
    assert result.squad_uplift == pytest.approx(12.0)
    assert result.upgradeable_slots == 1


def test_evaluate_wildcard_ignores_non_upgrades():
    team = _team()
    current = {3: _horizon(3, "DEF", {1: 8.0, 2: 8.0})}
    pool = {101: _horizon(101, "DEF", {1: 2.0, 2: 2.0})}  # worse, not counted
    result = evaluate_wildcard(team, current, pool, buy_prices={101: 50})
    assert result.squad_uplift == pytest.approx(0.0)
    assert result.upgradeable_slots == 0
    assert result.recommendation == "hold"
