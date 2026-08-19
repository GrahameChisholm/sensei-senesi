"""Tests for features/player_stats.py -- summing actual stats/points over a gameweek range
(PLAYER_STATS_PLAN D2/D12/D13)."""

from __future__ import annotations

from engine.data.player_history import PlayerGameweekActual
from engine.scoring import DEF, FWD
from features.player_stats import (
    SMALL_SAMPLE_APPS_THRESHOLD,
    build_actual_stats_by_player,
    summarize_actual_stats,
)


def _actual(gameweek: int, **overrides) -> PlayerGameweekActual:
    base = dict(
        gameweek=gameweek,
        minutes=90,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        saves=0,
        yellow_cards=0,
        red_cards=0,
        bonus=0,
        defensive_contribution=0,
        total_points=2,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        expected_goals_conceded=0.0,
    )
    base.update(overrides)
    return PlayerGameweekActual(**base)


def test_summarize_sums_raw_counts_across_the_range():
    history = [
        _actual(1, goals_scored=1, minutes=90, total_points=8),
        _actual(2, goals_scored=1, minutes=90, total_points=8),
        _actual(3, goals_scored=1, minutes=90, total_points=8),  # outside range, must be excluded
    ]

    result = summarize_actual_stats(history, FWD, gameweek_from=1, gameweek_to=2)

    assert result.goals_scored == 2
    assert result.minutes == 180
    assert result.total_points == 16
    assert result.gameweek_from == 1
    assert result.gameweek_to == 2


def test_summarize_converts_points_per_gameweek_before_summing():
    """A goal in each of two separate gameweeks scores GOAL_POINTS twice, matching the prediction
    engine's own per-gameweek treatment -- points_breakdown.goals must reflect summed per-gameweek
    conversion, not a single conversion of the summed raw total (they happen to agree for goals,
    but the goals-conceded penalty test below shows why the order matters in general)."""
    history = [_actual(1, goals_scored=1), _actual(2, goals_scored=1)]

    result = summarize_actual_stats(history, FWD, gameweek_from=1, gameweek_to=2)

    assert result.points_breakdown.goals == 8.0  # 2 goals * 4 points (FWD)


def test_summarize_goals_conceded_penalty_computed_per_gameweek_not_summed_first():
    """G4's core claim, exercised through the summarizer: conceding 1 goal in each of two
    gameweeks must score 0 penalty total, not -1 (which summing raw goals_conceded to 2 first and
    applying the per-2 penalty once would wrongly produce)."""
    history = [_actual(1, goals_conceded=1), _actual(2, goals_conceded=1)]

    result = summarize_actual_stats(history, DEF, gameweek_from=1, gameweek_to=2)

    assert result.points_breakdown.goals_conceded == 0.0
    assert result.goals_conceded == 2  # the raw total is still correctly 2


def test_summarize_returns_none_when_no_gameweek_falls_in_range():
    history = [_actual(1), _actual(2)]

    result = summarize_actual_stats(history, FWD, gameweek_from=5, gameweek_to=6)

    assert result is None


def test_summarize_apps_counts_only_gameweeks_with_minutes_played():
    history = [_actual(1, minutes=90), _actual(2, minutes=0), _actual(3, minutes=45)]

    result = summarize_actual_stats(history, FWD, gameweek_from=1, gameweek_to=3)

    assert result.apps == 2


def test_summarize_small_sample_flips_at_the_threshold():
    below_threshold = [_actual(gw, minutes=90) for gw in range(1, SMALL_SAMPLE_APPS_THRESHOLD)]
    at_threshold = [_actual(gw, minutes=90) for gw in range(1, SMALL_SAMPLE_APPS_THRESHOLD + 1)]

    result_below = summarize_actual_stats(below_threshold, FWD, gameweek_from=1, gameweek_to=38)
    result_at = summarize_actual_stats(at_threshold, FWD, gameweek_from=1, gameweek_to=38)

    assert result_below.small_sample is True
    assert result_at.small_sample is False


def test_summarize_carries_ownership_through():
    result = summarize_actual_stats(
        [_actual(1)], FWD, gameweek_from=1, gameweek_to=1, selected_by_percent=42.5
    )

    assert result.selected_by_percent == 42.5


# --- build_actual_stats_by_player ------------------------------------------------------------


def test_build_actual_stats_by_player_excludes_players_with_no_data_in_range():
    player_history = {1: [_actual(1)], 2: [_actual(5)]}
    position_by_player = {1: "FWD", 2: "FWD"}

    result = build_actual_stats_by_player(player_history, position_by_player, 1, 1)

    assert set(result) == {1}


def test_build_actual_stats_by_player_skips_players_missing_a_position():
    player_history = {1: [_actual(1)]}
    position_by_player: dict[int, str] = {}

    result = build_actual_stats_by_player(player_history, position_by_player, 1, 1)

    assert result == {}


def test_build_actual_stats_by_player_attaches_ownership_per_player():
    player_history = {1: [_actual(1)]}
    position_by_player = {1: "FWD"}

    result = build_actual_stats_by_player(
        player_history, position_by_player, 1, 1, ownership_by_player={1: 10.0}
    )

    assert result[1].selected_by_percent == 10.0
