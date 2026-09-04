"""Tests for features/player_stats.py -- summing actual stats/points over a gameweek range
(PLAYER_STATS_PLAN D2/D12/D13)."""

from __future__ import annotations

import math

import pytest

from engine.data.player_history import PlayerGameweekActual
from engine.scoring import DEF, FWD, MID
from features.player_stats import (
    SMALL_SAMPLE_APPS_THRESHOLD,
    build_actual_stats_by_player,
    expected_clean_sheets,
    fit_overperformance_priors,
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


def test_summarize_sums_defensive_contribution_as_a_top_level_raw_stat():
    """Defensive contribution must be readable as its own summed value, the same way bonus is --
    not only reachable indirectly via points_breakdown -- since the Player Stats page needs a
    top-level field to expose it as a real column rather than burying it in the points popover."""
    history = [
        _actual(1, defensive_contribution=2, total_points=4),
        _actual(2, defensive_contribution=0, total_points=2),
        _actual(3, defensive_contribution=2, total_points=4),  # outside range, must be excluded
    ]

    result = summarize_actual_stats(history, DEF, gameweek_from=1, gameweek_to=2)

    assert result.defensive_contribution == 2
    assert result.points_breakdown.defensive_contribution == 2.0


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
        [_actual(1)], FWD, gameweek_from=1, gameweek_to=1, ownership_percent=42.5
    )

    assert result.ownership_percent == 42.5


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

    assert result[1].ownership_percent == 10.0


# --- Overperformance ratios --------------------------------------------------------------------


def _scorer(goals: list[int], xgi_per_match: float) -> list:
    return [
        _actual(
            gameweek,
            goals_scored=goals[gameweek - 1],
            expected_goal_involvements=xgi_per_match,
        )
        for gameweek in range(1, len(goals) + 1)
    ]


def test_defensive_ratio_counts_clean_sheets_not_mean_goals_conceded():
    """The regression this whole design exists to prevent. Clean sheet points are a step function,
    so a defender conceding 0,0,0,6 (three clean sheets) must rank above one conceding 1,1,1,1
    (none), even though the first has the *worse* mean goals conceded per 90 (1.5 versus 1.0)."""
    lumpy = [
        _actual(
            gw,
            clean_sheets=1 if conceded == 0 else 0,
            goals_conceded=conceded,
            expected_goals_conceded=1.2,
        )
        for gw, conceded in enumerate([0, 0, 0, 6], start=1)
    ]
    steady = [
        _actual(gw, clean_sheets=0, goals_conceded=1, expected_goals_conceded=1.2)
        for gw in range(1, 5)
    ]
    history = {1: lumpy, 2: steady}
    positions = {1: DEF, 2: DEF}

    result = build_actual_stats_by_player(history, positions, 1, 4, full_season_history=history)

    assert result[1].defensive_ratio is not None
    assert result[2].defensive_ratio is not None
    assert result[1].defensive_ratio.ratio > result[2].defensive_ratio.ratio
    # And the mean-based metric this replaced would have ranked them the other way round.
    assert result[1].goals_conceded > result[2].goals_conceded


def test_defensive_ratio_is_none_for_forwards_but_present_for_midfielders():
    """A clean sheet pays a midfielder 1 point and a forward nothing, so only the forward has no
    defensive ratio to show."""
    history = {
        1: [_actual(1, clean_sheets=1, expected_goals_conceded=1.0)],
        2: [_actual(1, clean_sheets=1, expected_goals_conceded=1.0)],
    }
    positions = {1: FWD, 2: MID}

    result = build_actual_stats_by_player(history, positions, 1, 1, full_season_history=history)

    assert result[1].defensive_ratio is None
    assert result[2].defensive_ratio is not None


def test_ratios_stay_unpopulated_without_a_season_to_fit_priors_on():
    history = {1: [_actual(1, goals_scored=1, expected_goal_involvements=0.5)]}

    result = build_actual_stats_by_player(history, {1: FWD}, 1, 1)

    assert result[1].attacking_ratio is None
    assert result[1].defensive_ratio is None


def test_priors_are_fitted_per_position():
    """Each position gets its own prior fitted from its own players, so a position whose players
    genuinely differ from one another can report real spread while one whose players are
    interchangeable correctly reports none."""
    history: dict[int, list] = {}
    positions: dict[int, str] = {}

    # Forwards who genuinely differ: some convert well above their chances, some well below.
    for i in range(40):
        goals = [3, 3, 3, 3, 3, 3] if i % 2 else [0, 0, 0, 0, 0, 0]
        history[i] = _scorer(goals, xgi_per_match=0.6)
        positions[i] = FWD
    # Defenders who are all identical: nothing to tell apart beyond chance.
    for i in range(40, 80):
        history[i] = _scorer([0, 0, 0, 0, 0, 0], xgi_per_match=0.12)
        positions[i] = DEF

    attacking_k, _defensive_k = fit_overperformance_priors(history, positions)

    assert set(attacking_k) == {FWD, DEF}
    # Real spread among the forwards yields a finite prior, so their own rates carry weight.
    assert math.isfinite(attacking_k[FWD])
    # No detectable spread among the defenders, so every defender collapses to exactly 1.0.
    assert attacking_k[DEF] == float("inf")


def test_expected_clean_sheets_sums_per_match_probability():
    records = [
        _actual(1, expected_goals_conceded=0.0),  # certain clean sheet -> exp(0) == 1
        _actual(2, expected_goals_conceded=1.0),  # exp(-1) ~ 0.368
    ]

    assert expected_clean_sheets(records) == pytest.approx(1.0 + math.exp(-1.0))


def test_expected_clean_sheets_skips_matches_the_player_did_not_play():
    records = [_actual(1, minutes=0, expected_goals_conceded=0.0)]

    assert expected_clean_sheets(records) == 0.0


def test_penalty_taker_flag_is_carried_through():
    history = {1: [_actual(1)], 2: [_actual(1)]}
    positions = {1: FWD, 2: FWD}

    result = build_actual_stats_by_player(history, positions, 1, 1, penalty_takers=frozenset({1}))

    assert result[1].is_penalty_taker is True
    assert result[2].is_penalty_taker is False
