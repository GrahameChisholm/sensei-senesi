"""Tests for engine/models/goals.py — xG-based scoring rate + penalty sub-model (2.2)."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.models.goals import (
    DEFAULT_NPXG_SHARE_OF_TEAM_XG,
    DEFAULT_PENALTY_CONVERSION_RATE,
    GoalProjection,
    expected_non_penalty_goal_rate,
    expected_team_penalties,
    fit_penalty_conversion_rates,
    penalty_goals_and_misses,
    prior_npxg_rate_from_team_xg,
    project_goals,
    realized_penalty_goals,
    shrunk_player_npxg_per_90,
)


def test_expected_non_penalty_goal_rate_matches_formula():
    rate = expected_non_penalty_goal_rate(
        player_npxg_per_90=0.5,
        opponent_xga_per_90=1.6,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
    )
    assert rate == pytest.approx(0.5 * (1.6 / 1.4) * 1.0)


def test_expected_non_penalty_goal_rate_scales_with_minutes():
    full = expected_non_penalty_goal_rate(0.5, 1.4, 1.4, expected_minutes=90.0)
    half = expected_non_penalty_goal_rate(0.5, 1.4, 1.4, expected_minutes=45.0)
    assert half == pytest.approx(full / 2)


def test_expected_non_penalty_goal_rate_harder_fixture_lowers_rate():
    easy_fixture = expected_non_penalty_goal_rate(0.5, 2.0, 1.4, expected_minutes=90.0)
    hard_fixture = expected_non_penalty_goal_rate(0.5, 0.8, 1.4, expected_minutes=90.0)
    assert hard_fixture < easy_fixture


def test_expected_non_penalty_goal_rate_rejects_non_positive_league_average():
    with pytest.raises(ValueError):
        expected_non_penalty_goal_rate(0.5, 1.4, 0.0, 90.0)


def test_expected_non_penalty_goal_rate_rejects_negative_inputs():
    with pytest.raises(ValueError):
        expected_non_penalty_goal_rate(-0.1, 1.4, 1.4, 90.0)


def test_expected_non_penalty_goal_rate_rejects_nan_inputs():
    # ENGINE_IMPROVEMENTS_2.md C.2: a NaN rate (e.g. from an unmatched ID-crosswalk player) must
    # raise loudly here rather than silently produce a NaN expected_points downstream.
    with pytest.raises(ValueError):
        expected_non_penalty_goal_rate(float("nan"), 1.4, 1.4, 90.0)
    with pytest.raises(ValueError):
        expected_non_penalty_goal_rate(0.5, float("nan"), 1.4, 90.0)
    with pytest.raises(ValueError):
        expected_non_penalty_goal_rate(0.5, 1.4, 1.4, float("nan"))


def test_expected_team_penalties_matches_formula():
    result = expected_team_penalties(
        team_penalty_win_rate_per_game=0.2, opponent_xga_per_90=1.6, league_avg_xga_per_90=1.4
    )
    assert result == pytest.approx(0.2 * (1.6 / 1.4))


def test_penalty_goals_and_misses_split_by_conversion_rate():
    outcome = penalty_goals_and_misses(
        team_expected_penalties=0.3, taker_share=1.0, conversion_rate=0.8
    )
    assert outcome.expected_penalty_goals == pytest.approx(0.3 * 0.8)
    assert outcome.expected_penalty_misses == pytest.approx(0.3 * 0.2)


def test_penalty_goals_and_misses_non_taker_gets_nothing():
    outcome = penalty_goals_and_misses(team_expected_penalties=0.3, taker_share=0.0)
    assert outcome.expected_penalty_goals == 0.0
    assert outcome.expected_penalty_misses == 0.0


def test_penalty_goals_and_misses_rejects_out_of_range_shares():
    with pytest.raises(ValueError):
        penalty_goals_and_misses(0.3, taker_share=1.5)
    with pytest.raises(ValueError):
        penalty_goals_and_misses(0.3, taker_share=0.5, conversion_rate=1.5)


def test_goal_projection_expected_goals_sums_open_play_and_penalty():
    projection = GoalProjection(
        non_penalty_goal_rate=0.4, expected_penalty_goals=0.1, expected_penalty_misses=0.03
    )
    assert projection.expected_goals == pytest.approx(0.5)


def test_goal_projection_expected_points_is_position_weighted():
    projection = GoalProjection(
        non_penalty_goal_rate=0.4, expected_penalty_goals=0.1, expected_penalty_misses=0.03
    )
    fwd_points = projection.expected_points("FWD")
    def_points = projection.expected_points("DEF")
    # Same underlying goal expectation, but DEF goals are worth more (6 vs 4).
    assert def_points > fwd_points
    assert fwd_points == pytest.approx(0.5 * 4 + 0.03 * -2)


def test_goal_projection_rejects_unknown_position():
    projection = GoalProjection(0.4, 0.1, 0.03)
    with pytest.raises(ValueError):
        projection.expected_points("XYZ")


def test_goal_projection_rejects_negative_fields():
    with pytest.raises(ValueError):
        GoalProjection(
            non_penalty_goal_rate=-0.1, expected_penalty_goals=0.0, expected_penalty_misses=0.0
        )


def test_project_goals_end_to_end_with_penalties():
    projection = project_goals(
        player_npxg_per_90=0.5,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
        team_expected_penalties=0.25,
        taker_share=1.0,
    )
    assert projection.non_penalty_goal_rate == pytest.approx(0.5)
    expected_penalty_goals = 0.25 * DEFAULT_PENALTY_CONVERSION_RATE
    assert projection.expected_penalty_goals == pytest.approx(expected_penalty_goals)
    assert projection.expected_goals > projection.non_penalty_goal_rate


def test_project_goals_default_no_penalty_role():
    projection = project_goals(0.5, 1.4, 1.4, 90.0)
    assert projection.expected_penalty_goals == 0.0
    assert projection.expected_penalty_misses == 0.0


def test_prior_npxg_rate_from_team_xg_scales_with_share():
    prior = prior_npxg_rate_from_team_xg(team_xg_per_90=1.5, npxg_share=0.1)
    assert prior == pytest.approx(0.15)


def test_prior_npxg_rate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        prior_npxg_rate_from_team_xg(-1.0)
    with pytest.raises(ValueError):
        prior_npxg_rate_from_team_xg(1.0, npxg_share=1.5)


def test_shrunk_player_npxg_thin_sample_moves_toward_team_prior():
    shrunk = shrunk_player_npxg_per_90(
        player_npxg_per_90=2.5,  # a physically-implausible outlier from a single cameo
        individual_weight=5.0,
        team_xg_per_90=1.5,
        shrinkage_k=50.0,
        npxg_share=DEFAULT_NPXG_SHARE_OF_TEAM_XG,
    )
    prior = 1.5 * DEFAULT_NPXG_SHARE_OF_TEAM_XG
    assert shrunk < 2.5
    assert shrunk > prior


def test_shrunk_player_npxg_thick_sample_stays_near_individual():
    shrunk = shrunk_player_npxg_per_90(
        player_npxg_per_90=0.4,
        individual_weight=5000.0,
        team_xg_per_90=1.5,
        shrinkage_k=50.0,
    )
    assert shrunk == pytest.approx(0.4, abs=1e-2)


def test_project_goals_shrinkage_reduces_an_outlier_rate_when_activated():
    baseline = project_goals(
        player_npxg_per_90=3.0,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=20.0,
    )
    shrunk = project_goals(
        player_npxg_per_90=3.0,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=20.0,
        individual_weight=5.0,
        team_xg_per_90=1.5,
        shrinkage_k=180.0,
    )
    assert shrunk.non_penalty_goal_rate < baseline.non_penalty_goal_rate


def test_project_goals_shrinkage_not_applied_without_all_three_arguments():
    # Omitting any one of individual_weight/team_xg_per_90/shrinkage_k>0 must reproduce the
    # unmodified (pre-B.2) behavior exactly -- opt-in, not silently always-on.
    baseline = project_goals(0.5, 1.4, 1.4, 90.0)
    only_weight = project_goals(0.5, 1.4, 1.4, 90.0, individual_weight=5.0)
    only_team_xg = project_goals(0.5, 1.4, 1.4, 90.0, team_xg_per_90=1.5)
    zero_shrinkage_k = project_goals(
        0.5, 1.4, 1.4, 90.0, individual_weight=5.0, team_xg_per_90=1.5, shrinkage_k=0.0
    )
    for variant in (only_weight, only_team_xg, zero_shrinkage_k):
        assert variant.non_penalty_goal_rate == pytest.approx(baseline.non_penalty_goal_rate)


def test_realized_penalty_goals_is_goals_minus_non_penalty_goals():
    matches = pd.DataFrame({"goals": [2, 1, 0, 3], "npg": [1, 1, 0, 1]})

    result = realized_penalty_goals(matches)

    assert list(result) == [1, 0, 0, 2]


def test_realized_penalty_goals_clips_at_zero():
    # npg should never exceed goals in real data, but a data-quality edge case shouldn't produce
    # a negative "realized penalty goals" value.
    matches = pd.DataFrame({"goals": [1], "npg": [2]})

    result = realized_penalty_goals(matches)

    assert list(result) == [0]


def test_fit_penalty_conversion_rates_empty_returns_defaults():
    rates, league_avg = fit_penalty_conversion_rates(pd.DataFrame(columns=["player_id", "scored"]))
    assert rates == {}
    assert league_avg == DEFAULT_PENALTY_CONVERSION_RATE


def test_fit_penalty_conversion_rates_shrinks_thin_samples_toward_league_average():
    # Player 1: a large, consistent sample scoring 9/10 -- should stay close to its own rate.
    # Player 2: a single attempt, missed -- should shrink heavily toward the league average rather
    # than being taken at face value as a 0% converter.
    attempts = pd.DataFrame(
        {
            "player_id": [1] * 10 + [2],
            "scored": [1] * 9 + [0] + [0],
        }
    )

    rates, league_avg = fit_penalty_conversion_rates(attempts, shrinkage_k=8.0)

    assert rates[1] == pytest.approx(0.9, abs=0.05)
    assert rates[2] > 0.0  # shrunk toward the league average, not left at a raw 0%
    assert rates[2] < rates[1]
    assert league_avg == pytest.approx(attempts["scored"].mean())
