"""Tests for accuracy, bias, calibration, and captaincy hit-rate metrics (3.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    bias_by_group,
    captaincy_hit_rate,
    component_calibration,
    floor_ceiling_coverage,
    mean_calibration,
    minutes_model_diagnostics,
    player_accuracy,
    rank_correlation,
    top_n_mean_actual,
)


def test_player_accuracy_overall_and_per_position():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["MID", "MID", "FWD", "FWD"],
            "gameweek": [1, 1, 1, 1],
            "expected_points": [5.0, 3.0, 6.0, 2.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "total_points": [4.0, 3.0, 10.0, 2.0],
        }
    )

    report = player_accuracy(predictions, actuals)

    assert report.overall_mae == pytest.approx((1 + 0 + 4 + 0) / 4)
    by_position = report.by_position.set_index("position")
    assert by_position.loc["MID", "mae"] == pytest.approx(0.5)
    assert by_position.loc["FWD", "mae"] == pytest.approx(2.0)
    assert by_position.loc["MID", "n"] == 2


def test_player_accuracy_raises_when_no_overlap():
    predictions = pd.DataFrame(
        {"player_id": [1], "position": ["MID"], "gameweek": [1], "expected_points": [5.0]}
    )
    actuals = pd.DataFrame({"player_id": [1], "gameweek": [2], "total_points": [4.0]})

    with pytest.raises(ValueError):
        player_accuracy(predictions, actuals)


def test_bias_by_group_flags_severe_systematic_overrating():
    rng = np.random.default_rng(0)
    n = 50
    # Predicted is systematically ~3 points too high for this group -- a real, severe bias.
    predicted = 10.0 + rng.normal(0, 0.1, n)
    actual = 7.0 + rng.normal(0, 0.1, n)
    predictions = pd.DataFrame(
        {
            "player_id": range(n),
            "position": ["FWD"] * n,
            "gameweek": [1] * n,
            "expected_points": predicted,
        }
    )
    actuals = pd.DataFrame({"player_id": range(n), "gameweek": [1] * n, "total_points": actual})

    report = bias_by_group(predictions, actuals, group_col="position")

    row = report.by_group.set_index("position").loc["FWD"]
    assert row["mean_residual"] == pytest.approx(3.0, abs=0.2)
    assert row["severe"]


def test_bias_by_group_significant_but_small_effect_not_flagged_severe():
    # Reproduces ENGINE_IMPROVEMENTS.md Correction 2: a small (~4% of mean actual) but, at large n,
    # statistically significant bias should NOT be flagged severe once an effect-size floor applies.
    rng = np.random.default_rng(3)
    n = 2000
    actual = rng.normal(1.8, 1.0, n)
    predicted = actual - 0.07 + rng.normal(0, 1.0, n)
    predictions = pd.DataFrame(
        {
            "player_id": range(n),
            "position": ["MID"] * n,
            "gameweek": [1] * n,
            "expected_points": predicted,
        }
    )
    actuals = pd.DataFrame({"player_id": range(n), "gameweek": [1] * n, "total_points": actual})

    report = bias_by_group(predictions, actuals, group_col="position")

    row = report.by_group.set_index("position").loc["MID"]
    assert row["p_value"] < 0.01  # genuinely statistically significant
    assert not row["severe"]  # but too small to matter


def test_bias_by_group_significant_and_large_effect_flagged_severe():
    rng = np.random.default_rng(4)
    n = 2000
    actual = rng.normal(1.8, 1.0, n)
    predicted = actual - 1.0 + rng.normal(0, 1.0, n)
    predictions = pd.DataFrame(
        {
            "player_id": range(n),
            "position": ["FWD"] * n,
            "gameweek": [1] * n,
            "expected_points": predicted,
        }
    )
    actuals = pd.DataFrame({"player_id": range(n), "gameweek": [1] * n, "total_points": actual})

    report = bias_by_group(predictions, actuals, group_col="position")

    row = report.by_group.set_index("position").loc["FWD"]
    assert row["p_value"] < 0.01
    assert row["severe"]


def test_bias_by_group_no_bias_for_unbiased_predictions():
    rng = np.random.default_rng(1)
    n = 200
    actual = rng.normal(5, 1.0, n)
    predicted = actual + rng.normal(0, 1.0, n)  # noisy but unbiased
    predictions = pd.DataFrame(
        {
            "player_id": range(n),
            "position": ["MID"] * n,
            "gameweek": [1] * n,
            "expected_points": predicted,
        }
    )
    actuals = pd.DataFrame({"player_id": range(n), "gameweek": [1] * n, "total_points": actual})

    report = bias_by_group(predictions, actuals, group_col="position")

    assert not report.by_group.set_index("position").loc["MID", "severe"]


def test_component_calibration_perfectly_calibrated_probabilities():
    rng = np.random.default_rng(2)
    n = 5000
    predicted = rng.uniform(0, 1, n)
    actual = rng.uniform(0, 1, n) < predicted  # actual frequency matches predicted probability

    report = component_calibration(pd.Series(predicted), pd.Series(actual.astype(float)), n_bins=10)

    assert report.mean_absolute_calibration_error < 0.05


def test_component_calibration_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        component_calibration(pd.Series([0.1, 0.2]), pd.Series([1.0]))


def test_captaincy_hit_rate_raw_and_played_as_expected():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 1, 2],
            "gameweek": [1, 1, 2, 2],
            "expected_points": [8.0, 5.0, 8.0, 5.0],
            "expected_minutes": [90.0, 90.0, 90.0, 90.0],
        }
    )
    # GW1: recommended (1) actually scores highest and plays full match -> hit, played as expected.
    # GW2: recommended (1) is subbed off after 20 minutes and scores low; player 2 outscores -> a
    # miss, and not "played as expected" since player 1 didn't get near their expected minutes.
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 1, 2],
            "gameweek": [1, 1, 2, 2],
            "total_points": [10.0, 2.0, 1.0, 6.0],
            "minutes": [90, 90, 20, 90],
        }
    )
    starting_xi_by_gameweek = {1: {1, 2}, 2: {1, 2}}

    result = captaincy_hit_rate(predictions, actuals, starting_xi_by_gameweek)

    assert result.raw_hit_rate == pytest.approx(0.5)
    per_gw = result.per_gameweek.set_index("gameweek")
    assert per_gw.loc[1, "hit"]
    assert per_gw.loc[1, "played_as_expected"]
    assert not per_gw.loc[2, "hit"]
    assert not per_gw.loc[2, "played_as_expected"]
    # Restricted to the one played-as-expected gameweek (GW1), which was also a hit.
    assert result.played_as_expected_hit_rate == pytest.approx(1.0)


def test_captaincy_hit_rate_raises_when_no_overlapping_gameweeks():
    predictions = pd.DataFrame({"player_id": [1], "gameweek": [1], "expected_points": [5.0]})
    actuals = pd.DataFrame(
        {"player_id": [1], "gameweek": [1], "total_points": [5.0], "minutes": [90]}
    )

    with pytest.raises(ValueError):
        captaincy_hit_rate(predictions, actuals, starting_xi_by_gameweek={2: {1}})


def test_top_n_mean_actual_surfaces_skill_at_the_top():
    # Predicted ranking correctly identifies the two highest actual scorers per gameweek even
    # though the low end of the pool is noisy -- top-N should read that skill; a wide N wouldn't.
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4] * 2,
            "gameweek": [1] * 4 + [2] * 4,
            "expected_points": [9.0, 8.0, 2.0, 1.0, 9.0, 8.0, 2.0, 1.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4] * 2,
            "gameweek": [1] * 4 + [2] * 4,
            "total_points": [10.0, 6.0, 3.0, 0.0, 12.0, 4.0, 1.0, 2.0],
        }
    )

    report = top_n_mean_actual(predictions, actuals, ns=(1, 2, 4))

    by_n = report.by_n.set_index("n")
    assert by_n.loc[1, "mean_actual"] == pytest.approx((10.0 + 12.0) / 2)
    assert by_n.loc[2, "mean_actual"] == pytest.approx(((10.0 + 6.0) / 2 + (12.0 + 4.0) / 2) / 2)
    assert by_n.loc[4, "n_gameweeks"] == 2


def test_rank_correlation_overall_and_by_group_and_restricted_to_starters():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["MID", "MID", "FWD", "FWD"],
            "gameweek": [1, 1, 1, 1],
            "expected_points": [9.0, 1.0, 8.0, 2.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "total_points": [10.0, 0.0, 9.0, 1.0],
            "minutes": [90, 0, 90, 90],
        }
    )

    report = rank_correlation(predictions, actuals, group_col="position")
    assert report.overall == pytest.approx(1.0)
    by_group = report.by_group.set_index("position")
    assert by_group.loc["MID", "spearman"] == pytest.approx(1.0)
    assert by_group.loc["FWD", "spearman"] == pytest.approx(1.0)

    restricted = rank_correlation(predictions, actuals, minutes_col="minutes")
    # Pooled `overall` is unaffected by minutes_col -- it never gets silently replaced.
    assert restricted.overall == pytest.approx(1.0)
    # Player 2 (0 minutes) is dropped; the remaining 3 rows are still perfectly rank-correlated.
    assert restricted.overall_starters_only == pytest.approx(1.0)


def test_rank_correlation_pooled_and_starters_only_diverge_when_zero_minute_rows_hurt_pooled():
    # Players 1, 2, 4 (all starters) are perfectly rank-correlated. Player 3 is predicted highest
    # of all four (a rotation-risk player the model over-projected) but played 0 minutes and
    # scored nothing -- that single row wrecks the pooled ranking while starters-only stays clean
    # (Correction 5 / A.2).
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["MID", "MID", "FWD", "FWD"],
            "gameweek": [1, 1, 1, 1],
            "expected_points": [9.0, 5.0, 20.0, 8.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "total_points": [10.0, 6.0, 0.0, 8.0],
            "minutes": [90, 90, 0, 90],
        }
    )

    report = rank_correlation(predictions, actuals, minutes_col="minutes")

    assert report.overall != pytest.approx(report.overall_starters_only)
    assert report.overall_starters_only == pytest.approx(1.0)  # 3 starters, perfectly ranked


def test_mean_calibration_reports_gap_between_predicted_and_actual_means():
    predicted = pd.Series([0.5, 0.6, 0.4, 0.5])
    actual = pd.Series([0.3, 0.3, 0.3, 0.3])

    report = mean_calibration(predicted, actual)

    assert report.mean_predicted == pytest.approx(0.5)
    assert report.mean_actual == pytest.approx(0.3)
    assert report.absolute_gap == pytest.approx(0.2)
    assert report.relative_gap == pytest.approx(0.2 / 0.3)


def test_mean_calibration_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        mean_calibration(pd.Series([0.1, 0.2]), pd.Series([1.0]))


def test_minutes_model_diagnostics_scores_zero_minute_mass_and_auc():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "expected_points": [5.0, 1.0, 0.5, 6.0],
            "p_zero": [0.05, 0.9, 0.8, 0.02],
            "expected_minutes": [85.0, 5.0, 8.0, 88.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "minutes": [90, 0, 0, 90],
        }
    )

    report = minutes_model_diagnostics(predictions, actuals)

    assert report.n_scored_rows == 4
    assert report.zero_minute_share == pytest.approx(0.5)
    assert report.mean_expected_minutes_on_zero_rows == pytest.approx((5.0 + 8.0) / 2)
    assert report.predicted_points_mass_on_zero_rows == pytest.approx(1.0 + 0.5)
    assert report.predicted_points_mass_per_scored_row == pytest.approx(1.5 / 4)
    assert report.auc_played_at_all == pytest.approx(1.0)  # perfectly separates played vs not


def test_floor_ceiling_coverage_fraction_within_bounds():
    floor = pd.Series([0.0, 0.0, 5.0])
    ceiling = pd.Series([5.0, 5.0, 10.0])
    actual = pd.Series([2.0, 6.0, 7.0])  # within, outside, within

    coverage = floor_ceiling_coverage(floor, ceiling, actual)

    assert coverage == pytest.approx(2 / 3)


def test_floor_ceiling_coverage_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        floor_ceiling_coverage(pd.Series([0.0]), pd.Series([1.0, 2.0]), pd.Series([0.5, 0.5]))
