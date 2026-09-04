"""Tests for accuracy, bias, calibration, and captaincy hit-rate metrics (3.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    bias_by_group,
    brier_vs_constant,
    captaincy_hit_rate,
    club_minutes_coverage,
    component_calibration,
    decision_set_rank_correlation,
    floor_ceiling_coverage,
    mean_calibration,
    minutes_model_diagnostics,
    player_accuracy,
    rank_correlation,
    rate_calibration_at_realised_minutes,
    rate_plausibility_by_evidence,
    thin_tail_accuracy,
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


def test_bias_by_group_sources_group_col_from_actuals_when_absent_from_predictions():
    # ENGINE_IMPROVEMENTS_3.md B.2: a price tier lives only in ground truth, not on every
    # prediction row the way "position" already does.
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["MID", "MID", "MID", "MID"],
            "gameweek": [1, 1, 1, 1],
            "expected_points": [5.0, 5.0, 5.0, 5.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "total_points": [4.0, 4.0, 4.0, 4.0],
            "price_tier": ["cheap", "cheap", "expensive", "expensive"],
        }
    )

    report = bias_by_group(predictions, actuals, group_col="price_tier")

    assert set(report.by_group["price_tier"]) == {"cheap", "expensive"}
    assert report.by_group["mean_residual"].eq(1.0).all()


def test_bias_by_group_min_relative_effect_zero_uses_absolute_floor_only():
    # A bias that's a small % of a high-scoring group's mean actual, but a large absolute value,
    # should flag with min_relative_effect=0.0 even though the default (0.10) would clear it
    # (ENGINE_IMPROVEMENTS_3.md B.2 — the premium price tier's own failure mode).
    rng = np.random.default_rng(4)
    n = 200
    predicted = np.full(2 * n, 3.0)
    # +0.2 absolute, ~7% relative bias with a little noise so the significance test isn't singular.
    actual = 2.8 + rng.normal(0, 0.05, 2 * n)
    predictions = pd.DataFrame(
        {
            "player_id": range(2 * n),
            "position": ["MID"] * (2 * n),
            "gameweek": [1] * (2 * n),
            "expected_points": predicted,
        }
    )
    actuals = pd.DataFrame(
        {"player_id": range(2 * n), "gameweek": [1] * (2 * n), "total_points": actual}
    )

    default_floor = bias_by_group(predictions, actuals, group_col="position")
    absolute_only = bias_by_group(
        predictions, actuals, group_col="position", min_absolute_effect=0.1, min_relative_effect=0.0
    )

    assert not default_floor.by_group.set_index("position").loc["MID", "severe"]
    assert absolute_only.by_group.set_index("position").loc["MID", "severe"]


def test_brier_vs_constant_beats_constant_when_predictions_discriminate():
    rng = np.random.default_rng(3)
    n = 2000
    predicted = rng.uniform(0, 1, n)
    actual = (rng.uniform(0, 1, n) < predicted).astype(float)

    report = brier_vs_constant(pd.Series(predicted), pd.Series(actual))

    assert report.beats_constant
    assert report.brier < report.constant_brier


def test_brier_vs_constant_loses_to_constant_when_overconfident():
    # Every row predicts near-certainty in the wrong direction half the time -- badly
    # miscalibrated despite having real (if inverted) discriminative signal, the same shape as the
    # shipped clean-sheet defect (ENGINE_IMPROVEMENTS_3.md A.1/B.3).
    predicted = pd.Series([0.9] * 50 + [0.9] * 50)
    actual = pd.Series([1.0] * 20 + [0.0] * 80)  # base rate 0.2, but predicted is ~0.9 throughout

    report = brier_vs_constant(predicted, actual)

    assert not report.beats_constant
    assert report.brier > report.constant_brier


def test_brier_vs_constant_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        brier_vs_constant(pd.Series([0.1, 0.2]), pd.Series([1.0]))


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


def test_minutes_diagnostics_splits_zero_minute_mass_across_components():
    # B2: the aggregate mass says how much leaks; only the split says which component to gate.
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2],
            "gameweek": [1, 1],
            "expected_points": [1.0, 3.0],
            "p_zero": [0.9, 0.1],
            "expected_minutes": [5.0, 85.0],
            "appearance": [0.6, 1.8],
            "goals": [0.3, 1.0],
            "cards": [-0.1, -0.2],
        }
    )
    actuals = pd.DataFrame({"player_id": [1, 2], "gameweek": [1, 1], "minutes": [0, 90]})

    report = minutes_model_diagnostics(predictions, actuals)
    split = report.zero_minute_mass_by_component

    # Only player 1 played zero minutes, so the split is exactly that row's components.
    assert list(split["component"]) == ["appearance", "goals", "cards"]  # sorted by mass, desc
    assert split.set_index("component").loc["appearance", "mass"] == pytest.approx(0.6)
    assert split.set_index("component").loc["cards", "mass"] == pytest.approx(-0.1)
    assert split["mass"].sum() == pytest.approx(0.6 + 0.3 - 0.1)


def test_minutes_diagnostics_component_split_is_none_without_component_columns():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2],
            "gameweek": [1, 1],
            "expected_points": [1.0, 3.0],
            "p_zero": [0.9, 0.1],
            "expected_minutes": [5.0, 85.0],
        }
    )
    actuals = pd.DataFrame({"player_id": [1, 2], "gameweek": [1, 1], "minutes": [0, 90]})

    assert minutes_model_diagnostics(predictions, actuals).zero_minute_mass_by_component is None


def test_calibrated_floor_is_zero_for_a_perfectly_discriminating_model():
    # B2: a model that is never wrong about who plays carries no mass on zero-minute rows, so its
    # floor is zero and any observed mass is pure excess.
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "expected_points": [5.0, 0.0, 0.0, 6.0],
            "p_zero": [0.0, 1.0, 1.0, 0.0],
            "expected_minutes": [90.0, 0.0, 0.0, 90.0],
        }
    )
    actuals = pd.DataFrame(
        {"player_id": [1, 2, 3, 4], "gameweek": [1, 1, 1, 1], "minutes": [90, 0, 0, 90]}
    )

    report = minutes_model_diagnostics(predictions, actuals)

    assert report.calibrated_floor_mass_per_scored_row == pytest.approx(0.0)
    assert report.zero_minute_mass_excess == pytest.approx(0.0)


def test_calibrated_floor_absorbs_mass_a_correctly_uncertain_model_cannot_avoid():
    # Ten identical players each given a 50% chance of playing and 4 expected points; exactly five
    # play. The model is perfectly calibrated, so all the mass on the five who didn't play is
    # irreducible -- the floor must equal the observed mass and the excess must be zero.
    predictions = pd.DataFrame(
        {
            "player_id": list(range(1, 11)),
            "gameweek": [1] * 10,
            "expected_points": [4.0] * 10,
            "p_zero": [0.5] * 10,
            "expected_minutes": [45.0] * 10,
        }
    )
    actuals = pd.DataFrame(
        {"player_id": list(range(1, 11)), "gameweek": [1] * 10, "minutes": [90] * 5 + [0] * 5}
    )

    report = minutes_model_diagnostics(predictions, actuals)

    assert report.predicted_points_mass_per_scored_row == pytest.approx(2.0)  # 5 * 4.0 / 10
    assert report.calibrated_floor_mass_per_scored_row == pytest.approx(2.0)
    assert report.zero_minute_mass_excess == pytest.approx(0.0)


def test_calibrated_floor_is_below_observed_mass_for_an_overconfident_model():
    # Same setup, but the model claims 80% will play when only 50% do. The floor rescales to the
    # realised rate, so the excess is the part attributable to that over-confidence.
    predictions = pd.DataFrame(
        {
            "player_id": list(range(1, 11)),
            "gameweek": [1] * 10,
            "expected_points": [4.0] * 10,
            "p_zero": [0.2] * 10,
            "expected_minutes": [72.0] * 10,
        }
    )
    actuals = pd.DataFrame(
        {"player_id": list(range(1, 11)), "gameweek": [1] * 10, "minutes": [90] * 5 + [0] * 5}
    )

    report = minutes_model_diagnostics(predictions, actuals)

    assert report.predicted_points_mass_per_scored_row == pytest.approx(2.0)
    # Rescaled by realised/predicted = 0.5/0.8.
    assert report.calibrated_floor_mass_per_scored_row == pytest.approx(2.0 * 0.5 / 0.8)
    assert report.zero_minute_mass_excess > 0.0


def test_floor_ceiling_coverage_fraction_within_bounds():
    floor = pd.Series([0.0, 0.0, 5.0])
    ceiling = pd.Series([5.0, 5.0, 10.0])
    actual = pd.Series([2.0, 6.0, 7.0])  # within, outside, within

    coverage = floor_ceiling_coverage(floor, ceiling, actual)

    assert coverage == pytest.approx(2 / 3)


def test_floor_ceiling_coverage_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        floor_ceiling_coverage(pd.Series([0.0]), pd.Series([1.0, 2.0]), pd.Series([0.5, 0.5]))


# =================================================================================================
# ENGINE_IMPROVEMENTS_5.md Tier 0.1 — the decision-relevant metrics
# =================================================================================================


def test_bias_by_group_conditional_and_unconditional_diverge_when_errors_offset():
    """The finding that motivated ``minutes_col``: an engine can look unbiased overall while being
    badly biased on exactly the players a manager fields. Here the two zero-minute rows are
    over-predicted by +3 each and the two 60+ rows under-predicted by -3 each, so the unconditional
    mean residual is 0.0 while the played-60+ residual is -3.0. The real 2025/26 walk-forward shows
    the same structure at +0.008 against -0.990."""
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["MID"] * 4,
            "gameweek": [1, 1, 1, 1],
            "expected_points": [3.0, 3.0, 4.0, 4.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "total_points": [0.0, 0.0, 7.0, 7.0],
            "minutes": [0, 0, 90, 90],
        }
    )

    unconditional = bias_by_group(predictions, actuals, group_col="position")
    conditional = bias_by_group(predictions, actuals, group_col="position", minutes_col="minutes")

    assert unconditional.by_group["mean_residual"].iloc[0] == pytest.approx(0.0)
    assert conditional.by_group["mean_residual"].iloc[0] == pytest.approx(-3.0)
    assert conditional.by_group["n"].iloc[0] == 2


def test_bias_by_group_without_minutes_col_ignores_minutes_entirely():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2],
            "position": ["MID", "MID"],
            "gameweek": [1, 1],
            "expected_points": [5.0, 3.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2],
            "gameweek": [1, 1],
            "total_points": [4.0, 2.0],
            "minutes": [0, 0],
        }
    )

    # Every row has 0 minutes, so a conditional read would be empty. The default must not filter.
    report = bias_by_group(predictions, actuals, group_col="position")

    assert report.by_group["n"].iloc[0] == 2
    assert report.by_group["mean_residual"].iloc[0] == pytest.approx(1.0)


def test_bias_by_group_min_minutes_threshold_is_respected():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "position": ["MID"] * 3,
            "gameweek": [1, 1, 1],
            "expected_points": [2.0, 2.0, 2.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "gameweek": [1, 1, 1],
            "total_points": [2.0, 2.0, 2.0],
            "minutes": [0, 30, 90],
        }
    )

    any_minutes = bias_by_group(
        predictions, actuals, group_col="position", minutes_col="minutes", min_minutes=1.0
    )
    sixty_plus = bias_by_group(
        predictions, actuals, group_col="position", minutes_col="minutes", min_minutes=60.0
    )

    assert any_minutes.by_group["n"].iloc[0] == 2
    assert sixty_plus.by_group["n"].iloc[0] == 1


def test_decision_set_rank_correlation_perfect_within_shortlist():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 1, 2, 3],
            "position": ["MID"] * 6,
            "gameweek": [1, 1, 1, 2, 2, 2],
            "expected_points": [9.0, 5.0, 1.0, 8.0, 4.0, 1.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 1, 2, 3],
            "gameweek": [1, 1, 1, 2, 2, 2],
            "total_points": [10.0, 2.0, 99.0, 12.0, 3.0, 99.0],
        }
    )

    report = decision_set_rank_correlation(predictions, actuals, top_n=2)

    assert report.top_n == 2
    assert report.n_gameweeks == 2
    assert report.mean_spearman == pytest.approx(1.0)
    assert report.share_positive == pytest.approx(1.0)
    # Player 3 scored 99 but was never shortlisted, so must not enter the correlation at all.
    assert set(report.by_gameweek["n"]) == {2}


def test_decision_set_rank_correlation_is_averaged_per_gameweek_not_pooled():
    """Pooling shortlist rows across gameweeks would let a high-scoring gameweek's weak pick
    outrank a low-scoring gameweek's strong pick, a comparison no single decision ever spans. Each
    gameweek here is internally inverted (Spearman -1), while the pooled ordering looks positive
    purely because gameweek 2 scores higher across the board."""
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "position": ["MID"] * 4,
            "gameweek": [1, 1, 2, 2],
            "expected_points": [2.0, 1.0, 4.0, 3.0],
        }
    )
    actuals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 2, 2],
            "total_points": [1.0, 2.0, 30.0, 40.0],
        }
    )

    report = decision_set_rank_correlation(predictions, actuals, top_n=2)

    assert report.mean_spearman == pytest.approx(-1.0)
    assert report.share_positive == pytest.approx(0.0)


def test_decision_set_rank_correlation_reports_error_and_bias_within_shortlist():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2],
            "position": ["MID", "MID"],
            "gameweek": [1, 1],
            "expected_points": [5.0, 4.0],
        }
    )
    actuals = pd.DataFrame({"player_id": [1, 2], "gameweek": [1, 1], "total_points": [2.0, 2.0]})

    report = decision_set_rank_correlation(predictions, actuals, top_n=2)

    assert report.mean_absolute_error == pytest.approx(2.5)
    assert report.mean_bias == pytest.approx(2.5)


def test_decision_set_rank_correlation_skips_gameweeks_with_tied_actuals():
    # Every actual ties, so a rank correlation is undefined. That gameweek must not count as
    # scored, rather than contributing a NaN that poisons the mean.
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2],
            "position": ["MID", "MID"],
            "gameweek": [1, 1],
            "expected_points": [5.0, 4.0],
        }
    )
    actuals = pd.DataFrame({"player_id": [1, 2], "gameweek": [1, 1], "total_points": [3.0, 3.0]})

    report = decision_set_rank_correlation(predictions, actuals, top_n=2)

    assert report.n_gameweeks == 0
    assert np.isnan(report.mean_spearman)


def test_rate_calibration_strips_the_minutes_model_out():
    """Two players with identical true per-90 rates, one of whom the minutes model badly
    misjudged. The rate view must score them identically, since the rate model is not at fault."""
    # Both imply 1.0 per 90. Player A was expected to play 90 and did; player B was expected to
    # play 45 but played 90.
    report = rate_calibration_at_realised_minutes(
        predicted_quantity=pd.Series([1.0, 0.5]),
        predicted_minutes=pd.Series([90.0, 45.0]),
        realised_minutes=pd.Series([90.0, 90.0]),
        realised_count=pd.Series([1.0, 1.0]),
    )

    assert report.mean_predicted == pytest.approx(1.0)
    assert report.mean_actual == pytest.approx(1.0)
    assert report.relative_gap == pytest.approx(0.0)


def test_rate_calibration_contributes_zero_for_a_player_who_did_not_play():
    # A row nobody played must not be dropped (that would condition on the outcome) and must not
    # contribute predicted mass either, since the rate was never given a chance to express itself.
    report = rate_calibration_at_realised_minutes(
        predicted_quantity=pd.Series([1.0, 0.4]),
        predicted_minutes=pd.Series([90.0, 40.0]),
        realised_minutes=pd.Series([90.0, 0.0]),
        realised_count=pd.Series([1.0, 0.0]),
    )

    assert report.mean_predicted == pytest.approx(0.5)  # (1.0 + 0.0) / 2
    assert report.mean_actual == pytest.approx(0.5)
    assert report.relative_gap == pytest.approx(0.0)


def test_rate_calibration_detects_a_genuinely_hot_rate():
    report = rate_calibration_at_realised_minutes(
        predicted_quantity=pd.Series([1.2, 1.2]),
        predicted_minutes=pd.Series([90.0, 90.0]),
        realised_minutes=pd.Series([90.0, 90.0]),
        realised_count=pd.Series([1.0, 1.0]),
    )

    assert report.mean_predicted == pytest.approx(1.2)
    assert report.relative_gap == pytest.approx(0.2)


def test_rate_calibration_tolerates_zero_expected_minutes_without_dividing_by_zero():
    report = rate_calibration_at_realised_minutes(
        predicted_quantity=pd.Series([0.0]),
        predicted_minutes=pd.Series([0.0]),
        realised_minutes=pd.Series([90.0]),
        realised_count=pd.Series([0.0]),
    )

    assert report.mean_predicted == pytest.approx(0.0)


def test_rate_calibration_mismatched_lengths_raises():
    with pytest.raises(ValueError, match="same length"):
        rate_calibration_at_realised_minutes(
            pd.Series([1.0]), pd.Series([90.0]), pd.Series([90.0]), pd.Series([1.0, 2.0])
        )


def test_decision_set_rank_correlation_rejects_unrankable_top_n():
    predictions = pd.DataFrame(
        {
            "player_id": [1],
            "position": ["MID"],
            "gameweek": [1],
            "expected_points": [5.0],
        }
    )
    actuals = pd.DataFrame({"player_id": [1], "gameweek": [1], "total_points": [5.0]})

    with pytest.raises(ValueError, match="at least 2"):
        decision_set_rank_correlation(predictions, actuals, top_n=1)


def test_club_minutes_coverage_detects_over_and_under_covered_clubs():
    # Real 2026/27 GW3 pull: outfield p_60_plus summed 8.7 (under) to 15.3 (over) against a true
    # 10 per club -- a defect fixture_minutes_coverage's combined-both-squads check can't isolate
    # since one club's over-coverage can offset the other's under-coverage in the fixture total.
    predictions = pd.DataFrame(
        {
            "team": ["Over"] * 15 + ["Under"] * 8,
            "gameweek": [1] * 23,
            "position": ["MID"] * 15 + ["MID"] * 8,
            "p_60_plus": [15.3 / 15] * 15 + [8.7 / 8] * 8,
        }
    )
    report = club_minutes_coverage(predictions, outfield_target=10.0, goalkeeper_target=1.0)
    by_club = report.by_club.set_index("team")
    assert by_club.loc["Over", "sum_p_60_plus"] == pytest.approx(15.3)
    assert by_club.loc["Under", "sum_p_60_plus"] == pytest.approx(8.7)
    assert report.max_absolute_gap == pytest.approx(5.3, abs=0.01)


def test_club_minutes_coverage_scores_goalkeepers_separately_from_outfield():
    # The real defect: two goalkeepers at the same club both individually rating 76% to start.
    predictions = pd.DataFrame(
        {
            "team": ["Ipswich", "Ipswich", "Ipswich"],
            "gameweek": [1, 1, 1],
            "position": ["GK", "GK", "MID"],
            "p_60_plus": [0.76, 0.76, 0.9],
        }
    )
    report = club_minutes_coverage(predictions, outfield_target=10.0, goalkeeper_target=1.0)
    gk_row = report.by_club.loc[report.by_club["is_goalkeeper"]].iloc[0]
    assert gk_row["sum_p_60_plus"] == pytest.approx(1.52)
    assert gk_row["gap"] == pytest.approx(0.52)


def test_club_minutes_coverage_rejects_empty_predictions():
    with pytest.raises(ValueError):
        club_minutes_coverage(pd.DataFrame(columns=["team", "gameweek", "position", "p_60_plus"]))


def test_rate_plausibility_flags_thin_cohort_exceeding_established():
    # Real GW3 pull: a 13-minute cameo implied 28.3 dc_per_90, above the highest rate any
    # established (170+ effective minutes) player actually carried (18.5) -- the exact inverted
    # spread this check exists to catch.
    established_minutes = [180.0, 180.0, 176.0, 180.0, 180.0, 180.0, 180.0]
    predictions = pd.DataFrame(
        {
            "dc_per_90": [28.3, 21.0, 20.97] + [18.5, 13.96, 11.99, 7.0, 5.0, 4.0, 3.0],
            "effective_minutes": [13.0, 8.0, 28.0] + established_minutes,
        }
    )
    report = rate_plausibility_by_evidence(
        predictions,
        "dc_per_90",
        "effective_minutes",
        thin_threshold=100.0,
        established_threshold=150.0,
    )
    assert report.thin_cohort_exceeds_established is True


def test_rate_plausibility_does_not_flag_a_well_shrunk_rate():
    established_minutes = [180.0, 180.0, 176.0, 180.0, 180.0, 180.0, 180.0]
    predictions = pd.DataFrame(
        {
            "dc_per_90": [7.0, 7.2, 6.8] + [18.5, 13.96, 11.99, 7.0, 5.0, 4.0, 3.0],
            "effective_minutes": [13.0, 8.0, 28.0] + established_minutes,
        }
    )
    report = rate_plausibility_by_evidence(
        predictions,
        "dc_per_90",
        "effective_minutes",
        thin_threshold=100.0,
        established_threshold=150.0,
    )
    assert report.thin_cohort_exceeds_established is False


def test_rate_plausibility_handles_empty_cohort():
    predictions = pd.DataFrame({"dc_per_90": [7.0, 8.0], "effective_minutes": [500.0, 600.0]})
    report = rate_plausibility_by_evidence(predictions, "dc_per_90", "effective_minutes")
    assert report.thin_cohort_exceeds_established is False
    thin_row = report.by_cohort.set_index("cohort").loc["thin"]
    assert thin_row["n"] == 0


def test_thin_tail_accuracy_scores_only_low_evidence_rows():
    predictions = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "gameweek": [1, 1, 1],
            "expected_points": [10.0, 2.0, 3.0],
            "effective_minutes": [50.0, 500.0, 30.0],
        }
    )
    actuals = pd.DataFrame(
        {"player_id": [1, 2, 3], "gameweek": [1, 1, 1], "total_points": [2.0, 2.5, 3.5]}
    )
    report = thin_tail_accuracy(predictions, actuals, "effective_minutes", threshold=100.0)
    assert report.n == 2
    assert report.mae == pytest.approx((abs(10.0 - 2.0) + abs(3.0 - 3.5)) / 2)


def test_thin_tail_accuracy_rejects_missing_evidence_column():
    predictions = pd.DataFrame({"player_id": [1], "gameweek": [1], "expected_points": [1.0]})
    actuals = pd.DataFrame({"player_id": [1], "gameweek": [1], "total_points": [1.0]})
    with pytest.raises(ValueError):
        thin_tail_accuracy(predictions, actuals, "missing_col")
