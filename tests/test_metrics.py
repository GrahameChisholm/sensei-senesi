"""Tests for accuracy, bias, calibration, and captaincy hit-rate metrics (3.2)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    bias_by_group,
    captaincy_hit_rate,
    component_calibration,
    player_accuracy,
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
