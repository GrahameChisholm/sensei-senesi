"""Tests for engine/regression.py — per-position interpretable regression fitting (2.8)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.regression import (
    PerPositionRegression,
    benchmark_against_xgboost,
    variance_inflation_factors,
)


def _synthetic_linear_data(n_per_position: int = 60, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = ["DEF", "MID", "FWD"]
    # Different true slope per position -- fitting per-position should recover this, a single
    # blended model would average the distinctions away (exactly what BUILD_PLAN 2.8 warns about).
    true_slope = {"DEF": 0.5, "MID": 1.5, "FWD": 3.0}
    rows = []
    for position in positions:
        xg = rng.uniform(0, 1, n_per_position)
        fixture_difficulty = rng.uniform(0.5, 1.5, n_per_position)
        noise = rng.normal(0, 0.05, n_per_position)
        goals = true_slope[position] * xg * fixture_difficulty + noise
        for i in range(n_per_position):
            rows.append(
                {
                    "position": position,
                    "xg": xg[i],
                    "fixture_difficulty": fixture_difficulty[i],
                    "goals": goals[i],
                }
            )
    return pd.DataFrame(rows)


def _synthetic_logistic_data(n_per_position: int = 200, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    positions = ["DEF", "MID"]
    true_slope = {"DEF": 2.0, "MID": 4.0}
    rows = []
    for position in positions:
        action_rate = rng.uniform(0, 1, n_per_position)
        possession_adj = rng.uniform(0.5, 1.5, n_per_position)
        logit = true_slope[position] * (action_rate * possession_adj - 0.5) + rng.normal(
            0, 0.3, n_per_position
        )
        prob = 1 / (1 + np.exp(-logit))
        cleared = (rng.uniform(0, 1, n_per_position) < prob).astype(int)
        for i in range(n_per_position):
            rows.append(
                {
                    "position": position,
                    "action_rate": action_rate[i],
                    "possession_adj": possession_adj[i],
                    "cleared": cleared[i],
                }
            )
    return pd.DataFrame(rows)


def test_per_position_regression_linear_fits_separately_per_position():
    data = _synthetic_linear_data()
    model = PerPositionRegression(feature_columns=["xg", "fixture_difficulty"], kind="linear")
    model.fit(data, target_col="goals")

    summary = model.coefficients_summary()
    assert set(summary["position"]) == {"DEF", "MID", "FWD"}

    fwd_xg_coef = summary[(summary.position == "FWD") & (summary.feature == "xg")][
        "coefficient"
    ].iloc[0]
    def_xg_coef = summary[(summary.position == "DEF") & (summary.feature == "xg")][
        "coefficient"
    ].iloc[0]
    # FWD's true slope (3.0) is much steeper than DEF's (0.5) -- per-position fitting should
    # recover meaningfully different coefficients, not one blended average.
    assert fwd_xg_coef > def_xg_coef * 2


def test_per_position_regression_predict_matches_position_specific_model():
    data = _synthetic_linear_data()
    model = PerPositionRegression(feature_columns=["xg", "fixture_difficulty"], kind="linear")
    model.fit(data, target_col="goals")
    predictions = model.predict(data)
    assert len(predictions) == len(data)
    assert not np.isnan(predictions).any()
    # In-sample fit should be reasonably close to actual for a low-noise synthetic relationship.
    assert np.mean(np.abs(predictions - data["goals"])) < 0.2


def test_per_position_regression_predict_before_fit_raises():
    model = PerPositionRegression(feature_columns=["xg"], kind="linear")
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"xg": [0.1], "position": ["FWD"]}))


def test_per_position_regression_coefficients_summary_before_fit_raises():
    model = PerPositionRegression(feature_columns=["xg"], kind="linear")
    with pytest.raises(RuntimeError):
        model.coefficients_summary()


def test_per_position_regression_predict_unknown_position_raises():
    data = _synthetic_linear_data()
    model = PerPositionRegression(feature_columns=["xg", "fixture_difficulty"], kind="linear")
    model.fit(data, target_col="goals")
    unseen = pd.DataFrame({"xg": [0.5], "fixture_difficulty": [1.0], "position": ["GK"]})
    with pytest.raises(ValueError):
        model.predict(unseen)


def test_per_position_regression_rejects_unknown_kind():
    data = _synthetic_linear_data()
    model = PerPositionRegression(feature_columns=["xg", "fixture_difficulty"], kind="quadratic")
    with pytest.raises(ValueError):
        model.fit(data, target_col="goals")


def test_per_position_regression_logistic_fits_and_predicts_probabilities():
    data = _synthetic_logistic_data()
    model = PerPositionRegression(
        feature_columns=["action_rate", "possession_adj"], kind="logistic"
    )
    model.fit(data, target_col="cleared")
    predictions = model.predict(data)
    assert ((predictions >= 0) & (predictions <= 1)).all()


def test_variance_inflation_factors_flags_collinear_feature():
    rng = np.random.default_rng(0)
    independent = rng.uniform(0, 1, 200)
    collinear = independent * 2 + rng.normal(0, 0.001, 200)  # near-perfect collinearity
    unrelated = rng.uniform(0, 1, 200)
    data = pd.DataFrame(
        {"independent": independent, "collinear": collinear, "unrelated": unrelated}
    )

    vifs = variance_inflation_factors(data, ["independent", "collinear", "unrelated"])
    assert vifs["collinear"] > vifs["unrelated"]
    assert vifs["independent"] > vifs["unrelated"]


def test_benchmark_against_xgboost_returns_per_position_scores():
    data = _synthetic_linear_data(n_per_position=40)
    result = benchmark_against_xgboost(
        data, feature_columns=["xg", "fixture_difficulty"], target_col="goals", kind="linear"
    )
    assert set(result["position"]) == {"DEF", "MID", "FWD"}
    assert (result["interpretable_score"] >= 0).all()
    assert (result["xgboost_score"] >= 0).all()


def test_benchmark_against_xgboost_skips_thin_positions():
    data = _synthetic_linear_data(n_per_position=40)
    thin_row = pd.DataFrame(
        [{"position": "GK", "xg": 0.1, "fixture_difficulty": 1.0, "goals": 0.05}]
    )
    combined = pd.concat([data, thin_row], ignore_index=True)
    result = benchmark_against_xgboost(
        combined, feature_columns=["xg", "fixture_difficulty"], target_col="goals", kind="linear"
    )
    assert "GK" not in set(result["position"])


def test_benchmark_against_xgboost_rejects_unknown_kind():
    data = _synthetic_linear_data(n_per_position=40)
    with pytest.raises(ValueError):
        benchmark_against_xgboost(
            data,
            feature_columns=["xg", "fixture_difficulty"],
            target_col="goals",
            kind="quadratic",
        )
