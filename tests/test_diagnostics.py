"""Tests for backtest/diagnostics.py — the per-component regression/VIF/xgboost reporting pass
(ENGINE_IMPROVEMENTS_2.md D.4)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.diagnostics import (
    ComponentRegressionReport,
    assists_regression_diagnostics,
    bonus_regression_diagnostics,
    defensive_contribution_regression_diagnostics,
    goals_regression_diagnostics,
    run_all_component_regression_diagnostics,
)


def _synthetic_engineered(n_per_position: int = 60, seed: int = 0) -> pd.DataFrame:
    """A synthetic engineered-features frame with real, learnable relationships, big enough for
    per-position OLS/Logit to fit without degenerating (unlike the tiny 4-player fixture
    tests/test_run_season.py uses for wiring checks) -- mirrors how engine/regression.py's own
    tests size their synthetic data."""
    rng = np.random.default_rng(seed)
    rows = []
    for position in ("DEF", "MID", "FWD"):
        npxg = rng.uniform(0.0, 0.6, n_per_position)
        xa = rng.uniform(0.0, 0.4, n_per_position)
        opponent_xga = rng.uniform(0.8, 1.8, n_per_position)
        league_avg_xga = np.full(n_per_position, 1.4)
        dc_per_90 = rng.uniform(2.0, 14.0, n_per_position)
        opponent_possession = rng.uniform(0.35, 0.65, n_per_position)
        minutes = rng.choice([90.0, 75.0, 60.0], n_per_position)
        clean_sheet_prob = rng.uniform(0.1, 0.5, n_per_position)

        fixture_adj = opponent_xga / league_avg_xga
        goals_scored = rng.poisson(npxg * fixture_adj * (minutes / 90.0))
        assists = rng.poisson(xa * fixture_adj * (minutes / 90.0))
        dc_mu = dc_per_90 * (opponent_possession / 0.5) * (minutes / 90.0)
        defensive_contribution = rng.poisson(dc_mu)
        bonus = np.clip(
            1.5 * goals_scored + assists + clean_sheet_prob + rng.normal(0, 0.3, n_per_position),
            0,
            3,
        )

        for i in range(n_per_position):
            rows.append(
                {
                    "position": position,
                    "npxg_per_90": npxg[i],
                    "xa_per_90": xa[i],
                    "opponent_xga_per_90": opponent_xga[i],
                    "league_avg_xga_per_90": league_avg_xga[i],
                    "dc_per_90": dc_per_90[i],
                    "opponent_possession_share": opponent_possession[i],
                    "minutes": minutes[i],
                    "goals_scored": float(goals_scored[i]),
                    "assists": float(assists[i]),
                    "defensive_contribution": float(defensive_contribution[i]),
                    "bonus": float(bonus[i]),
                    "clean_sheet_probability_default_rho": clean_sheet_prob[i],
                }
            )
    return pd.DataFrame(rows)


def test_goals_regression_diagnostics_returns_coefficients_per_position():
    engineered = _synthetic_engineered()

    report = goals_regression_diagnostics(engineered)

    assert isinstance(report, ComponentRegressionReport)
    assert report.component == "goals"
    assert set(report.coefficients["position"]) == {"DEF", "MID", "FWD"}
    assert set(report.coefficients["feature"]) == {
        "npxg_per_90",
        "opponent_xga_per_90",
        "league_avg_xga_per_90",
        "const",
    }
    # The synthetic data's own generative process makes npxg_per_90 a strong, real predictor.
    npxg_rows = report.coefficients[report.coefficients["feature"] == "npxg_per_90"]
    assert (npxg_rows["coefficient"] > 0).all()


def test_goals_regression_diagnostics_excludes_gk():
    engineered = _synthetic_engineered()
    gk_row = engineered.iloc[[0]].copy()
    gk_row["position"] = "GK"
    with_gk = pd.concat([engineered, gk_row], ignore_index=True)

    report = goals_regression_diagnostics(with_gk)

    assert "GK" not in set(report.coefficients["position"])


def test_assists_regression_diagnostics_returns_coefficients_per_position():
    engineered = _synthetic_engineered()
    report = assists_regression_diagnostics(engineered)
    assert report.component == "assists"
    assert set(report.coefficients["position"]) == {"DEF", "MID", "FWD"}


def test_defensive_contribution_regression_diagnostics_uses_position_threshold():
    engineered = _synthetic_engineered()
    report = defensive_contribution_regression_diagnostics(engineered)
    assert report.component == "defensive_contribution"
    # Logistic fit -- dc_per_90 should be a positive predictor of clearing the threshold.
    dc_rows = report.coefficients[report.coefficients["feature"] == "dc_per_90"]
    assert (dc_rows["coefficient"] > 0).all()


def test_bonus_regression_diagnostics_computes_features_from_raw_columns():
    engineered = _synthetic_engineered()
    report = bonus_regression_diagnostics(engineered)
    assert report.component == "bonus"
    assert set(report.coefficients["feature"]) == {
        "expected_goals",
        "expected_assists",
        "clean_sheet_probability",
        "defensive_action_rate",
        "const",
    }


def test_vif_and_xgboost_benchmark_present_for_every_component():
    engineered = _synthetic_engineered()
    report = goals_regression_diagnostics(engineered)

    assert isinstance(report.vif, pd.Series)
    assert set(report.vif.index) == {"npxg_per_90", "opponent_xga_per_90", "league_avg_xga_per_90"}
    assert (report.vif >= 1.0).all()  # VIF is always >= 1 by construction

    assert isinstance(report.xgboost_benchmark, pd.DataFrame)
    assert set(report.xgboost_benchmark["position"]) == {"DEF", "MID", "FWD"}
    assert "xgboost_beats_interpretable" in report.xgboost_benchmark.columns


def test_run_all_component_regression_diagnostics_covers_every_component():
    engineered = _synthetic_engineered()
    reports = run_all_component_regression_diagnostics(engineered)
    assert set(reports) == {"goals", "assists", "defensive_contribution", "bonus"}
    for report in reports.values():
        assert isinstance(report, ComponentRegressionReport)
