"""Versioned diagnostics — the per-component regression/VIF/xgboost-benchmark reporting pass
(ENGINE_IMPROVEMENTS_2.md D.4), promoted out of ad hoc scripts into a real, testable module (A.6)
per that document's own recommendation ("production logic always lives in versioned modules, never
in notebooks" — the same principle that motivated promoting ``backtest/run_season.py`` itself).

``engine/regression.py`` (``PerPositionRegression``, ``variance_inflation_factors``,
``benchmark_against_xgboost``) was built, tested, and never called by any production path —
ENGINE_IMPROVEMENTS.md Tier 1.2's five untuned constants ended up fitted via direct closed-form
MLE/method-of-moments calls in ``backtest/run_season.py:fit_fn`` instead, which was the right call
for *those* constants (they have closed-form estimators; a per-position OLS/Logit fit doesn't apply
to a single scoreline-correlation parameter or an overdispersion parameter). But that left "which
stats matter" — one of the project's original goals per BUILD_PLAN 2.8 — still unanswered for the
components that genuinely are regression-shaped: goals, assists, defensive contribution, bonus.
This module runs that regression, per component, against real engineered features:

- **Coefficients** — the measured answer to "which stats matter", not a guess.
- **VIF** — BUILD_PLAN 2.8's "check for overfitting/multicollinearity empirically" (a value above
  ~5-10 signals a feature is well-explained by the others).
- **xgboost benchmark** — whether a flexible model materially beats the interpretable one; if it
  doesn't, that's the measured confirmation that ``BUILD_PLAN.md``'s "explainable over clever"
  default is earning its keep for this component, not just an assumption.

Deliberately **not** re-promoted here: the Jensen's-inequality check (B.1), the rate-outlier
distribution (B.2), and the minutes feature-set ablation (B.3) that also gated real decisions in
ENGINE_IMPROVEMENTS_2.md. Those were one-off measurements that directly produced fixes already
landed in the engine itself (the bucket-weighted DC/goals-conceded integration, the goals/assists
shrinkage wiring, and the crowd-feature minutes columns respectively) — the fix is now load-bearing
production code with its own tests, so re-running the diagnostic that justified it adds no ongoing
value the way this regression pass does (a report someone will want to re-run and read every time
real data changes, not a one-time before/after comparison).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.regression import (
    PerPositionRegression,
    RegressionKind,
    benchmark_against_xgboost,
    variance_inflation_factors,
)
from engine.scoring import DEFENSIVE_CONTRIBUTION_THRESHOLD, GK

__all__ = [
    "ComponentRegressionReport",
    "goals_regression_diagnostics",
    "assists_regression_diagnostics",
    "defensive_contribution_regression_diagnostics",
    "bonus_regression_diagnostics",
    "run_all_component_regression_diagnostics",
]


@dataclass(frozen=True)
class ComponentRegressionReport:
    """One component's regression diagnostics: fitted coefficients per position (the "which stats
    matter" answer), VIF per feature (pooled across positions — a coarser but still useful
    multicollinearity check than a per-position VIF would be), and the xgboost-vs-interpretable
    benchmark per position."""

    component: str
    coefficients: pd.DataFrame
    vif: pd.Series
    xgboost_benchmark: pd.DataFrame


def _fit_component(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    kind: RegressionKind,
    component: str,
) -> ComponentRegressionReport:
    fitted = PerPositionRegression(feature_columns=feature_columns, kind=kind).fit(data, target_col)
    return ComponentRegressionReport(
        component=component,
        coefficients=fitted.coefficients_summary(),
        vif=variance_inflation_factors(data, feature_columns),
        xgboost_benchmark=benchmark_against_xgboost(data, feature_columns, target_col, kind),
    )


def goals_regression_diagnostics(engineered: pd.DataFrame) -> ComponentRegressionReport:
    """Candidate inputs: a player's own npxG/90, the opponent's xGA/90, and the league-average
    xGA/90 normalizer — exactly BUILD_PLAN 2.2's multiplicative functional form's own ingredients,
    let loose as free regression coefficients instead of a fixed multiplicative rule."""
    data = engineered[engineered["position"] != GK]
    return _fit_component(
        data,
        ["npxg_per_90", "opponent_xga_per_90", "league_avg_xga_per_90"],
        "goals_scored",
        "linear",
        "goals",
    )


def assists_regression_diagnostics(engineered: pd.DataFrame) -> ComponentRegressionReport:
    """Symmetric with :func:`goals_regression_diagnostics` — BUILD_PLAN 2.3's own inputs."""
    data = engineered[engineered["position"] != GK]
    return _fit_component(
        data,
        ["xa_per_90", "opponent_xga_per_90", "league_avg_xga_per_90"],
        "assists",
        "linear",
        "assists",
    )


def defensive_contribution_regression_diagnostics(engineered: pd.DataFrame) -> ComponentRegressionReport:
    """Logistic fit of "cleared the position's own action threshold" against the player's own
    per-90 rate and the opponent-possession adjustment (BUILD_PLAN 2.5) — the component the plan
    calls "your edge", and the strongest rate-to-outcome relationship of any component measured
    (Spearman 0.757, ENGINE_IMPROVEMENTS.md)."""
    data = engineered[engineered["position"] != GK].copy()
    data["dc_cleared_threshold"] = (
        data["defensive_contribution"] >= data["position"].map(DEFENSIVE_CONTRIBUTION_THRESHOLD)
    ).astype(int)
    return _fit_component(
        data,
        ["dc_per_90", "opponent_possession_share"],
        "dc_cleared_threshold",
        "logistic",
        "defensive_contribution",
    )


def bonus_regression_diagnostics(engineered: pd.DataFrame) -> ComponentRegressionReport:
    """Same four BPS-relevant inputs ``engine/models/bonus.py`` already uses
    (``engine.models.bonus.FEATURE_COLUMNS``), recomputed here from ``engineered``'s own raw rate
    columns rather than requiring the caller to have run the full fit/predict pipeline first —
    position dummies are omitted since :class:`~engine.regression.PerPositionRegression` already
    fits separately per position, so a position indicator would be constant (and singular) within
    each position's own regression. Target is still the raw ``bonus`` column (Tier 3.3's BPS
    recompute remains blocked on the missing 2026/27 numeric table)."""
    data = engineered[engineered["position"] != GK].copy()
    fixture_adjustment = data["opponent_xga_per_90"] / data["league_avg_xga_per_90"]
    minutes_fraction = data["minutes"] / 90.0
    data["expected_goals"] = data["npxg_per_90"] * fixture_adjustment * minutes_fraction
    data["expected_assists"] = data["xa_per_90"] * fixture_adjustment * minutes_fraction
    data["clean_sheet_probability"] = data["clean_sheet_probability_default_rho"]
    data["defensive_action_rate"] = data["dc_per_90"]
    return _fit_component(
        data,
        ["expected_goals", "expected_assists", "clean_sheet_probability", "defensive_action_rate"],
        "bonus",
        "linear",
        "bonus",
    )


def run_all_component_regression_diagnostics(
    engineered: pd.DataFrame,
) -> dict[str, ComponentRegressionReport]:
    """D.4: run every component's regression/VIF/xgboost-benchmark diagnostic against one
    engineered-features frame (e.g. ``backtest.run_season.engineer_features``'s output) and return
    them keyed by component name."""
    return {
        "goals": goals_regression_diagnostics(engineered),
        "assists": assists_regression_diagnostics(engineered),
        "defensive_contribution": defensive_contribution_regression_diagnostics(engineered),
        "bonus": bonus_regression_diagnostics(engineered),
    }
