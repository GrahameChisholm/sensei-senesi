"""Per-position, per-component regression fitting candidate inputs to real outcomes (2.8).

"Which stats matter" becomes a measured output rather than a guess: this fits an interpretable
regression (statsmodels — coefficients, p-values, diagnostics) **separately per position**,
because value concentrates so differently by role (BUILD_PLAN 2.8) that a single blended model
would average the distinctions away.

**Locked in: per-component, never a unified points regression.** This module is a general-purpose
fitting/reporting utility, reused independently by each component (goals, assists, clean sheets,
defensive contribution, bonus) against its *own* real-world outcome and *own* candidate inputs —
never one model that takes every feature and predicts total FPL points directly. The point
*conversion* rates (10/6/5/4 per goal, thresholds per position, etc.) are already fixed by the
game's rules (``engine/scoring.py``); only the underlying rates/probabilities are genuinely
uncertain and worth fitting from data.

Start with interpretable linear/logistic regression. `xgboost` is only ever a *benchmark* — see
:func:`benchmark_against_xgboost` — to check whether a flexible model materially beats the
interpretable one; if it doesn't, that confirms the simple model is enough (BUILD_PLAN 2.8).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.discrete.discrete_model as sm_discrete
import statsmodels.regression.linear_model as sm_linear
from sklearn.metrics import log_loss, mean_absolute_error
from sklearn.model_selection import KFold
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier, XGBRegressor

RegressionKind = Literal["linear", "logistic"]
# The precise statsmodels result type differs between OLS (linear) and Logit (logistic) fits;
# both expose the same `.params` / `.pvalues` / `.predict()` interface this module relies on.
StatsmodelsResult = sm_linear.RegressionResultsWrapper | sm_discrete.BinaryResultsWrapper

DEFAULT_N_SPLITS = 5
DEFAULT_RANDOM_STATE = 0
# xgboost benchmark hyperparameters -- deliberately modest (this is a sanity-check benchmark, not
# a tuned production model; BUILD_PLAN 2.8 only asks whether it materially beats the interpretable
# model, not for xgboost itself to become part of the engine).
BENCHMARK_N_ESTIMATORS = 100
BENCHMARK_MAX_DEPTH = 3


@dataclass
class FittedPositionModel:
    """One position's fitted statsmodels regression, with its own coefficients/p-values."""

    position: str
    kind: RegressionKind
    result: StatsmodelsResult
    feature_columns: list[str]

    @property
    def coefficients(self) -> pd.Series:
        return self.result.params

    @property
    def p_values(self) -> pd.Series:
        return self.result.pvalues

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        X = sm.add_constant(data[self.feature_columns], has_constant="add")
        return np.asarray(self.result.predict(X))


@dataclass
class PerPositionRegression:
    """Fits one interpretable regression per position on real (candidate inputs, actual outcome)
    pairs -- ``kind="linear"`` for a continuous target (e.g. actual goals), ``kind="logistic"``
    for a binary one (e.g. cleared the defensive-contribution threshold)."""

    feature_columns: list[str]
    kind: RegressionKind
    models: dict[str, FittedPositionModel] = field(default_factory=dict, init=False, repr=False)

    def fit(
        self, data: pd.DataFrame, target_col: str, position_col: str = "position"
    ) -> PerPositionRegression:
        if self.kind not in ("linear", "logistic"):
            raise ValueError(f"unknown regression kind: {self.kind!r}")
        self.models = {}
        for position, group in data.groupby(position_col):
            X = sm.add_constant(group[self.feature_columns], has_constant="add")
            y = group[target_col]
            if self.kind == "linear":
                result = sm.OLS(y, X).fit()
            else:
                result = sm.Logit(y, X).fit(disp=0)
            self.models[position] = FittedPositionModel(
                position=position,
                kind=self.kind,
                result=result,
                feature_columns=self.feature_columns,
            )
        return self

    def coefficients_summary(self) -> pd.DataFrame:
        """One row per (position, feature) — the measured answer to "which stats matter"
        (BUILD_PLAN 2.8), ready to inspect or display."""
        if not self.models:
            raise RuntimeError("PerPositionRegression.coefficients_summary called before fit")
        rows = [
            {
                "position": position,
                "feature": feature,
                "coefficient": model.coefficients[feature],
                "p_value": model.p_values[feature],
            }
            for position, model in self.models.items()
            for feature in model.coefficients.index
        ]
        return pd.DataFrame(rows)

    def predict(self, data: pd.DataFrame, position_col: str = "position") -> np.ndarray:
        if not self.models:
            raise RuntimeError("PerPositionRegression.predict called before fit")
        predictions = np.full(len(data), np.nan)
        for position, group in data.groupby(position_col):
            if position not in self.models:
                raise ValueError(f"no fitted model for position {position!r}")
            predictions[group.index.to_numpy()] = self.models[position].predict(group)
        return predictions


def variance_inflation_factors(data: pd.DataFrame, feature_columns: list[str]) -> pd.Series:
    """VIF per feature — BUILD_PLAN 2.8's "check for overfitting/multicollinearity empirically".
    A VIF above roughly 5-10 signals a feature is well-explained by the others and its own
    coefficient may be unstable to interpret in isolation."""
    X = sm.add_constant(data[feature_columns], has_constant="add")
    vifs = {
        column: variance_inflation_factor(X.to_numpy(), X.columns.get_loc(column))
        for column in feature_columns
    }
    return pd.Series(vifs, name="vif")


def benchmark_against_xgboost(
    data: pd.DataFrame,
    feature_columns: list[str],
    target_col: str,
    kind: RegressionKind,
    position_col: str = "position",
    n_splits: int = DEFAULT_N_SPLITS,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """Cross-validated, out-of-sample, per-position comparison between the interpretable
    regression and an xgboost benchmark. Lower score is better for both metrics (mean absolute
    error for ``linear``, log loss for ``logistic``) — BUILD_PLAN 2.8: "if it materially beats the
    interpretable model... consider adopting it; if it doesn't, you've confirmed the simple model
    is enough." Positions with too few rows to cross-validate are skipped, not silently guessed at.
    """
    if kind not in ("linear", "logistic"):
        raise ValueError(f"unknown regression kind: {kind!r}")

    rows = []
    for position, group in data.groupby(position_col):
        n = len(group)
        effective_splits = min(n_splits, n)
        if effective_splits < 2:
            continue

        X = group[feature_columns].to_numpy(dtype=float)
        y = group[target_col].to_numpy(dtype=float)
        kfold = KFold(n_splits=effective_splits, shuffle=True, random_state=random_state)

        interpretable_scores: list[float] = []
        xgboost_scores: list[float] = []
        for train_idx, test_idx in kfold.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            X_train_sm = sm.add_constant(
                pd.DataFrame(X_train, columns=feature_columns), has_constant="add"
            )
            X_test_sm = sm.add_constant(
                pd.DataFrame(X_test, columns=feature_columns), has_constant="add"
            )
            X_test_sm = X_test_sm.reindex(columns=X_train_sm.columns, fill_value=0.0)

            if kind == "linear":
                result = sm.OLS(y_train, X_train_sm).fit()
                interpretable_preds = result.predict(X_test_sm)
                interpretable_scores.append(mean_absolute_error(y_test, interpretable_preds))

                xgb_model = XGBRegressor(
                    n_estimators=BENCHMARK_N_ESTIMATORS,
                    max_depth=BENCHMARK_MAX_DEPTH,
                    random_state=random_state,
                )
                xgb_model.fit(X_train, y_train)
                xgboost_scores.append(mean_absolute_error(y_test, xgb_model.predict(X_test)))
            else:
                result = sm.Logit(y_train, X_train_sm).fit(disp=0)
                interpretable_preds = np.clip(result.predict(X_test_sm), 1e-6, 1 - 1e-6)
                interpretable_scores.append(log_loss(y_test, interpretable_preds, labels=[0, 1]))

                xgb_model = XGBClassifier(
                    n_estimators=BENCHMARK_N_ESTIMATORS,
                    max_depth=BENCHMARK_MAX_DEPTH,
                    random_state=random_state,
                    eval_metric="logloss",
                )
                xgb_model.fit(X_train, y_train)
                xgb_preds = np.clip(xgb_model.predict_proba(X_test)[:, 1], 1e-6, 1 - 1e-6)
                xgboost_scores.append(log_loss(y_test, xgb_preds, labels=[0, 1]))

        interpretable_score = float(np.mean(interpretable_scores))
        xgboost_score = float(np.mean(xgboost_scores))
        rows.append(
            {
                "position": position,
                "interpretable_score": interpretable_score,
                "xgboost_score": xgboost_score,
                "xgboost_beats_interpretable": xgboost_score < interpretable_score,
            }
        )
    return pd.DataFrame(rows)
