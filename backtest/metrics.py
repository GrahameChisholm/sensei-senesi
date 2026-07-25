"""Accuracy, bias, calibration, and captaincy hit-rate metrics (3.2).

A single "is it right" number hides which part of the engine is actually failing, so this tracks
several levels separately: player-level MAE/RMSE overall and per position, systematic bias by
group (residual analysis, not just average error), per-component probability calibration ("when
the model says 40% clean sheet, do clean sheets actually happen ~40% of the time?"), and the
single most decision-relevant metric — captaincy hit-rate, tracked as two side-by-side variants so
a bad pick and a good pick undone by an in-game injury don't collapse into one number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

PLAYER_ID_COL = "player_id"
POSITION_COL = "position"
GAMEWEEK_COL = "gameweek"
PREDICTED_COL = "expected_points"
ACTUAL_COL = "total_points"

__all__ = [
    "PLAYER_ID_COL",
    "POSITION_COL",
    "GAMEWEEK_COL",
    "PREDICTED_COL",
    "ACTUAL_COL",
    "AccuracyReport",
    "player_accuracy",
    "BiasReport",
    "bias_by_group",
    "CalibrationReport",
    "component_calibration",
    "CaptaincyHitRateResult",
    "captaincy_hit_rate",
]


def _merge_predictions_and_actuals(
    predictions: pd.DataFrame, actuals: pd.DataFrame, predicted_col: str, actual_col: str
) -> pd.DataFrame:
    merged = predictions.merge(
        actuals[[PLAYER_ID_COL, GAMEWEEK_COL, actual_col]],
        on=[PLAYER_ID_COL, GAMEWEEK_COL],
        how="inner",
    )
    if merged.empty:
        raise ValueError(
            "no overlapping (player_id, gameweek) rows between predictions and actuals"
        )
    merged["error"] = merged[predicted_col] - merged[actual_col]
    return merged


@dataclass(frozen=True)
class AccuracyReport:
    """Player-level prediction error, overall and per position (BUILD_PLAN 3.2)."""

    overall_mae: float
    overall_rmse: float
    by_position: pd.DataFrame  # columns: position, mae, rmse, n


def player_accuracy(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
) -> AccuracyReport:
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)
    overall_mae = float(merged["error"].abs().mean())
    overall_rmse = float(np.sqrt((merged["error"] ** 2).mean()))

    by_position = (
        merged.groupby(POSITION_COL)["error"]
        .agg(
            mae=lambda e: float(e.abs().mean()),
            rmse=lambda e: float(np.sqrt((e**2).mean())),
            n="count",
        )
        .reset_index()
    )
    return AccuracyReport(
        overall_mae=overall_mae, overall_rmse=overall_rmse, by_position=by_position
    )


@dataclass(frozen=True)
class BiasReport:
    """Residual (predicted − actual) analysis per group — detects systematic over/under-rating of
    a position or price tier, not just average error (BUILD_PLAN 3.2). ``severe`` flags groups
    where a one-sample t-test rejects "mean residual is zero" at ``severity_p_threshold`` — the
    signal the Phase 3.6 Definition-of-Done gate checks."""

    by_group: (
        pd.DataFrame
    )  # columns: <group_col>, mean_residual, std_residual, n, t_stat, p_value, severe


def bias_by_group(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    group_col: str,
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
    severity_p_threshold: float = 0.01,
) -> BiasReport:
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)
    rows = []
    for group, g in merged.groupby(group_col):
        residuals = g["error"].to_numpy(dtype=float)
        n = len(residuals)
        mean_residual = float(residuals.mean())
        std_residual = float(residuals.std(ddof=1)) if n > 1 else 0.0
        if n > 1 and std_residual > 0:
            t_stat, p_value = scipy_stats.ttest_1samp(residuals, popmean=0.0)
            t_stat, p_value = float(t_stat), float(p_value)
        else:
            t_stat, p_value = float("nan"), float("nan")
        rows.append(
            {
                group_col: group,
                "mean_residual": mean_residual,
                "std_residual": std_residual,
                "n": n,
                "t_stat": t_stat,
                "p_value": p_value,
                "severe": bool(p_value < severity_p_threshold) if not np.isnan(p_value) else False,
            }
        )
    return BiasReport(by_group=pd.DataFrame(rows))


@dataclass(frozen=True)
class CalibrationReport:
    """Reliability table: bucket predicted probabilities into bins, compare mean predicted vs
    actual observed frequency per bin (BUILD_PLAN 3.2). ``mean_absolute_calibration_error`` is the
    n-weighted mean absolute gap between predicted and observed per bin — a single number for how
    well- or mis-calibrated this component currently is, feeding the Phase 3.6 gate."""

    by_bin: pd.DataFrame  # columns: bin, predicted_mean, actual_rate, n
    mean_absolute_calibration_error: float


def component_calibration(
    predicted_probability: pd.Series, actual_outcome: pd.Series, n_bins: int = 10
) -> CalibrationReport:
    """Generic reliability check for any component that outputs a probability against a binary
    realised outcome (e.g. predicted clean-sheet probability vs whether a clean sheet actually
    happened, or predicted defensive-contribution-threshold probability vs whether the threshold
    was actually cleared) — the caller picks which predicted/actual pair to check."""
    if len(predicted_probability) != len(actual_outcome):
        raise ValueError("predicted_probability and actual_outcome must be the same length")
    df = pd.DataFrame(
        {
            "predicted": np.asarray(predicted_probability, dtype=float),
            "actual": np.asarray(actual_outcome, dtype=float),
        }
    )
    df["bin"] = pd.cut(df["predicted"], bins=np.linspace(0, 1, n_bins + 1), include_lowest=True)
    by_bin = (
        df.groupby("bin", observed=True)
        .agg(
            predicted_mean=("predicted", "mean"),
            actual_rate=("actual", "mean"),
            n=("actual", "count"),
        )
        .reset_index()
    )
    weights = by_bin["n"] / by_bin["n"].sum()
    mace = float(((by_bin["predicted_mean"] - by_bin["actual_rate"]).abs() * weights).sum())
    return CalibrationReport(by_bin=by_bin, mean_absolute_calibration_error=mace)


@dataclass(frozen=True)
class CaptaincyHitRateResult:
    """BUILD_PLAN 3.2 — the single most decision-relevant metric. Two rates tracked side by side:
    ``raw_hit_rate`` (post-deadline misfortune — an unexpected injury, a red card — still counts as
    a miss, because a real captaincy tool has to live with that risk every week) and
    ``played_as_expected_hit_rate`` (restricted to gameweeks where the recommended captain actually
    got close to their expected minutes), so "the model picked wrong" and "the pick was right but
    football happened" don't collapse into one number."""

    per_gameweek: (
        pd.DataFrame
    )  # gameweek, recommended_player_id, actual_top_scorer_id, hit, played_as_expected
    raw_hit_rate: float
    played_as_expected_hit_rate: float


def captaincy_hit_rate(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    starting_xi_by_gameweek: dict[int, set[int]],
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
    minutes_col: str = "minutes",
    expected_minutes_col: str = "expected_minutes",
    played_as_expected_minutes_ratio: float = 0.75,
) -> CaptaincyHitRateResult:
    """``starting_xi_by_gameweek`` maps gameweek -> the actual starting XI's player ids — the
    decision-realistic "eligible options" (BUILD_PLAN 3.2: bench players were never really in
    contention for the armband). The recommended captain is whichever eligible player the
    predictions rank highest that gameweek; "played as expected" means their actual minutes came in
    at or above ``played_as_expected_minutes_ratio`` of their predicted expected minutes (falls
    back to assuming a full match was expected if ``expected_minutes_col`` isn't present) — a
    proxy for "the pick was right but football happened" vs "the pick itself was wrong".
    """
    rows = []
    for gw, eligible in starting_xi_by_gameweek.items():
        preds_gw = predictions[
            (predictions[GAMEWEEK_COL] == gw) & (predictions[PLAYER_ID_COL].isin(eligible))
        ]
        acts_gw = actuals[(actuals[GAMEWEEK_COL] == gw) & (actuals[PLAYER_ID_COL].isin(eligible))]
        if preds_gw.empty or acts_gw.empty:
            continue

        recommended_id = int(preds_gw.loc[preds_gw[predicted_col].idxmax(), PLAYER_ID_COL])
        actual_top_scorer_id = int(acts_gw.loc[acts_gw[actual_col].idxmax(), PLAYER_ID_COL])
        hit = recommended_id == actual_top_scorer_id

        recommended_actual = acts_gw[acts_gw[PLAYER_ID_COL] == recommended_id]
        recommended_predicted = preds_gw[preds_gw[PLAYER_ID_COL] == recommended_id]
        played_as_expected = False
        if not recommended_actual.empty:
            actual_minutes = float(recommended_actual[minutes_col].iloc[0])
            if (
                expected_minutes_col in recommended_predicted.columns
                and not recommended_predicted.empty
            ):
                expected_minutes = float(recommended_predicted[expected_minutes_col].iloc[0])
            else:
                expected_minutes = 90.0
            played_as_expected = (
                expected_minutes > 0
                and actual_minutes >= played_as_expected_minutes_ratio * expected_minutes
            )

        rows.append(
            {
                GAMEWEEK_COL: gw,
                "recommended_player_id": recommended_id,
                "actual_top_scorer_id": actual_top_scorer_id,
                "hit": hit,
                "played_as_expected": played_as_expected,
            }
        )

    per_gameweek = pd.DataFrame(rows)
    if per_gameweek.empty:
        raise ValueError("no gameweeks with overlapping eligible predictions and actuals")

    raw_hit_rate = float(per_gameweek["hit"].mean())
    played_subset = per_gameweek[per_gameweek["played_as_expected"]]
    played_as_expected_hit_rate = (
        float(played_subset["hit"].mean()) if not played_subset.empty else float("nan")
    )
    return CaptaincyHitRateResult(
        per_gameweek=per_gameweek,
        raw_hit_rate=raw_hit_rate,
        played_as_expected_hit_rate=played_as_expected_hit_rate,
    )
