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
    "TopNReport",
    "top_n_mean_actual",
    "RankCorrelationReport",
    "rank_correlation",
    "floor_ceiling_coverage",
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
    a position or price tier, not just average error (BUILD_PLAN 3.2). ``severe`` requires **both**
    statistical significance (a one-sample t-test rejects "mean residual is zero" at
    ``severity_p_threshold``) **and** an effect-size floor (``abs(mean_residual)`` exceeds
    ``min_absolute_effect`` points, or ``min_relative_effect`` of the group's mean actual points) —
    significance alone isn't enough, since with a large enough sample any nonzero bias becomes
    "significant" regardless of whether it's big enough to matter (see
    planning/ENGINE_IMPROVEMENTS.md Correction 2, which found the p-value-only version flagged a
    4%-of-mean MID bias while clearing a 6.8%-of-mean FWD bias purely because MID had 4x the
    sample). ``severe`` is the signal the Phase 3.6 Definition-of-Done gate checks."""

    by_group: (
        pd.DataFrame
    )  # columns: <group_col>, mean_residual, mean_actual, std_residual, n, t_stat, p_value,
    # effect_size_floor, severe


def bias_by_group(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    group_col: str,
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
    severity_p_threshold: float = 0.01,
    min_absolute_effect: float = 0.25,
    min_relative_effect: float = 0.10,
) -> BiasReport:
    """``min_absolute_effect`` (points) and ``min_relative_effect`` (fraction of the group's mean
    actual points) jointly set the effect-size floor a group's mean residual must clear — whichever
    is larger — before statistical significance alone can flag it ``severe``."""
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)
    rows = []
    for group, g in merged.groupby(group_col):
        residuals = g["error"].to_numpy(dtype=float)
        n = len(residuals)
        mean_residual = float(residuals.mean())
        mean_actual = float(g[actual_col].abs().mean())
        std_residual = float(residuals.std(ddof=1)) if n > 1 else 0.0
        if n > 1 and std_residual > 0:
            t_stat, p_value = scipy_stats.ttest_1samp(residuals, popmean=0.0)
            t_stat, p_value = float(t_stat), float(p_value)
        else:
            t_stat, p_value = float("nan"), float("nan")
        effect_size_floor = max(min_absolute_effect, min_relative_effect * mean_actual)
        significant = not np.isnan(p_value) and p_value < severity_p_threshold
        severe = bool(significant and abs(mean_residual) > effect_size_floor)
        rows.append(
            {
                group_col: group,
                "mean_residual": mean_residual,
                "mean_actual": mean_actual,
                "std_residual": std_residual,
                "n": n,
                "t_stat": t_stat,
                "p_value": p_value,
                "effect_size_floor": effect_size_floor,
                "severe": severe,
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


@dataclass(frozen=True)
class TopNReport:
    """Mean actual points of the top-N predicted picks, per gameweek then averaged across
    gameweeks (ENGINE_IMPROVEMENTS.md 2.1) — the metric that actually surfaces skill at the top of
    the distribution, which pooled MAE/Spearman can't see: every real FPL decision (captaincy,
    transfers, chips) only ever reads the top of the ranking, so a model can be simultaneously
    worse at whole-pool rank correlation and better at the decisions that matter."""

    by_n: pd.DataFrame  # columns: n, mean_actual, n_gameweeks


def top_n_mean_actual(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    ns: tuple[int, ...] = (1, 5, 10, 20),
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
    gameweek_col: str = GAMEWEEK_COL,
) -> TopNReport:
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)
    rows = []
    for n in ns:
        per_gw_means = [
            float(g.nlargest(min(n, len(g)), predicted_col)[actual_col].mean())
            for _, g in merged.groupby(gameweek_col)
        ]
        rows.append(
            {
                "n": n,
                "mean_actual": float(np.mean(per_gw_means)) if per_gw_means else float("nan"),
                "n_gameweeks": len(per_gw_means),
            }
        )
    return TopNReport(by_n=pd.DataFrame(rows))


@dataclass(frozen=True)
class RankCorrelationReport:
    """Spearman rank correlation between predicted and actual points — overall, and (optionally)
    broken out by an arbitrary group column (position, price tier, ...). A single pooled Spearman
    is dominated by the will-they-play axis (ENGINE_IMPROVEMENTS.md 2.1): pass ``minutes_col`` to
    restrict to players who actually started, separating "ranks footballers well" from "ranks
    availability well"."""

    overall: float
    by_group: pd.DataFrame | None  # columns: <group_col>, spearman, n


def rank_correlation(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
    group_col: str | None = None,
    minutes_col: str | None = None,
) -> RankCorrelationReport:
    """``minutes_col``, if given, filters to rows with minutes > 0 before computing correlation —
    the "restricted to starters" variant (``minutes_col`` is read from ``actuals``, matching
    ``captaincy_hit_rate``'s convention). ``group_col``, if given, additionally breaks the
    correlation out per group (e.g. per position or per price tier)."""
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)
    if minutes_col is not None:
        merged = merged.merge(
            actuals[[PLAYER_ID_COL, GAMEWEEK_COL, minutes_col]],
            on=[PLAYER_ID_COL, GAMEWEEK_COL],
        )
        merged = merged[merged[minutes_col] > 0]

    overall = (
        float(scipy_stats.spearmanr(merged[predicted_col], merged[actual_col]).correlation)
        if len(merged) >= 2
        else float("nan")
    )

    by_group = None
    if group_col is not None:
        rows = [
            {
                group_col: group,
                "spearman": (
                    float(scipy_stats.spearmanr(g[predicted_col], g[actual_col]).correlation)
                    if len(g) >= 2
                    else float("nan")
                ),
                "n": len(g),
            }
            for group, g in merged.groupby(group_col)
        ]
        by_group = pd.DataFrame(rows)
    return RankCorrelationReport(overall=overall, by_group=by_group)


def floor_ceiling_coverage(floor: pd.Series, ceiling: pd.Series, actual: pd.Series) -> float:
    """Fraction of rows where the realised outcome fell within ``[floor, ceiling]`` — checks
    whether the simulation layer's (``engine/simulate.py``, BUILD_PLAN 2.9) spread is honest, not
    just its mean (ENGINE_IMPROVEMENTS.md 2.1). Pair with ``component_calibration`` on
    ``prob_big_haul`` vs ``actual >= 10`` for the other half of that same recommendation — both
    read straight off ``PlayerSimulationSummary``, no new modelling required."""
    floor_arr = np.asarray(floor, dtype=float)
    ceiling_arr = np.asarray(ceiling, dtype=float)
    actual_arr = np.asarray(actual, dtype=float)
    if not (len(floor_arr) == len(ceiling_arr) == len(actual_arr)):
        raise ValueError("floor, ceiling, and actual must be the same length")
    within = (actual_arr >= floor_arr) & (actual_arr <= ceiling_arr)
    return float(within.mean())
