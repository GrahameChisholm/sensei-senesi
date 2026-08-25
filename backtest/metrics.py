"""Accuracy, bias, calibration, and captaincy hit-rate metrics (3.2).

A single "is it right" number hides which part of the engine is actually failing, so this tracks
several levels separately: player-level MAE/RMSE overall and per position, systematic bias by
group (residual analysis, not just average error), per-component probability calibration ("when
the model says 40% clean sheet, do clean sheets actually happen ~40% of the time?"), and the
single most decision-relevant metric — captaincy hit-rate, tracked as two side-by-side variants so
a bad pick and a good pick undone by an in-game injury don't collapse into one number.

It also carries a family of aggregate, per-fixture and pool-wide plausibility checks (T-J,
``planning/ENGINE_AUDIT_FIXES-implementation.md``): every check above scores per-player accuracy,
none of them sum or average across a fixture or the whole pool, which is exactly how a real audit
found several defects (minutes mass on the pitch well short of 22 per fixture, bonus not summing
to a real match's 6 points, pool-wide minutes-bucket shares far from the prior season's empirical
split, understated goalkeeper saves, and a spurious horizon decay) that no per-player metric here
ever caught. These checks report raw measurements only, the same split as :class:`CalibrationReport`
carrying MACE while ``backtest.gate`` owns the acceptance threshold; they do not embed a pass/fail
verdict themselves.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.metrics import roc_auc_score

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
    "DecisionSetRankReport",
    "decision_set_rank_correlation",
    "rate_calibration_at_realised_minutes",
    "floor_ceiling_coverage",
    "MeanCalibrationReport",
    "mean_calibration",
    "MinutesDiagnosticsReport",
    "minutes_model_diagnostics",
    "BrierComparisonReport",
    "brier_vs_constant",
    "FixtureMinutesCoverageReport",
    "fixture_minutes_coverage",
    "FixtureBonusTotalReport",
    "fixture_bonus_total",
    "MinutesBucketShareReport",
    "minutes_bucket_pool_shares",
    "GoalkeeperSavesReport",
    "goalkeeper_saves_plausibility",
    "HorizonMonotonicityReport",
    "horizon_minutes_monotonicity",
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
    minutes_col: str | None = None,
    min_minutes: float = 60.0,
) -> BiasReport:
    """``min_absolute_effect`` (points) and ``min_relative_effect`` (fraction of the group's mean
    actual points) jointly set the effect-size floor a group's mean residual must clear — whichever
    is larger — before statistical significance alone can flag it ``severe``. Pass
    ``min_relative_effect=0.0`` for a grouping like price tier where the *absolute* error is what
    costs points regardless of scale (ENGINE_IMPROVEMENTS_3.md B.2 — a relative floor that scales
    with the group's own mean actual is backwards for a premium price tier, where a large mean
    actual makes a large absolute bias easier, not harder, to clear).

    ``group_col`` may live in ``predictions`` (e.g. "position", already on every prediction row) or
    only in ``actuals`` (e.g. a price tier derived from ground truth) — it's merged in from
    ``actuals`` automatically when not already present in ``predictions``.

    ``minutes_col`` (ENGINE_IMPROVEMENTS_5.md Tier 0.1), if given, restricts the residual to rows
    where that column in ``actuals`` is at least ``min_minutes`` before grouping.

    **Only pass a conditional prediction alongside it** (Tier 2.1 correction). Restricting to rows
    where the player played and then scoring an *unconditional* prediction measures the act of
    conditioning, not the model: since ``expected_points`` is P(plays) times E[points | plays],
    selecting the rows where they played selects the branch on which it was always going to look
    low. A simulated model handed P(plays) and E[points | plays] exactly scores **-1.31** on that
    statistic, worse than the engine's -0.96, so it is unpassable by construction. Scored against
    ``conditional_expected_points`` (``engine.pipeline``, Tier 2.1) the same rows read -0.088, which
    is a real and informative check. Left as ``None`` (the default) this function behaves exactly as
    it did before either finding, so existing callers are unaffected.
    """
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)
    if minutes_col is not None:
        merged = merged.merge(
            actuals[[PLAYER_ID_COL, GAMEWEEK_COL, minutes_col]], on=[PLAYER_ID_COL, GAMEWEEK_COL]
        )
        merged = merged[merged[minutes_col] >= min_minutes]
    if group_col not in merged.columns:
        merged = merged.merge(
            actuals[[PLAYER_ID_COL, GAMEWEEK_COL, group_col]], on=[PLAYER_ID_COL, GAMEWEEK_COL]
        )
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
class BrierComparisonReport:
    """Does this probability component beat predicting the sample's own base rate for every row
    (ENGINE_IMPROVEMENTS_3.md B.3)? A reliability curve (:class:`CalibrationReport`) can look
    reasonable bin-by-bin while the component is still, in aggregate, worse than a constant —
    exactly what the shipped clean-sheet component turned out to be at team level (Brier 0.1895 vs
    a constant-base-rate Brier of 0.1872) despite a real, measurable AUC. ``beats_constant`` is the
    floor MACE alone cannot express."""

    brier: float
    constant_brier: float
    beats_constant: bool


def brier_vs_constant(
    predicted_probability: pd.Series, actual_outcome: pd.Series
) -> BrierComparisonReport:
    """Compare a component's own Brier score against the Brier score of predicting the sample's
    realised base rate (``actual_outcome.mean()``) for every row — the simplest possible baseline
    for any probability forecast."""
    p = np.asarray(predicted_probability, dtype=float)
    y = np.asarray(actual_outcome, dtype=float)
    if len(p) != len(y):
        raise ValueError("predicted_probability and actual_outcome must be the same length")
    brier = float(np.mean((p - y) ** 2))
    base_rate = float(np.mean(y))
    constant_brier = float(np.mean((base_rate - y) ** 2))
    return BrierComparisonReport(
        brier=brier, constant_brier=constant_brier, beats_constant=brier < constant_brier
    )


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


def _spearman_or_nan(a: pd.Series, b: pd.Series) -> float:
    return float(scipy_stats.spearmanr(a, b).correlation) if len(a) >= 2 else float("nan")


def _by_group_spearman(
    merged: pd.DataFrame, group_col: str, predicted_col: str, actual_col: str
) -> pd.DataFrame:
    rows = [
        {
            group_col: group,
            "spearman": _spearman_or_nan(g[predicted_col], g[actual_col]),
            "n": len(g),
        }
        for group, g in merged.groupby(group_col)
    ]
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class RankCorrelationReport:
    """Spearman rank correlation between predicted and actual points — pooled (every row) and,
    when ``minutes_col`` is supplied, separately restricted to players who actually started
    (ENGINE_IMPROVEMENTS_2.md Correction 5 / A.2). A single pooled Spearman is dominated by the
    will-they-play axis (ENGINE_IMPROVEMENTS.md 2.1), so the two numbers answer different
    questions — "ranks footballers well" (starters-only) vs "ranks the whole pool, availability
    included, well" (pooled) — and collapsing them into one field previously meant the version
    that got reported (``overall``) was silently the restricted one whenever ``minutes_col`` was
    passed, which is exactly what happened in the first re-measurement pass.

    ``overall`` is always the pooled (unfiltered) figure. ``overall_starters_only`` and
    ``by_group_starters_only`` are populated only when ``minutes_col`` is given; otherwise both
    are ``None``, matching this report's pre-A.2 shape when no restriction was requested at all.
    """

    overall: float
    by_group: pd.DataFrame | None  # columns: <group_col>, spearman, n
    overall_starters_only: float | None = None
    by_group_starters_only: pd.DataFrame | None = None


def rank_correlation(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
    group_col: str | None = None,
    minutes_col: str | None = None,
) -> RankCorrelationReport:
    """``group_col``, if given, additionally breaks the pooled correlation out per group (e.g. per
    position or per price tier). ``minutes_col``, if given, filters to rows with minutes > 0 (read
    from ``actuals``, matching ``captaincy_hit_rate``'s convention) and reports that restricted
    correlation *separately*, in ``overall_starters_only``/``by_group_starters_only`` — it never
    replaces the pooled ``overall``/``by_group`` fields (see the report's own docstring)."""
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)

    overall = _spearman_or_nan(merged[predicted_col], merged[actual_col])
    by_group = (
        _by_group_spearman(merged, group_col, predicted_col, actual_col)
        if group_col is not None
        else None
    )

    overall_starters_only = None
    by_group_starters_only = None
    if minutes_col is not None:
        starters = merged.merge(
            actuals[[PLAYER_ID_COL, GAMEWEEK_COL, minutes_col]],
            on=[PLAYER_ID_COL, GAMEWEEK_COL],
        )
        starters = starters[starters[minutes_col] > 0]
        overall_starters_only = _spearman_or_nan(starters[predicted_col], starters[actual_col])
        if group_col is not None:
            by_group_starters_only = _by_group_spearman(
                starters, group_col, predicted_col, actual_col
            )

    return RankCorrelationReport(
        overall=overall,
        by_group=by_group,
        overall_starters_only=overall_starters_only,
        by_group_starters_only=by_group_starters_only,
    )


def rate_calibration_at_realised_minutes(
    predicted_quantity: pd.Series,
    predicted_minutes: pd.Series,
    realised_minutes: pd.Series,
    realised_count: pd.Series,
) -> MeanCalibrationReport:
    """Calibration of a component's underlying *rate*, with both the minutes model and the
    played-rows selection effect removed (ENGINE_IMPROVEMENTS_5.md Tier 2.3).

    Each row's implied per-90 rate is recovered as ``predicted_quantity / (predicted_minutes / 90)``
    and re-evaluated at the minutes the player actually played. A row the model expected to sit and
    who then did sit contributes 0 to both sides rather than being dropped, so nothing is
    conditioned on.

    This exists because neither existing instrument can isolate a rate model. Scored across all
    rows, :func:`mean_calibration` is dominated by the minutes model. Scored on played rows only,
    it is confounded by selection: ``predicted_quantity`` is an unconditional expectation, so
    restricting to rows where the player did play selects the branch on which it was always going
    to look low, which *masks* an over-prediction and *exaggerates* an under-prediction. On the real
    2025/26 walk-forward the played-rows view reported goals 5.7% over and assists 22.2% under,
    while this view reported goals **18% over** and assists **12% under** — and the all-rows view
    agreed with this one (24% over, 8% under), not with the played-rows one.
    """
    if not (
        len(predicted_quantity)
        == len(predicted_minutes)
        == len(realised_minutes)
        == len(realised_count)
    ):
        raise ValueError("all four series must be the same length")
    predicted_minutes_arr = np.asarray(predicted_minutes, dtype=float)
    quantity = np.asarray(predicted_quantity, dtype=float)
    implied_rate_per_90 = np.divide(
        quantity,
        predicted_minutes_arr / 90.0,
        out=np.zeros_like(quantity),
        where=predicted_minutes_arr > 0,
    )
    predicted_at_realised = implied_rate_per_90 * (np.asarray(realised_minutes, dtype=float) / 90.0)
    return mean_calibration(
        pd.Series(predicted_at_realised), pd.Series(np.asarray(realised_count, dtype=float))
    )


@dataclass(frozen=True)
class DecisionSetRankReport:
    """Rank correlation measured *within one gameweek's own shortlist* — the top ``top_n`` players
    by predicted points in that gameweek, scored against what they actually returned, then averaged
    across gameweeks (ENGINE_IMPROVEMENTS_5.md Tier 0.1).

    This exists because every pooled ranking metric in this module, including
    :class:`RankCorrelationReport`'s starters-only variant, answers a question no manager asks. A
    manager never sorts 600 players; they pick between the 15 or 20 the tool already surfaced. On
    the real 2025/26 walk-forward the engine scores pooled Spearman 0.636, starters-only 0.355,
    played-60+ 0.188, and **+0.049 within the top 20 it recommended** — and the pooled figure is
    matched by ``1 - p_zero`` alone (0.643), so it measures availability prediction rather than
    points prediction. A gate built on pooled numbers cannot fail on the shortlist being unordered,
    which is the engine's actual deficit, and so cannot reward fixing it.

    ``mean_spearman`` is the average of the per-gameweek within-shortlist correlations, not a
    correlation computed on the pooled shortlist rows: pooling across gameweeks would reintroduce
    between-gameweek variance (a high-scoring gameweek's mid-table player outscoring a low-scoring
    gameweek's best) that no single decision ever spans.

    ``share_positive`` guards against reading skill into noise. With a mean of 0.049, a standard
    deviation of 0.224 and only 22 of 35 gameweeks positive, the honest description is near-zero
    true skill at this resolution rather than a skilled model having a bad run.
    """

    top_n: int
    mean_spearman: float
    median_spearman: float
    std_spearman: float
    share_positive: float
    mean_absolute_error: float
    mean_bias: float
    n_gameweeks: int
    by_gameweek: pd.DataFrame  # columns: gameweek, spearman, mae, bias, n


def decision_set_rank_correlation(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    top_n: int = 20,
    predicted_col: str = PREDICTED_COL,
    actual_col: str = ACTUAL_COL,
) -> DecisionSetRankReport:
    """Per gameweek, take the ``top_n`` highest-predicted players and correlate their predicted
    points against their realised points, then summarise across gameweeks. See
    :class:`DecisionSetRankReport` for why this is scored per gameweek and then averaged rather
    than pooled.

    A gameweek contributing fewer than 2 shortlist rows, or one where every realised outcome ties
    (so a rank correlation is undefined), is skipped for the correlation but still contributes its
    error and bias, matching :func:`_spearman_or_nan`'s own degrade-rather-than-raise contract.
    """
    if top_n < 2:
        raise ValueError(f"top_n must be at least 2 to rank within a shortlist, got {top_n}")
    merged = _merge_predictions_and_actuals(predictions, actuals, predicted_col, actual_col)

    rows = []
    for gameweek, group in merged.groupby(GAMEWEEK_COL):
        shortlist = group.nlargest(top_n, predicted_col)
        if shortlist.empty:
            continue
        residuals = shortlist[predicted_col] - shortlist[actual_col]
        rows.append(
            {
                GAMEWEEK_COL: gameweek,
                "spearman": _spearman_or_nan(shortlist[predicted_col], shortlist[actual_col]),
                "mae": float(residuals.abs().mean()),
                "bias": float(residuals.mean()),
                "n": len(shortlist),
            }
        )

    by_gameweek = pd.DataFrame(rows, columns=[GAMEWEEK_COL, "spearman", "mae", "bias", "n"])
    scored = by_gameweek["spearman"].dropna()
    return DecisionSetRankReport(
        top_n=top_n,
        mean_spearman=float(scored.mean()) if len(scored) else float("nan"),
        median_spearman=float(scored.median()) if len(scored) else float("nan"),
        std_spearman=float(scored.std(ddof=1)) if len(scored) > 1 else float("nan"),
        share_positive=float((scored > 0).mean()) if len(scored) else float("nan"),
        mean_absolute_error=(
            float(by_gameweek["mae"].mean()) if len(by_gameweek) else float("nan")
        ),
        mean_bias=float(by_gameweek["bias"].mean()) if len(by_gameweek) else float("nan"),
        n_gameweeks=int(len(scored)),
        by_gameweek=by_gameweek,
    )


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


@dataclass(frozen=True)
class MeanCalibrationReport:
    """Aggregate calibration for a continuous (non-probability) predicted quantity — mean
    predicted vs mean actual, and the gap between them (ENGINE_IMPROVEMENTS_2.md A.4). Goals,
    assists, and bonus aren't bounded-[0,1] probabilities the way clean-sheet/DC-threshold outputs
    are, so :func:`component_calibration`'s binned reliability curve doesn't apply to them (its
    bins span exactly [0, 1] — a predicted value above 1.0, entirely normal for e.g.
    ``expected_goals`` on a big-favourite fixture, would silently fall into the top bin's edge or
    be miscounted rather than raise). This is a coarser, mean-only check — informational, not fed
    to the Definition-of-Done gate the way :class:`CalibrationReport` is."""

    mean_predicted: float
    mean_actual: float
    absolute_gap: float
    relative_gap: float  # abs gap / |mean_actual|; NaN when mean_actual == 0


def mean_calibration(predicted: pd.Series, actual: pd.Series) -> MeanCalibrationReport:
    if len(predicted) != len(actual):
        raise ValueError("predicted and actual must be the same length")
    mean_predicted = float(np.mean(np.asarray(predicted, dtype=float)))
    mean_actual = float(np.mean(np.asarray(actual, dtype=float)))
    absolute_gap = abs(mean_predicted - mean_actual)
    relative_gap = absolute_gap / abs(mean_actual) if mean_actual else float("nan")
    return MeanCalibrationReport(
        mean_predicted=mean_predicted,
        mean_actual=mean_actual,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
    )


@dataclass(frozen=True)
class MinutesDiagnosticsReport:
    """First-class scoring for the minutes model — the engine's dominant component
    (ENGINE_IMPROVEMENTS.md 1.1 / ENGINE_IMPROVEMENTS_2.md A.3), previously scored only as a
    downstream contributor to overall MAE with no diagnostic of its own.

    ``predicted_points_mass_per_scored_row`` (not the raw total mass) is the sample-size-invariant
    number to track across runs — Correction 3 found the raw total wasn't comparable once the
    scored sample size changed between passes. ``auc_played_at_all`` is computed from ``1 -
    p_zero`` (the model's own bucket probability), not from ``expected_minutes`` — the latter is a
    proxy and scores measurably worse.
    """

    zero_minute_share: float
    mean_expected_minutes_on_zero_rows: float
    predicted_points_mass_on_zero_rows: float
    predicted_points_mass_per_scored_row: float
    auc_played_at_all: float
    n_scored_rows: int
    # B2: which components the zero-minute mass is actually made of. The aggregate number says
    # how much leaks; only this says *where from*, and therefore which component to gate. The
    # minutes model itself is well calibrated (ENGINE_IMPROVEMENTS_3.md C.1), so a mass above
    # target is by elimination a downstream component not fully gated by availability -- but
    # which one has only ever been established by an ad hoc script, never by the report.
    zero_minute_mass_by_component: pd.DataFrame | None = None
    # B2: the mass a *perfectly calibrated* model would still carry at this level of
    # discrimination. A model that predicts 30% play probability for a group of whom 30% really do
    # play is correct, and still assigns points to the 70% who didn't — so the raw mass is not a
    # defect measure, and driving it to zero would require an under-confident model, not a better
    # one. Only `zero_minute_mass_excess` (observed minus floor) is attributable to
    # miscalibration; the floor itself falls only by improving discrimination.
    calibrated_floor_mass_per_scored_row: float = float("nan")

    @property
    def zero_minute_mass_excess(self) -> float:
        return self.predicted_points_mass_per_scored_row - self.calibrated_floor_mass_per_scored_row


COMPONENT_POINT_COLUMNS = (
    "appearance",
    "goals",
    "assists",
    "clean_sheet",
    "goals_conceded",
    "defensive_contribution",
    "saves",
    "bonus",
    "cards",
    "penalty_misses",
    "own_goals",
)


def minutes_model_diagnostics(
    predictions: pd.DataFrame,
    actuals: pd.DataFrame,
    minutes_col: str = "minutes",
    p_zero_col: str = "p_zero",
    expected_minutes_col: str = "expected_minutes",
    predicted_points_col: str = PREDICTED_COL,
    component_columns: Sequence[str] = COMPONENT_POINT_COLUMNS,
) -> MinutesDiagnosticsReport:
    merged = predictions.merge(
        actuals[[PLAYER_ID_COL, GAMEWEEK_COL, minutes_col]], on=[PLAYER_ID_COL, GAMEWEEK_COL]
    )
    n = len(merged)
    if n == 0:
        raise ValueError(
            "no overlapping (player_id, gameweek) rows between predictions and actuals"
        )

    zero_rows = merged[merged[minutes_col] == 0]
    mass_on_zero_rows = float(zero_rows[predicted_points_col].sum())
    played = (merged[minutes_col] > 0).astype(int)
    p_played = 1.0 - merged[p_zero_col]
    auc = float(roc_auc_score(played, p_played)) if played.nunique() > 1 else float("nan")

    return MinutesDiagnosticsReport(
        zero_minute_share=float(len(zero_rows) / n),
        mean_expected_minutes_on_zero_rows=(
            float(zero_rows[expected_minutes_col].mean()) if len(zero_rows) else float("nan")
        ),
        predicted_points_mass_on_zero_rows=mass_on_zero_rows,
        predicted_points_mass_per_scored_row=mass_on_zero_rows / n,
        auc_played_at_all=auc,
        n_scored_rows=n,
        zero_minute_mass_by_component=_zero_minute_mass_by_component(
            zero_rows, component_columns, n
        ),
        calibrated_floor_mass_per_scored_row=_calibrated_floor_mass(
            merged, minutes_col, p_zero_col, predicted_points_col
        ),
    )


def _calibrated_floor_mass(
    merged: pd.DataFrame,
    minutes_col: str,
    p_zero_col: str,
    predicted_points_col: str,
    n_bins: int = 10,
) -> float:
    """The zero-minute predicted-points mass a perfectly calibrated model would still carry.

    Within each play-probability bin, rescale the predictions by the ratio of the bin's *realised*
    play rate to its *predicted* one — i.e. ask what the mass would have been had the model been
    exactly right about how often this kind of player appears, holding its ability to tell the
    bins apart fixed. What remains is irreducible: the price of correctly expressing uncertainty.
    """
    played = merged[minutes_col] > 0
    p_played = 1.0 - merged[p_zero_col]
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = np.clip(np.digitize(p_played, edges[1:-1]), 0, n_bins - 1)

    floor = 0.0
    for b in range(n_bins):
        in_bin = bins == b
        if not in_bin.any():
            continue
        predicted = float(p_played[in_bin].mean())
        if predicted <= 0:
            continue
        scale = float(played[in_bin].mean()) / predicted
        floor += float(merged.loc[in_bin & ~played, predicted_points_col].sum()) * scale
    return floor / len(merged)


def _zero_minute_mass_by_component(
    zero_rows: pd.DataFrame, component_columns: Sequence[str], n_scored_rows: int
) -> pd.DataFrame | None:
    """Split the zero-minute predicted-points mass across the components that produced it.

    Reported per scored row (the sample-size-invariant unit the aggregate figure already uses) and
    as a share of the mass, largest first, so the component to gate is the top line rather than
    something to be inferred. Returns ``None`` when the predictions frame carries no component
    columns at all — several callers (baselines, the simulation frame) legitimately don't.
    """
    present = [column for column in component_columns if column in zero_rows.columns]
    if not present or zero_rows.empty:
        return None
    totals = zero_rows[present].sum()
    total_mass = float(totals.sum())
    frame = pd.DataFrame(
        {
            "component": totals.index,
            "mass": totals.to_numpy(dtype=float),
            "per_scored_row": totals.to_numpy(dtype=float) / n_scored_rows,
            "share_of_zero_minute_mass": (
                totals.to_numpy(dtype=float) / total_mass if total_mass else np.nan
            ),
        }
    )
    return frame.sort_values("mass", ascending=False).reset_index(drop=True)


@dataclass(frozen=True)
class FixtureMinutesCoverageReport:
    """Per-fixture sum of ``p_60_plus`` across both squads (ENGINE_AUDIT_FIXES T-J item 1).

    A real match has 22 players on the pitch at the hour mark, before injury substitutions, so the
    minutes model's own bucket probabilities should sum close to 22 within any one fixture. No
    per-player accuracy metric above can see this: each scores one row at a time and never sums
    across the ~22 rows that share a match, which is exactly how the current shortfall (measured
    at 18.2) went unnoticed."""

    by_fixture: pd.DataFrame  # columns: <fixture_id_col>, sum_p_60_plus, n_players, gap
    mean_absolute_gap: float
    max_absolute_gap: float


def fixture_minutes_coverage(
    predictions: pd.DataFrame,
    fixture_id_col: str = "fixture_id",
    p_60_plus_col: str = "p_60_plus",
    target_sum: float = 22.0,
) -> FixtureMinutesCoverageReport:
    """``fixture_id_col`` must identify one real match shared by every player from both squads
    (e.g. a ``(team, opponent, gameweek)`` composite). This function only needs one column value
    common to everyone on the pitch that match, however the caller derives it."""
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    by_fixture = (
        predictions.groupby(fixture_id_col)[p_60_plus_col]
        .agg(sum_p_60_plus="sum", n_players="count")
        .reset_index()
    )
    by_fixture["gap"] = by_fixture["sum_p_60_plus"] - target_sum
    return FixtureMinutesCoverageReport(
        by_fixture=by_fixture,
        mean_absolute_gap=float(by_fixture["gap"].abs().mean()),
        max_absolute_gap=float(by_fixture["gap"].abs().max()),
    )


@dataclass(frozen=True)
class FixtureBonusTotalReport:
    """Per-fixture total expected bonus (ENGINE_AUDIT_FIXES T-J item 2).

    FPL awards exactly 3 + 2 + 1 = 6 bonus points per match, so total expected bonus summed across
    both squads in one fixture should land close to 6.0. Aggregate expected bonus can look roughly
    conserved (60.70 across 10 GW1 fixtures, close to the true 60) while still being badly wrong
    fixture by fixture and player by player, which is why this checks the per-fixture total rather
    than trusting the pool-wide sum alone."""

    by_fixture: pd.DataFrame  # columns: <fixture_id_col>, sum_bonus, n_players, gap
    mean_absolute_gap: float
    max_absolute_gap: float


def fixture_bonus_total(
    predictions: pd.DataFrame,
    fixture_id_col: str = "fixture_id",
    bonus_col: str = "bonus",
    target_sum: float = 6.0,
) -> FixtureBonusTotalReport:
    """``bonus_col`` defaults to the points-scale bonus column (``breakdown.bonus`` in
    ``engine/pipeline.py``'s output), the quantity that actually sums to 6 per real match, not the
    pre-allocation raw regression output."""
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    by_fixture = (
        predictions.groupby(fixture_id_col)[bonus_col]
        .agg(sum_bonus="sum", n_players="count")
        .reset_index()
    )
    by_fixture["gap"] = by_fixture["sum_bonus"] - target_sum
    return FixtureBonusTotalReport(
        by_fixture=by_fixture,
        mean_absolute_gap=float(by_fixture["gap"].abs().mean()),
        max_absolute_gap=float(by_fixture["gap"].abs().max()),
    )


@dataclass(frozen=True)
class MinutesBucketShareReport:
    """Pool-wide mean minutes-bucket probabilities against the prior season's real empirical
    shares (ENGINE_AUDIT_FIXES T-J item 3).

    Three per-row probabilities that each individually look plausible can still be collectively
    wrong across the whole player pool. Pool-wide ``p_1_to_59`` was measured at 2.00x the real
    2025-26 rate of 0.124 even though no single row's bucket split looks obviously broken. The
    empirical shares are supplied by the caller (see ``backtest.gate``'s documented 2025-26
    constants) rather than hardcoded here, so this function has no dependency on any particular
    season's cache."""

    predicted_shares: dict[str, float]
    empirical_shares: dict[str, float]
    absolute_gaps: dict[str, float]


def minutes_bucket_pool_shares(
    predictions: pd.DataFrame,
    empirical_shares: Mapping[str, float],
    p_zero_col: str = "p_zero",
    p_1_to_59_col: str = "p_1_to_59",
    p_60_plus_col: str = "p_60_plus",
) -> MinutesBucketShareReport:
    """``empirical_shares`` must supply ``"p_zero"``, ``"p_1_to_59"`` and ``"p_60_plus"`` keys."""
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    predicted_shares = {
        "p_zero": float(predictions[p_zero_col].mean()),
        "p_1_to_59": float(predictions[p_1_to_59_col].mean()),
        "p_60_plus": float(predictions[p_60_plus_col].mean()),
    }
    absolute_gaps = {
        bucket: abs(predicted_shares[bucket] - float(empirical_shares[bucket]))
        for bucket in predicted_shares
    }
    return MinutesBucketShareReport(
        predicted_shares=predicted_shares,
        empirical_shares={bucket: float(value) for bucket, value in empirical_shares.items()},
        absolute_gaps=absolute_gaps,
    )


@dataclass(frozen=True)
class GoalkeeperSavesReport:
    """Mean predicted saves for goalkeepers the minutes model itself rates likely to play 60+
    minutes, against the prior season's real per-match rate (ENGINE_AUDIT_FIXES T-J item 4).

    Real goalkeepers playing 60+ minutes in 2025-26 made 2.78 saves on average; the own-rate
    fallback in ``engine/models/saves.py`` currently implies 1.63 per 90, a defect a per-player MAE
    check never isolates because it is swamped by every other component's error."""

    mean_predicted_saves: float
    empirical_saves_per_match: float
    absolute_gap: float
    relative_gap: float  # abs gap / empirical_saves_per_match
    n_players: int


def goalkeeper_saves_plausibility(
    predictions: pd.DataFrame,
    empirical_saves_per_match: float,
    position_col: str = "position",
    saves_col: str = "expected_saves",
    p_60_plus_col: str = "p_60_plus",
    p_60_plus_threshold: float = 0.5,
) -> GoalkeeperSavesReport:
    """Restricted to goalkeeper rows with ``p_60_plus`` at or above ``p_60_plus_threshold``, the
    closest proxy for "played 60+" a predictions-only frame (no ground-truth minutes required) can
    offer. ``saves_col`` defaults to the raw expected-save count
    (``raw_components["expected_saves"]`` in ``engine/pipeline.py``'s output), not the
    floor-divided saves-points column, since the empirical reference is a save count, not a
    points value."""
    gk_rows = predictions[
        (predictions[position_col] == "GK") & (predictions[p_60_plus_col] >= p_60_plus_threshold)
    ]
    if gk_rows.empty:
        raise ValueError(
            "no goalkeeper rows with p_60_plus at or above p_60_plus_threshold to score"
        )
    mean_predicted_saves = float(gk_rows[saves_col].mean())
    absolute_gap = abs(mean_predicted_saves - empirical_saves_per_match)
    relative_gap = (
        absolute_gap / abs(empirical_saves_per_match) if empirical_saves_per_match else float("nan")
    )
    return GoalkeeperSavesReport(
        mean_predicted_saves=mean_predicted_saves,
        empirical_saves_per_match=empirical_saves_per_match,
        absolute_gap=absolute_gap,
        relative_gap=relative_gap,
        n_players=len(gk_rows),
    )


@dataclass(frozen=True)
class HorizonMonotonicityReport:
    """Mean ``p_60_plus`` per gameweek across a multi-gameweek horizon (ENGINE_AUDIT_FIXES T-J
    item 5).

    Carries only the raw per-gameweek means. There is no footballing reason projected playing time
    should systematically fall the further out a horizon looks, but deciding what counts as
    "decaying" versus ordinary week-to-week noise is a tolerance judgement left to the caller (see
    ``backtest.gate``'s own decay check), the same split as :class:`CalibrationReport` carrying MACE
    while the gate owns the acceptance threshold."""

    by_gameweek: pd.DataFrame  # columns: gameweek, mean_p_60_plus, n_players


def horizon_minutes_monotonicity(
    predictions: pd.DataFrame,
    gameweek_col: str = "gameweek",
    p_60_plus_col: str = "p_60_plus",
) -> HorizonMonotonicityReport:
    if predictions.empty:
        raise ValueError("predictions must not be empty")
    by_gameweek = (
        predictions.groupby(gameweek_col)[p_60_plus_col]
        .agg(mean_p_60_plus="mean", n_players="count")
        .reset_index()
        .sort_values(gameweek_col)
        .reset_index(drop=True)
    )
    return HorizonMonotonicityReport(by_gameweek=by_gameweek)
