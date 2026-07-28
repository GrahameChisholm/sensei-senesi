"""Baselines the engine must beat: template captain, naive form, pure xG (3.3).

An accuracy number is meaningless without something to beat. Each baseline below produces a
predictions frame in the same shape the engine's own predictions take (``player_id``, ``position``,
``gameweek``, ``expected_points``), so it plugs directly into ``backtest.metrics``'s accuracy and
captaincy functions for a like-for-like comparison.

"Beats the baselines" is a statistical test, not a point-estimate comparison (BUILD_PLAN 3.3):
football's match-to-match variance is large enough that a small genuine edge and a small lucky
edge can look identical over a modest sample. :func:`paired_bootstrap_test` and
:func:`permutation_test_hit_rate` are the actual mechanism for that — the engine only counts as
having beaten a baseline once the relevant confidence interval excludes zero in its favour.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.scoring import ASSIST_POINTS, GOAL_POINTS

PLAYER_ID_COL = "player_id"
POSITION_COL = "position"
GAMEWEEK_COL = "gameweek"
EXPECTED_POINTS_COL = "expected_points"

__all__ = [
    "template_captain_predictions",
    "naive_form_predictions",
    "pure_xg_predictions",
    "training_median",
    "constant_predictions",
    "PairedBootstrapResult",
    "paired_bootstrap_test",
    "PermutationTestResult",
    "permutation_test_hit_rate",
]


def template_captain_predictions(
    ownership: pd.DataFrame, ownership_col: str = "selected_by_percent"
) -> pd.DataFrame:
    """BUILD_PLAN 3.3 — "captaining the highest-owned/highest-price player each week." ``ownership``
    has one row per (player_id, gameweek, position, ``ownership_col``); this baseline's
    ``expected_points`` is simply the ownership metric itself, so the highest-owned/highest-priced
    player always ranks first when compared through ``backtest.metrics.captaincy_hit_rate``. This
    baseline is only ever meaningful as a captaincy pick — it is not a points-accuracy (MAE)
    baseline, since the ownership metric has no points scale.
    """
    out = ownership.copy()
    out[EXPECTED_POINTS_COL] = out[ownership_col]
    return out[[PLAYER_ID_COL, POSITION_COL, GAMEWEEK_COL, EXPECTED_POINTS_COL]]


def naive_form_predictions(
    history: pd.DataFrame, n_recent: int = 4, actual_col: str = "total_points"
) -> pd.DataFrame:
    """BUILD_PLAN 3.3 — "projecting last-few-weeks' points forward." For each player, an unweighted
    rolling mean of their last ``n_recent`` gameweeks' actual points, shifted forward one gameweek
    so gameweek N's projection only ever uses gameweeks strictly before N (no leakage, per the
    guiding principle) — the same discipline the walk-forward harness enforces for the engine
    itself.
    """
    df = history.sort_values([PLAYER_ID_COL, GAMEWEEK_COL]).copy()
    df[EXPECTED_POINTS_COL] = df.groupby(PLAYER_ID_COL)[actual_col].transform(
        lambda s: s.rolling(n_recent, min_periods=1).mean().shift(1)
    )
    return df.dropna(subset=[EXPECTED_POINTS_COL])[
        [PLAYER_ID_COL, POSITION_COL, GAMEWEEK_COL, EXPECTED_POINTS_COL]
    ]


def pure_xg_predictions(
    player_rates: pd.DataFrame,
    npxg_per_90_col: str = "npxg_per_90",
    xa_per_90_col: str = "xa_per_90",
    recent_minutes_col: str = "recent_minutes_ewma",
) -> pd.DataFrame:
    """BUILD_PLAN 3.3 — "a version using only raw xG/xA per 90 with no minutes-model gating or
    regression weighting, to prove the fuller engine earns its extra complexity over the simplest
    possible stats approach." No opponent adjustment and no minutes-model bucket/conditional
    structure (2.1) — just each player's own rate stats, position-converted, scaled by a trailing
    minutes figure.

    ``recent_minutes_col`` is deliberately a point-in-time *trailing* minutes figure (e.g. the same
    per-player EWMA already computed in Phase 1), never this gameweek's realised minutes — using
    the actual outcome to scale the very prediction being scored against it would leak information
    that isn't available before kickoff (the "no leakage, ever" guiding principle applies to
    baselines too, or the comparison in 3.3 isn't honest).
    """
    df = player_rates.copy()
    goal_points = df[POSITION_COL].map(GOAL_POINTS)
    minutes_factor = df[recent_minutes_col] / 90.0
    df[EXPECTED_POINTS_COL] = (
        df[npxg_per_90_col] * goal_points + df[xa_per_90_col] * ASSIST_POINTS
    ) * minutes_factor
    return df[[PLAYER_ID_COL, POSITION_COL, GAMEWEEK_COL, EXPECTED_POINTS_COL]]


def training_median(history: pd.DataFrame, actual_col: str = "total_points") -> float:
    """The training-sample median actual points (ENGINE_IMPROVEMENTS.md 2.2: "the current bar is
    too low" -- the naive-form baseline turned out to score *worse* than a flat median predictor,
    so a constant baseline is a genuine floor, not a strawman). Point-in-time-safe by construction
    as long as the caller passes only ``training_history`` (never the gameweek being predicted) —
    the same discipline :func:`naive_form_predictions` applies via its forward shift.
    """
    return float(history[actual_col].median())


def constant_predictions(players: pd.DataFrame, value: float) -> pd.DataFrame:
    """BUILD_PLAN 3.3 / ENGINE_IMPROVEMENTS.md 2.2 — "predict the median every time." Assigns the
    same flat ``value`` to every row in ``players`` regardless of any player-specific signal.
    ``players`` needs only ``player_id``, ``position``, ``gameweek`` — no outcome column at all, so
    this baseline can never leak (there is nothing to leak from). Derive ``value`` with
    :func:`training_median` (or any other point-in-time-safe statistic) before calling this.
    """
    out = players[[PLAYER_ID_COL, POSITION_COL, GAMEWEEK_COL]].copy()
    out[EXPECTED_POINTS_COL] = float(value)
    return out


@dataclass(frozen=True)
class PairedBootstrapResult:
    """BUILD_PLAN 3.3 — "a paired bootstrap or paired t-test on the MAE difference." Bootstraps the
    mean of the paired (engine absolute error − baseline absolute error) difference; negative means
    the engine has lower error. ``beats_baseline`` is True only when the whole confidence interval
    sits below zero, i.e. the interval excludes zero in the engine's favour."""

    mean_diff: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    beats_baseline: bool


def paired_bootstrap_test(
    engine_absolute_errors: np.ndarray,
    baseline_absolute_errors: np.ndarray,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
    block_by: np.ndarray | None = None,
) -> PairedBootstrapResult:
    """``block_by``, if given, is a per-row block key (e.g. fixture id or gameweek) the same
    length as the error arrays — resampling then draws whole *blocks* with replacement instead of
    individual rows (ENGINE_IMPROVEMENTS.md "Not a correction" / ENGINE_IMPROVEMENTS_2.md D.2:
    players in the same match share shocks — a red card, a 5-0 drubbing, a rested squad — so an
    i.i.d.-by-row bootstrap understates uncertainty; a real 2025/26 check found the conclusion held
    but the interval widened as expected once blocked by fixture or gameweek). Omitting it
    reproduces the exact prior i.i.d.-by-row behavior.
    """
    engine_errors = np.asarray(engine_absolute_errors, dtype=float)
    baseline_errors = np.asarray(baseline_absolute_errors, dtype=float)
    if engine_errors.shape != baseline_errors.shape:
        raise ValueError("engine and baseline error arrays must be paired (same shape)")
    if engine_errors.size == 0:
        raise ValueError("no paired errors to test")

    diffs = engine_errors - baseline_errors
    rng = np.random.default_rng(seed)
    n = diffs.shape[0]

    if block_by is None:
        resample_indices = rng.integers(0, n, size=(n_bootstrap, n))
        bootstrap_means = diffs[resample_indices].mean(axis=1)
    else:
        block_keys = np.asarray(block_by)
        if block_keys.shape[0] != n:
            raise ValueError("block_by must be the same length as the paired error arrays")
        _, block_index = np.unique(block_keys, return_inverse=True)
        n_blocks = int(block_index.max()) + 1
        block_sums = np.zeros(n_blocks)
        block_counts = np.zeros(n_blocks)
        np.add.at(block_sums, block_index, diffs)
        np.add.at(block_counts, block_index, 1)
        chosen_blocks = rng.integers(0, n_blocks, size=(n_bootstrap, n_blocks))
        bootstrap_means = block_sums[chosen_blocks].sum(axis=1) / block_counts[chosen_blocks].sum(
            axis=1
        )

    alpha = 1 - confidence
    ci_low, ci_high = np.quantile(bootstrap_means, [alpha / 2, 1 - alpha / 2])
    return PairedBootstrapResult(
        mean_diff=float(diffs.mean()),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        n_bootstrap=n_bootstrap,
        beats_baseline=bool(ci_high < 0),
    )


@dataclass(frozen=True)
class PermutationTestResult:
    """BUILD_PLAN 3.3 — "a binomial/permutation test on the captaincy hit-rate difference."
    ``observed_diff`` is engine hit-rate minus baseline hit-rate; ``beats_baseline`` requires both
    a positive observed difference and a p-value below ``alpha``."""

    observed_diff: float
    p_value: float
    n_permutations: int
    beats_baseline: bool


def permutation_test_hit_rate(
    engine_hits: np.ndarray,
    baseline_hits: np.ndarray,
    n_permutations: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> PermutationTestResult:
    """Paired permutation test on the per-gameweek engine-vs-baseline captain hit/miss difference:
    for each gameweek, randomly decide whether to swap which series that gameweek's hit/miss is
    attributed to, and see how often a difference at least as extreme as the one actually observed
    arises purely by chance."""
    engine = np.asarray(engine_hits, dtype=float)
    baseline = np.asarray(baseline_hits, dtype=float)
    if engine.shape != baseline.shape:
        raise ValueError("engine and baseline hit arrays must be paired (same shape)")
    if engine.size == 0:
        raise ValueError("no paired hits to test")

    n = engine.shape[0]
    observed_diff = float(engine.mean() - baseline.mean())

    rng = np.random.default_rng(seed)
    swap = rng.integers(0, 2, size=(n_permutations, n)).astype(bool)
    engine_permuted = np.where(swap, baseline, engine)
    baseline_permuted = np.where(swap, engine, baseline)
    permuted_diffs = engine_permuted.mean(axis=1) - baseline_permuted.mean(axis=1)

    p_value = float(np.mean(np.abs(permuted_diffs) >= abs(observed_diff)))
    return PermutationTestResult(
        observed_diff=observed_diff,
        p_value=p_value,
        n_permutations=n_permutations,
        beats_baseline=bool(observed_diff > 0 and p_value < alpha),
    )
