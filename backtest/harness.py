"""Walk-forward validation harness — expanding window, refit every gameweek (3.1).

Never uses random train/test splits — that leaks the future. Fits on gameweeks 1..N-1, predicts
gameweek N using only data available before N's deadline, records the prediction, rolls forward,
repeats. Mirrors exactly how the engine is used in real life (BUILD_PLAN 3.1).

Refits every gameweek on an *expanding* window (train on everything from gameweek 1 to N-1, growing
every step) rather than periodically or on a sliding window — cheap enough for the engine's
interpretable regressions that there's no real trade-off, and the EWMA decay already built into the
rate-stat inputs (Phase 1) gives recency its due weight without a second, redundant recency
mechanism stacked on top of the first (BUILD_PLAN 3.1).

This harness is deliberately data-source-agnostic: callers supply a ``fit_fn``/``predict_fn`` pair,
so it can drive the real engine against real point-in-time snapshots, a baseline (backtest/
baselines.py), or a synthetic setup in tests — this module only enforces the one invariant that
matters (fitting never sees gameweek ``gw`` or later), not how a prediction is actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import pandas as pd

GAMEWEEK_COL = "gameweek"

__all__ = ["GAMEWEEK_COL", "FitFn", "PredictFn", "WalkForwardResult", "run_walk_forward"]


class FitFn(Protocol):
    def __call__(self, training_history: pd.DataFrame) -> object: ...


class PredictFn(Protocol):
    def __call__(self, fitted_state: object, gameweek: int) -> pd.DataFrame: ...


@dataclass(frozen=True)
class WalkForwardResult:
    """All out-of-sample predictions produced across the walk, concatenated with a ``gameweek``
    column — one row per (player, gameweek) — plus which gameweeks were skipped for not yet having
    enough training history to fit on."""

    predictions: pd.DataFrame
    skipped_gameweeks: tuple[int, ...] = field(default_factory=tuple)


def run_walk_forward(
    gameweeks: list[int],
    history: pd.DataFrame,
    fit_fn: FitFn,
    predict_fn: PredictFn,
    min_training_gameweeks: int = 1,
    gameweek_col: str = GAMEWEEK_COL,
) -> WalkForwardResult:
    """Walk forward across ``gameweeks`` in ascending order.

    For each gameweek ``gw``, fit on every row of ``history`` strictly before ``gw`` (expanding
    window), then predict ``gw`` — never the reverse. ``history`` is the ground truth (e.g.
    ``engine.data.storage.GameweekResult`` rows as a DataFrame) that the walk accumulates as it
    progresses; ``predict_fn`` is responsible for sourcing whatever pre-deadline, point-in-time
    inputs it needs for ``gw`` itself (the snapshot mechanism, Phase 1.2) — this harness only
    guarantees that fitting never sees gameweek ``gw`` or later.

    Gameweeks with fewer than ``min_training_gameweeks`` distinct prior gameweeks of history are
    skipped rather than fit on an unreasonably small window (recorded in
    :attr:`WalkForwardResult.skipped_gameweeks`, not silently dropped).
    """
    ordered = sorted(gameweeks)
    predictions: list[pd.DataFrame] = []
    skipped: list[int] = []

    for gw in ordered:
        training_history = history[history[gameweek_col] < gw]
        n_seen = training_history[gameweek_col].nunique()
        if n_seen < min_training_gameweeks:
            skipped.append(gw)
            continue

        fitted_state = fit_fn(training_history)
        gw_predictions = predict_fn(fitted_state, gw).copy()
        gw_predictions[gameweek_col] = gw
        predictions.append(gw_predictions)

    all_predictions = (
        pd.concat(predictions, ignore_index=True)
        if predictions
        else pd.DataFrame(columns=[gameweek_col])
    )
    return WalkForwardResult(predictions=all_predictions, skipped_gameweeks=tuple(skipped))
