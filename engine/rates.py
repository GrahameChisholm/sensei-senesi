"""Shared per-90 rate-stat utility — EWMA over match history (BUILD_PLAN 1.1).

Every stats-led component in Phase 2 (goals 2.2, assists 2.3, clean sheets 2.4, defensive
contribution 2.5) references a player's or team's "current" per-90 rate for some underlying stat
(non-penalty xG, xA, xGA, defensive actions, ...). Rather than each model reimplementing its own
windowing logic, they all go through this one utility.

**Why EWMA over the numerator and denominator separately, not EWMA of the per-match rate.** A
per-match rate (stat / minutes_that_match * 90) is wildly noisy for cameo appearances — five
minutes and a fluke goal is a per-90 rate of 18. Smoothing that noisy per-match ratio with EWMA
still lets one substitute appearance swing the average. Instead this EWMA-decays the numerator
and denominator (minutes) independently and only divides at the end, which naturally
minutes-weights every match's contribution — exactly the standard sabermetrics-style treatment
for rate stats.

**Point-in-time correctness.** :func:`ewma_rate_asof` returns, for every row, the rate as it stood
*before* that match kicked off (a `shift(1)`) — this is what a walk-forward backtest replays
against, so a match's own outcome is never included in its own rate for leakage purposes.
:func:`latest_ewma_rate` instead uses every row available (rate *after* the last known match) —
this is what a live, not-yet-played projection uses.

**Cold start.** Callers should pass a single chronologically-sorted history spanning prior
seasons into the current one — the EWMA decay parameter (``halflife_matches``) then lets recent
matches dominate while older ones fade smoothly, with no separate hand-coded blend rule needed for
the first few gameweeks of a new season (BUILD_PLAN 1.1).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EwmaRateConfig:
    """``halflife_matches``: matches for the weight on a past observation to halve. Tuned by
    backtesting (BUILD_PLAN 1.1) — not asserted correct here, just given a reasonable default."""

    halflife_matches: float = 10.0


DEFAULT_CONFIG = EwmaRateConfig()


def _ewma(series: pd.Series, config: EwmaRateConfig) -> pd.Series:
    return series.astype(float).ewm(halflife=config.halflife_matches, adjust=True).mean()


def ewma_rate_asof(
    matches: pd.DataFrame,
    stat_col: str,
    minutes_col: str = "time",
    config: EwmaRateConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Point-in-time per-90 EWMA rate of ``stat_col``, one value per row of ``matches``.

    ``matches`` must already be sorted chronologically (oldest first). Row *i*'s value reflects
    only rows strictly before *i* — the first row is always ``NaN`` (no prior history).
    """
    ewm_stat = _ewma(matches[stat_col], config)
    ewm_minutes = _ewma(matches[minutes_col], config)
    rate = (ewm_stat / ewm_minutes.replace(0.0, np.nan)) * 90.0
    return rate.shift(1)


def latest_ewma_rate(
    matches: pd.DataFrame,
    stat_col: str,
    minutes_col: str = "time",
    config: EwmaRateConfig = DEFAULT_CONFIG,
) -> float:
    """Current per-90 EWMA rate of ``stat_col`` using every row in ``matches`` — for projecting a
    not-yet-played gameweek. ``matches`` must be sorted chronologically (oldest first)."""
    if len(matches) == 0:
        return float("nan")
    ewm_stat = _ewma(matches[stat_col], config)
    ewm_minutes = _ewma(matches[minutes_col], config)
    minutes_total = ewm_minutes.iloc[-1]
    if not minutes_total:
        return float("nan")
    return float(ewm_stat.iloc[-1] / minutes_total * 90.0)


def _ewm_sum(values: pd.Series, config: EwmaRateConfig) -> float:
    """Exponentially-decayed *sum* (not average) — grows as matches accumulate, saturating around
    ``per_match_value / (1 - decay)`` for a long, roughly-constant history. Unlike
    ``.ewm().mean()`` (a normalized weighted average, which does not grow with sample size), this
    is what a genuine "how much evidence do we have" quantity needs to do.

    A plain recurrence (not a pandas built-in) since a normalized EWM can't express this — a
    naive power-series formulation would also risk numerical underflow over a decade-plus of
    match history.
    """
    if len(values) == 0:
        return 0.0
    decay = 0.5 ** (1.0 / config.halflife_matches)
    running = 0.0
    for value in values.astype(float):
        running = value + decay * running
    return running


def effective_sample_minutes(
    matches: pd.DataFrame,
    minutes_col: str = "time",
    config: EwmaRateConfig = DEFAULT_CONFIG,
) -> float:
    """Exponentially-decayed total minutes played — a proxy for "how much do we trust this
    player's own rate" used by shrinkage (see :func:`shrink_toward_prior`). Grows with more
    matches (unlike a plain EWM average, which saturates at the per-match minutes level
    regardless of history length) and low-minutes players (new signings, just back from injury)
    get a low value here and lean more on the prior."""
    return _ewm_sum(matches[minutes_col], config) if len(matches) else 0.0


def effective_sample_minutes_asof(
    matches: pd.DataFrame,
    minutes_col: str = "time",
    config: EwmaRateConfig = DEFAULT_CONFIG,
) -> pd.Series:
    """Point-in-time (``shift(1)``-equivalent) version of :func:`effective_sample_minutes` — one
    value per row of ``matches``, reflecting only strictly-prior rows, mirroring how
    :func:`ewma_rate_asof` relates to the scalar :func:`latest_ewma_rate`. ``matches`` must already
    be sorted chronologically (oldest first).

    Used by the card/own-goal rate shrinkage (ENGINE_IMPROVEMENTS_3.md A.3) as the point-in-time
    evidence weight behind each row's own card rate, the same role
    :func:`~engine.rates.effective_sample_minutes` already plays for the goals/assists shrinkage
    (ENGINE_IMPROVEMENTS_2.md B.2) — but computed incrementally here since that shrinkage needs a
    value *at every gameweek*, not just the latest one.
    """
    decay = 0.5 ** (1.0 / config.halflife_matches)
    running = 0.0
    values: list[float] = []
    for value in matches[minutes_col].astype(float):
        values.append(running)
        running = value + decay * running
    return pd.Series(values, index=matches.index, dtype=float)


def shrink_toward_prior(
    individual_rate: float,
    individual_weight: float,
    prior_rate: float,
    shrinkage_k: float,
) -> float:
    """Empirical-Bayes-style shrinkage of a thin-sample individual rate toward a more stable prior
    (e.g. team-level rate) — used by the assists model (2.3) for low-sample players.

    ``individual_weight`` is a sample-size-like quantity (e.g. :func:`effective_sample_minutes`,
    or a match count) and ``shrinkage_k`` is the weight given to the prior, in the same units —
    a bigger ``shrinkage_k`` means more shrinkage. Returns ``prior_rate`` outright when
    ``individual_weight`` is zero, NaN, or negative, or ``individual_rate`` is NaN (no individual
    signal at all) — a NaN weight must fall back to the prior rather than silently NaN-poisoning
    the blended result (ENGINE_IMPROVEMENTS_2.md C.2's "fail loudly or degrade safely, never
    propagate silently" principle applied to this shared shrinkage helper).
    """
    if shrinkage_k < 0:
        raise ValueError("shrinkage_k must be non-negative")
    if np.isnan(individual_weight) or individual_weight <= 0 or np.isnan(individual_rate):
        return prior_rate
    return (individual_weight * individual_rate + shrinkage_k * prior_rate) / (
        individual_weight + shrinkage_k
    )


def league_average_rate(team_rates: dict[str, float]) -> float:
    """Plain mean of every team's rate — the ``league_avg_xGA90`` denominator the opponent
    adjustment in goals (2.2), assists (2.3), and clean sheets (2.4) all divide by. A single
    shared helper so all three components normalize against the same number."""
    if not team_rates:
        raise ValueError("team_rates must not be empty")
    return float(np.mean(list(team_rates.values())))
