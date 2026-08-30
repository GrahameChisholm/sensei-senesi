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

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gamma


@dataclass(frozen=True)
class EwmaRateConfig:
    """``halflife_matches``: matches for the weight on a past observation to halve. Tuned by
    backtesting (BUILD_PLAN 1.1) — not asserted correct here, just given a reasonable default."""

    halflife_matches: float = 10.0


DEFAULT_CONFIG = EwmaRateConfig()


def _ewma(series: pd.Series, config: EwmaRateConfig) -> pd.Series:
    return series.astype(float).ewm(halflife=config.halflife_matches, adjust=True).mean()


def _winsorized_stat(
    matches: pd.DataFrame,
    stat_col: str,
    minutes_col: str,
    max_rate_per_90: float | None,
) -> pd.Series:
    """``stat_col``, clipped per-match to what ``max_rate_per_90`` implies for that match's own
    minutes, when a cap is given.

    Minutes-weighting the EWMA (see module docstring) still lets a single sub-appearance dominate
    when nearly *all* of a player's history is cameos, since nothing bounds how large ``stat_col``
    can be relative to the minutes it was recorded in. A real Understat pull showed non-penalty
    xG/90 rates up to 35.4, concentrated entirely in players with well under 90 minutes of total
    prior history — a fluke high-xG kick taken in 1-2 minutes on the pitch. Capping each match's own
    implied per-90 rate before it enters the EWMA fixes the input at the source rather than relying
    on shrinkage alone to pull an already-extreme rate back down.
    """
    stat = matches[stat_col].astype(float)
    if max_rate_per_90 is None:
        return stat
    if max_rate_per_90 <= 0:
        raise ValueError("max_rate_per_90 must be positive")
    cap = max_rate_per_90 * matches[minutes_col].astype(float) / 90.0
    return stat.clip(upper=cap)


def ewma_rate_asof(
    matches: pd.DataFrame,
    stat_col: str,
    minutes_col: str = "time",
    config: EwmaRateConfig = DEFAULT_CONFIG,
    max_rate_per_90: float | None = None,
) -> pd.Series:
    """Point-in-time per-90 EWMA rate of ``stat_col``, one value per row of ``matches``.

    ``matches`` must already be sorted chronologically (oldest first). Row *i*'s value reflects
    only rows strictly before *i* — the first row is always ``NaN`` (no prior history).

    ``max_rate_per_90``, when given, winsorizes each match's contribution first — see
    :func:`_winsorized_stat`.
    """
    stat = _winsorized_stat(matches, stat_col, minutes_col, max_rate_per_90)
    ewm_stat = _ewma(stat, config)
    ewm_minutes = _ewma(matches[minutes_col], config)
    rate = (ewm_stat / ewm_minutes.replace(0.0, np.nan)) * 90.0
    return rate.shift(1)


def latest_ewma_rate(
    matches: pd.DataFrame,
    stat_col: str,
    minutes_col: str = "time",
    config: EwmaRateConfig = DEFAULT_CONFIG,
    max_rate_per_90: float | None = None,
) -> float:
    """Current per-90 EWMA rate of ``stat_col`` using every row in ``matches`` — for projecting a
    not-yet-played gameweek. ``matches`` must be sorted chronologically (oldest first).

    ``max_rate_per_90``, when given, winsorizes each match's contribution first — see
    :func:`_winsorized_stat`.
    """
    if len(matches) == 0:
        return float("nan")
    stat = _winsorized_stat(matches, stat_col, minutes_col, max_rate_per_90)
    ewm_stat = _ewma(stat, config)
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


# --- Gamma-Poisson rate ratios (actual vs expected) ---------------------------------------------
#
# "Is this player outperforming their expected stats, and is it real?" is a count-against-exposure
# problem: an actual count ``y`` (goals + assists, clean sheets) against an expected count ``E``
# (xGI, summed clean-sheet probability). The naive answer, a raw difference of per-90 rates, is
# unusable as a ranked column: one goal in 25 minutes is a +3.3 per-90 gap and tops any sort.
# features/differentials.py's ``luck_gap`` is exactly that quantity, which is why it is only ever
# used there as an internal binary gate and never exposed as a sortable column.
#
# The standard treatment is a gamma-Poisson (negative binomial) rate ratio: ``y ~ Poisson(E*theta)``
# with ``theta ~ Gamma(k, k)``, prior mean 1 ("performing exactly as expected"). Conjugacy gives a
# closed-form posterior ``Gamma(k + y, k + E)``, so both the shrunk point estimate and its credible
# interval are analytic. Thin samples shrink hard toward 1.0 and carry wide intervals, which is what
# makes the result safe to sort on and what stops the top of a ~700-row table filling with cameos.
#
# ``k`` is *fitted*, not asserted (see :func:`fit_rate_ratio_prior`), and callers fit it separately
# per position: a defender's xGI exposure per match is roughly a quarter of a forward's, so a single
# shared prior would either under-shrink forwards or flatten every defender to exactly 1.00.

# Credible interval mass. 90% (5th to 95th percentile) rather than 95%, matching the "is this
# distinguishable from 1.0 yet" question this supports rather than a formal hypothesis test.
RATE_RATIO_INTERVAL = 0.90


@dataclass(frozen=True)
class RateRatio:
    """One player's actual-vs-expected count ratio under the gamma-Poisson posterior.

    ``ratio`` is the posterior mean ``(y + k) / (E + k)``, read directly as "scoring 1.42x their
    expected rate". ``low``/``high`` bound it at :data:`RATE_RATIO_INTERVAL`; an interval that
    still contains 1.0 means the deviation is not yet distinguishable from chance, which is the
    honest answer for most players over a short window and is the whole reason the interval is
    carried alongside the point estimate rather than the estimate being shown alone.
    """

    ratio: float
    low: float
    high: float
    exposure: float

    @property
    def is_hot(self) -> bool:
        """Overperforming beyond what chance explains — the entire interval sits above 1.0."""
        return self.low > 1.0

    @property
    def is_cold(self) -> bool:
        """Underperforming beyond what chance explains — the entire interval sits below 1.0."""
        return self.high < 1.0


def fit_rate_ratio_prior(actuals: Sequence[float], exposures: Sequence[float]) -> float:
    """Method-of-moments estimate of the gamma prior strength ``k`` from a whole population
    (one position's players), returning ``inf`` when the data show no heterogeneity to detect.

    Under the model, ``Var(y) = E + E^2/k``: the first term is irreducible Poisson noise, the
    second is genuine between-player spread in ``theta``. Pooling the squared residuals gives
    ``sum((y - E)^2 - E) = (1/k) * sum(E^2)``, hence ``k = sum(E^2) / sum((y - E)^2 - E)``.

    A non-positive denominator means the observed spread is at or below pure Poisson noise, i.e.
    there is no detectable finishing/clean-sheet skill in this sample at all. That returns ``inf``,
    which collapses every ratio in the position to exactly 1.0 — a real and reportable finding
    ("all of this variation is consistent with chance"), never an error to swallow. This is also
    the correct behaviour early in a season, when a handful of matches genuinely cannot separate
    skill from luck.
    """
    actual_array = np.asarray(actuals, dtype=float)
    exposure_array = np.asarray(exposures, dtype=float)
    if actual_array.shape != exposure_array.shape:
        raise ValueError("actuals and exposures must be the same length")

    valid = (exposure_array > 0) & np.isfinite(exposure_array) & np.isfinite(actual_array)
    if not valid.any():
        return float("inf")

    actual_array = actual_array[valid]
    exposure_array = exposure_array[valid]

    excess_variance = float(np.sum((actual_array - exposure_array) ** 2 - exposure_array))
    if excess_variance <= 0:
        return float("inf")
    return float(np.sum(exposure_array**2) / excess_variance)


def rate_ratio_posterior(actual: float, exposure: float, k: float) -> RateRatio | None:
    """One player's :class:`RateRatio` under prior strength ``k`` (from
    :func:`fit_rate_ratio_prior`).

    Returns ``None`` for non-positive exposure: a player with no expected involvements at all has
    no ratio to speak of, which is a real state ("not on the pitch enough to say"), not a zero.
    Infinite ``k`` (no detectable heterogeneity) degenerates to exactly 1.0 with a zero-width
    interval, matching what the fitted prior is actually asserting.
    """
    if exposure <= 0 or not np.isfinite(exposure) or not np.isfinite(actual):
        return None
    if not np.isfinite(k):
        return RateRatio(ratio=1.0, low=1.0, high=1.0, exposure=float(exposure))

    shape = k + actual
    rate = k + exposure
    tail = (1.0 - RATE_RATIO_INTERVAL) / 2.0
    low, high = gamma.ppf([tail, 1.0 - tail], a=shape, scale=1.0 / rate)
    return RateRatio(
        ratio=float(shape / rate),
        low=float(low),
        high=float(high),
        exposure=float(exposure),
    )
