"""Defensive contribution — per-90 action rates, opponent-possession adjusted (2.5).

The 2025/26 scoring addition, and bookmakers offer no market for it at all — most tools
underweight it, which is exactly why it's the engine's stated edge (BUILD_PLAN 2.5). Modelled
from each player's historical rate of qualifying defensive actions per 90 (tackles,
interceptions, blocks, clearances; plus recoveries for MID/FWD), converted into the probability
of clearing the position's threshold and banking a flat 2 points.

**Opponent-possession adjustment — the one fixture adjustment that points the opposite way from
goals/assists/clean sheets.** A player makes more defensive actions when their team *doesn't*
have the ball, so facing a possession-dominant opponent means *more* tackling/intercepting
opportunities, not fewer.

**Distributional form.** Defensive actions tend to cluster (a scrappy, backs-to-the-wall game
spikes hard), so a Negative Binomial — which allows variance to exceed the mean, unlike Poisson —
is the default here rather than assumed away. The overdispersion parameter is a placeholder
pending Phase 3's empirical check against real data (BUILD_PLAN 2.5), not asserted calibrated.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import nbinom

from engine.scoring import DEFENSIVE_CONTRIBUTION_POINTS, DEFENSIVE_CONTRIBUTION_THRESHOLD, GK

# Overdispersion parameter (alpha) for the Negative Binomial: variance = mu + alpha * mu^2.
# Placeholder pending Phase 3's empirical check against real defensive-action data (BUILD_PLAN
# 2.5's "caveat: limited history under the new rule").
DEFAULT_OVERDISPERSION = 0.15

# Neutral default when no possession data is supplied for a fixture -- an exactly average
# opponent, per BUILD_PLAN 2.5's "opponent possession share (or pass-volume)" framing.
LEAGUE_AVERAGE_POSSESSION_SHARE = 0.5


def opponent_possession_adjustment(
    opponent_possession_share: float,
    league_avg_possession_share: float = LEAGUE_AVERAGE_POSSESSION_SHARE,
) -> float:
    """Scale factor for a player's defensive-action rate: *more* actions against a
    possession-dominant opponent, the reverse direction from goals/assists/clean sheets
    (BUILD_PLAN 2.5)."""
    if league_avg_possession_share <= 0:
        raise ValueError("league_avg_possession_share must be positive")
    if not 0.0 <= opponent_possession_share <= 1.0:
        raise ValueError("opponent_possession_share must be in [0, 1]")
    return opponent_possession_share / league_avg_possession_share


def expected_defensive_action_rate(
    player_actions_per_90: float,
    opponent_possession_share: float,
    league_avg_possession_share: float = LEAGUE_AVERAGE_POSSESSION_SHARE,
    expected_minutes: float = 90.0,
) -> float:
    """Expected qualifying defensive actions this gameweek — the mean parameter (mu) for the
    Negative Binomial outcome distribution."""
    if player_actions_per_90 < 0:
        raise ValueError("player_actions_per_90 must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    adjustment = opponent_possession_adjustment(
        opponent_possession_share, league_avg_possession_share
    )
    return player_actions_per_90 * adjustment * (expected_minutes / 90.0)


def negative_binomial_params(
    mu: float, alpha: float = DEFAULT_OVERDISPERSION
) -> tuple[float, float]:
    """Convert a mean (``mu``) and overdispersion (``alpha``, where variance = mu + alpha*mu^2)
    into scipy's ``(n, p)`` Negative Binomial parametrization."""
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if alpha <= 0:
        raise ValueError("alpha must be positive (use a Poisson directly for the alpha=0 limit)")
    n = 1.0 / alpha
    p = n / (n + mu) if mu > 0 else 1.0
    return n, p


def probability_clears_threshold(
    mu: float, threshold: int, alpha: float = DEFAULT_OVERDISPERSION
) -> float:
    """P(defensive actions >= threshold) under a Negative Binomial(mu, alpha)."""
    if threshold <= 0:
        raise ValueError("threshold must be positive")
    if mu == 0:
        return 0.0
    n, p = negative_binomial_params(mu, alpha)
    return float(1.0 - nbinom.cdf(threshold - 1, n, p))


def fit_overdispersion(
    actual_actions: pd.Series, expected_mu: pd.Series, min_rows: int = 100
) -> float:
    """Method-of-moments Negative Binomial dispersion, refit every gameweek per position from real
    defensive-action counts (ENGINE_IMPROVEMENTS.md 1.2 — this was a hardcoded placeholder despite
    the regression layer existing). A distributional dispersion parameter, not a regression
    coefficient — there's no feature set being weighted, just a variance-vs-mean estimate, so this
    is a direct method-of-moments fit rather than a :class:`engine.regression.PerPositionRegression`
    call (the driver calls this once per position via a plain ``groupby``, not that class).

    Since ``variance = mu + alpha*mu^2`` under this parametrization,
    ``alpha = mean((y - mu)^2 - mu) / mean(mu^2)``. Falls back to :data:`DEFAULT_OVERDISPERSION`
    when the sample is too thin, or when the estimate is non-positive (an under-dispersed sample —
    the Negative Binomial requires ``alpha > 0``).
    """
    y = np.asarray(actual_actions, dtype=float)
    mu = np.asarray(expected_mu, dtype=float)
    if len(y) < min_rows:
        return DEFAULT_OVERDISPERSION
    denominator = np.mean(mu**2)
    if denominator <= 0:
        return DEFAULT_OVERDISPERSION
    alpha = float(np.mean((y - mu) ** 2 - mu) / denominator)
    return alpha if alpha > 0 else DEFAULT_OVERDISPERSION


@dataclass(frozen=True)
class DefensiveContributionProjection:
    """A player's defensive-contribution-component projection for one gameweek."""

    p_clears_threshold: float
    threshold: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_clears_threshold <= 1.0:
            raise ValueError("p_clears_threshold must be in [0, 1]")
        if self.threshold <= 0:
            raise ValueError("threshold must be positive")

    @property
    def expected_points(self) -> float:
        """Flat 2 points if the threshold is cleared — a single all-or-nothing bonus, not scaled
        by how far over the threshold a player gets (BUILD_PLAN scoring.py)."""
        return self.p_clears_threshold * DEFENSIVE_CONTRIBUTION_POINTS


def project_defensive_contribution(
    position: str,
    player_actions_per_90: float,
    opponent_possession_share: float,
    league_avg_possession_share: float = LEAGUE_AVERAGE_POSSESSION_SHARE,
    expected_minutes: float = 90.0,
    alpha: float = DEFAULT_OVERDISPERSION,
    *,
    p_1_to_59: float | None = None,
    minutes_given_1_to_59: float | None = None,
    p_60_plus: float | None = None,
    minutes_given_60_plus: float | None = None,
) -> DefensiveContributionProjection:
    """Top-level entry point: combine the opponent-possession-adjusted rate and the Negative
    Binomial threshold probability into one projection.

    By default (``expected_minutes`` only) this evaluates ``P(actions >= threshold)`` at a single
    point-estimate minutes value — correct for a linear quantity, but
    ``probability_clears_threshold`` is convex in its mean, so by Jensen's inequality evaluating it
    at ``E[minutes]`` *understates* the true expectation whenever minutes are actually uncertain
    (ENGINE_IMPROVEMENTS_2.md B.1): real walk-forward data showed this understating the
    DC-threshold probability by ~34% overall, worst for rotation-risk players (a "played 0" and a
    "played 80" blended into one midpoint minutes figure corresponds to no real match). Passing all
    four ``p_1_to_59``/``minutes_given_1_to_59``/
    ``p_60_plus``/``minutes_given_60_plus`` keyword arguments (typically straight from a
    :class:`engine.models.minutes.MinutesDistribution`) instead computes the properly-weighted
    expectation over the two non-zero minutes buckets, which is what :mod:`engine.pipeline` does.
    Omitting them reproduces the exact prior (point-estimate) behavior unchanged, so every existing
    standalone/backtest-in-isolation caller is unaffected.
    """
    if position == GK:
        raise ValueError("defensive contribution is not modelled for GK")
    if position not in DEFENSIVE_CONTRIBUTION_THRESHOLD:
        raise ValueError(f"unknown position: {position!r}")
    threshold = DEFENSIVE_CONTRIBUTION_THRESHOLD[position]

    bucket_args = (p_1_to_59, minutes_given_1_to_59, p_60_plus, minutes_given_60_plus)
    if any(arg is not None for arg in bucket_args):
        if any(arg is None for arg in bucket_args):
            raise ValueError(
                "p_1_to_59, minutes_given_1_to_59, p_60_plus, and minutes_given_60_plus must be "
                "given together or not at all"
            )
        mu_1_to_59 = expected_defensive_action_rate(
            player_actions_per_90,
            opponent_possession_share,
            league_avg_possession_share,
            minutes_given_1_to_59,
        )
        mu_60_plus = expected_defensive_action_rate(
            player_actions_per_90,
            opponent_possession_share,
            league_avg_possession_share,
            minutes_given_60_plus,
        )
        p_clears_threshold = p_1_to_59 * probability_clears_threshold(
            mu_1_to_59, threshold, alpha
        ) + p_60_plus * probability_clears_threshold(mu_60_plus, threshold, alpha)
    else:
        mu = expected_defensive_action_rate(
            player_actions_per_90,
            opponent_possession_share,
            league_avg_possession_share,
            expected_minutes,
        )
        p_clears_threshold = probability_clears_threshold(mu, threshold, alpha)

    return DefensiveContributionProjection(
        p_clears_threshold=p_clears_threshold, threshold=threshold
    )
