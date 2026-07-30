"""Compares stats-engine projections vs market-implied probability; surfaces the gap (BUILD_PLAN
4b.2).

Never used in backtesting or inside the engine — a live, decision-time-only comparison. Removes
the bookmaker's margin (the "overround") from raw decimal odds before comparing, since raw
implied probabilities across a market's full outcome set always sum to slightly over 1.0 (the
bookmaker's built-in edge); comparing an unadjusted implied probability against the engine's own
(properly normalized) probability would flag divergence that's just this constant vig, not a
genuine signal.

**Surface the gap, don't silently resolve it (BUILD_PLAN 4b.2).** This module never adjusts the
engine's own number — it only classifies how far apart the two views are. Whether a divergence
should ever feed back into the projection is a separate, later, evidence-based question (BUILD_PLAN
4b.3), deferred until enough live gameweeks exist to measure it honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "SMALL_DIVERGENCE_THRESHOLD",
    "LARGE_DIVERGENCE_THRESHOLD",
    "DivergenceFlag",
    "implied_probability",
    "remove_overround",
    "compare_probabilities",
]

# Divergence bands (absolute probability-point gap between the engine's projection and the
# margin-removed market-implied probability). Not fitted -- BUILD_PLAN 4b.3 defers "should this
# ever adjust the number" to a later, evidence-based comparison once enough live gameweeks
# accumulate; these thresholds only gate the "worth a human's attention" flag itself.
SMALL_DIVERGENCE_THRESHOLD = 0.05
LARGE_DIVERGENCE_THRESHOLD = 0.15


def implied_probability(decimal_odds: float) -> float:
    """Raw (margin-inclusive) implied probability from decimal odds — ``1 / odds``. Summed
    across a market's full outcome set this normally exceeds 1.0 by the bookmaker's overround;
    use :func:`remove_overround` for a properly normalized set of probabilities."""
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be > 1.0")
    return 1.0 / decimal_odds


def remove_overround(decimal_odds: list[float]) -> list[float]:
    """Margin-removed implied probabilities for one full, mutually exclusive outcome set (e.g.
    home/draw/away) — each raw ``1/odds`` scaled down proportionally so the set sums to exactly
    1.0, the standard way of stripping a bookmaker's overround (BUILD_PLAN 4b.2)."""
    if not decimal_odds:
        raise ValueError("decimal_odds must not be empty")
    raw = [implied_probability(odds) for odds in decimal_odds]
    total = sum(raw)
    return [p / total for p in raw]


@dataclass(frozen=True)
class DivergenceFlag:
    """One outcome's engine-vs-market comparison for one player/fixture."""

    label: str  # human-readable, e.g. "Haaland anytime scorer"
    engine_probability: float
    market_probability: float  # margin-removed
    gap: float  # engine_probability - market_probability; positive = engine more bullish
    severity: str  # "none" | "small" | "large"


def compare_probabilities(
    label: str, engine_probability: float, market_probability: float
) -> DivergenceFlag:
    """Compare one engine-projected probability against the market's margin-removed equivalent
    for the *same* outcome, and classify the gap. Never adjusts ``engine_probability`` — only
    reports (BUILD_PLAN 4b.2: "surface the gap, don't silently resolve it")."""
    if not 0.0 <= engine_probability <= 1.0:
        raise ValueError("engine_probability must be in [0, 1]")
    if not 0.0 <= market_probability <= 1.0:
        raise ValueError("market_probability must be in [0, 1]")

    gap = engine_probability - market_probability
    abs_gap = abs(gap)
    if abs_gap >= LARGE_DIVERGENCE_THRESHOLD:
        severity = "large"
    elif abs_gap >= SMALL_DIVERGENCE_THRESHOLD:
        severity = "small"
    else:
        severity = "none"

    return DivergenceFlag(
        label=label,
        engine_probability=engine_probability,
        market_probability=market_probability,
        gap=gap,
        severity=severity,
    )
