"""Shared helper for FPL's "every N units banks a point" discrete scoring rules.

Saves (1 point per 3), and goals conceded (-1 per 2) both convert a Poisson-ish count into points
via integer floor division — a non-linear function of the count, so the correct expectation is
``E[floor(X / divisor)]`` over the full outcome distribution, not ``floor(E[X] / divisor)``. Both
:mod:`engine.models.clean_sheets` and :mod:`engine.models.saves` need this, so it lives here once.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import poisson

DEFAULT_MAX_COUNT = 30


def expected_floor_division(mu: float, divisor: int, max_count: int = DEFAULT_MAX_COUNT) -> float:
    """``E[floor(X / divisor)]`` for ``X ~ Poisson(mu)``, truncated at ``max_count`` (negligible
    tail mass for any realistic football count) and renormalized."""
    if mu < 0:
        raise ValueError("mu must be non-negative")
    if divisor <= 0:
        raise ValueError("divisor must be positive")
    counts = np.arange(max_count + 1)
    pmf = poisson.pmf(counts, mu)
    pmf = pmf / pmf.sum()
    units = counts // divisor
    return float((pmf * units).sum())
