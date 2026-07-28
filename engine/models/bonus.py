"""Bonus model — regression proxy against own expected BPS-relevant stats (2.6).

Bonus is structurally different from every other component: FPL awards 3/2/1 to the top three BPS
scorers *within that specific match* — a relative ranking across ~22 players, not an absolute
per-player threshold. Full accuracy would mean jointly simulating every player's BPS-contributing
events for both full lineups and ranking them each run. Given bonus is explicitly one of the
smaller components (BUILD_PLAN 2.6), this instead uses a **regression proxy**: regress a player's
actual bonus received directly against their own expected BPS-relevant stats (own goals, assists,
clean sheet probability, defensive-action rate, position), letting the regression implicitly
absorb "how much competition for bonus typically exists" without explicitly modelling the other 21
players. Revisit with full joint simulation only if Phase 3 backtesting shows bonus calibration is
a material error source.

**The training target must be bonus recomputed under the current (2026/27) BPS formula from raw
match-event data, not any pre-2026/27 "actual bonus received" column** (BUILD_PLAN 2.6) — the BPS
rework means that historical column reflects a superseded formula. Recomputing it is a data-prep
concern upstream of this module; this module just fits/predicts given whatever target column it's
handed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from engine.scoring import POSITIONS

MIN_BONUS = 0.0
MAX_BONUS = 3.0

FEATURE_COLUMNS = [
    "expected_goals",
    "expected_assists",
    "clean_sheet_probability",
    "defensive_action_rate",
    "expected_minutes",
    *[f"position_{position}" for position in POSITIONS],
]


def plackett_luce_rank_probabilities(strengths: np.ndarray, max_rank: int = 3) -> np.ndarray:
    """Exact Plackett-Luce marginal probability that each of ``n`` items in one single contest
    (here: one real fixture's ~22 on-pitch players) finishes in each of the top ``max_rank``
    positions (ENGINE_IMPROVEMENTS_3.md D.2 — "the right shape is a soft P(top-3 in this fixture)
    converted to an expected 3/2/1, not a hard assignment").

    ``strengths`` are non-negative Plackett-Luce "worths" (here, each player's own expected BPS
    this fixture) — the probability an item is ranked first is ``strength / sum(strengths)``;
    second is the sum, over who could plausibly have finished first, of that player's own P(first)
    times this item's conditional share of the remaining strength; and so on. This is the
    generative model BUILD_PLAN 2.6 itself describes bonus as ("the top three BPS scorers... a
    relative ranking across ~22 players"), made explicit rather than left entirely to a linear
    regression proxy that never sees the other players in the same match.

    Returns an ``(n, max_rank)`` array whose ``[i, r]`` entry is P(item ``i`` finishes exactly rank
    ``r + 1``). Exact (not simulated) — tractable at fixture scale (n ~20-30) since it only
    requires enumerating ordered pairs/triples of "who took the earlier ranks", not full
    permutations.
    """
    w = np.asarray(strengths, dtype=float)
    n = len(w)
    if n == 0:
        return np.zeros((0, max_rank))
    if np.any(w < 0):
        raise ValueError("strengths must be non-negative")
    if max_rank < 1:
        raise ValueError("max_rank must be at least 1")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("strengths must sum to a positive value")

    probs = np.zeros((n, max_rank))
    p_first = w / total
    probs[:, 0] = p_first
    if max_rank == 1 or n == 1:
        return probs

    # Rank 2: sum over who took rank 1 (j != i) of P(j first) * P(i first | j removed).
    remaining_after_one = total - w  # remaining_after_one[j] = total strength excluding item j
    with np.errstate(divide="ignore", invalid="ignore"):
        # cond_first_removed[i, j] = P(i first | j removed) -- i's share of the strength left once
        # j is taken out of contention. The diagonal (i == j) is meaningless and zeroed below.
        cond_first_removed = w[:, None] / remaining_after_one[None, :]
    np.fill_diagonal(cond_first_removed, 0.0)
    probs[:, 1] = cond_first_removed @ p_first
    if max_rank == 2 or n == 2:
        return probs

    # Rank 3: sum over ordered (j, k), j != k, of P(j first) * P(k second | j first) *
    # P(i first | j, k removed), restricted to i not in {j, k} (a player can't take a later rank
    # in a scenario where they already took an earlier one). n is fixture-sized (~20-30), so an
    # explicit triple-nested-but-i-vectorized loop over (j, k) pairs is cheap (~n^2 iterations).
    rank3 = np.zeros(n)
    for j in range(n):
        for k in range(n):
            if k == j:
                continue
            p_jk = p_first[j] * cond_first_removed[k, j]  # P(j 1st) * P(k 2nd | j 1st)
            if p_jk <= 0:
                continue
            remaining = total - w[j] - w[k]
            if remaining <= 0:
                continue
            contribution = p_jk * (w / remaining)
            contribution[[j, k]] = 0.0  # i can't be the same player as j or k
            rank3 += contribution
    probs[:, 2] = rank3
    return probs


def expected_bonus_from_fixture_strengths(
    strengths: np.ndarray, bonus_points: tuple[float, float, float] = (3.0, 2.0, 1.0)
) -> np.ndarray:
    """Expected bonus per player in one fixture: ``sum_r P(rank r) * bonus_points[r]``
    (ENGINE_IMPROVEMENTS_3.md D.2), from :func:`plackett_luce_rank_probabilities`."""
    probs = plackett_luce_rank_probabilities(strengths, max_rank=len(bonus_points))
    return probs @ np.asarray(bonus_points, dtype=float)


def build_features(
    expected_goals: float,
    expected_assists: float,
    clean_sheet_probability: float,
    defensive_action_rate: float,
    position: str,
    expected_minutes: float,
) -> dict[str, float]:
    """Assemble one row of the bonus regression's own BPS-relevant inputs (BUILD_PLAN 2.6),
    one-hot encoding position since bonus competition differs structurally by role.

    ``expected_minutes`` (ENGINE_IMPROVEMENTS_3.md A.2) is required, not optional: without it,
    nothing in this feature set depends on whether the player is expected to appear at all —
    ``clean_sheet_probability`` and ``defensive_action_rate`` are team/rate-level and
    ``expected_goals``/``expected_assists`` are the *only* other minutes-scaled inputs, so a
    fitted intercept alone was assigning non-trivial bonus to players the minutes model was
    confident would not play. The caller must pass the same modelled (not realised) expected
    minutes on both the fit and predict paths — see ``backtest/run_season.py:fit_fn``'s own note
    on the train/serve skew this closes.
    """
    if position not in POSITIONS:
        raise ValueError(f"unknown position: {position!r}")
    row = {
        "expected_goals": expected_goals,
        "expected_assists": expected_assists,
        "clean_sheet_probability": clean_sheet_probability,
        "defensive_action_rate": defensive_action_rate,
        "expected_minutes": expected_minutes,
    }
    row.update({f"position_{p}": (1.0 if p == position else 0.0) for p in POSITIONS})
    return row


@dataclass(frozen=True)
class BonusProjection:
    expected_bonus: float

    def __post_init__(self) -> None:
        if not MIN_BONUS <= self.expected_bonus <= MAX_BONUS:
            raise ValueError(f"expected_bonus must be in [{MIN_BONUS}, {MAX_BONUS}]")

    @property
    def expected_points(self) -> float:
        """Bonus points *are* the stat — no separate conversion table, unlike every other
        component."""
        return self.expected_bonus


@dataclass
class BonusModel:
    """Interpretable linear regression proxy (BUILD_PLAN "explainable over clever") mapping a
    player's own BPS-relevant inputs to expected bonus received."""

    regression: LinearRegression = field(default_factory=LinearRegression)
    _fitted: bool = field(default=False, init=False, repr=False)

    def fit(self, features: pd.DataFrame, actual_bonus: pd.Series) -> BonusModel:
        X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
        y = actual_bonus.to_numpy(dtype=float)
        self.regression.fit(X, y)
        self._fitted = True
        return self

    def predict(self, features: pd.DataFrame) -> list[BonusProjection]:
        if not self._fitted:
            raise RuntimeError("BonusModel.predict called before fit")
        X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
        predicted = np.clip(self.regression.predict(X), MIN_BONUS, MAX_BONUS)
        return [BonusProjection(expected_bonus=float(value)) for value in predicted]
