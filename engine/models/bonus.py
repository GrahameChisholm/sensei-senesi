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
    *[f"position_{position}" for position in POSITIONS],
]


def build_features(
    expected_goals: float,
    expected_assists: float,
    clean_sheet_probability: float,
    defensive_action_rate: float,
    position: str,
) -> dict[str, float]:
    """Assemble one row of the bonus regression's own BPS-relevant inputs (BUILD_PLAN 2.6),
    one-hot encoding position since bonus competition differs structurally by role."""
    if position not in POSITIONS:
        raise ValueError(f"unknown position: {position!r}")
    row = {
        "expected_goals": expected_goals,
        "expected_assists": expected_assists,
        "clean_sheet_probability": clean_sheet_probability,
        "defensive_action_rate": defensive_action_rate,
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
