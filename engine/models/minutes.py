"""Minutes model — the foundation. Bucket probabilities + conditional expected minutes (2.1).

Two linked outputs per BUILD_PLAN 2.1:
1. Bucket probabilities: P(0 minutes), P(1-59), P(60+).
2. Conditional expected minutes within each non-zero bucket.

Fit as a **two-stage** model, mirroring the real sequence of decisions (team-sheet, then in-game
substitution): first start/no-start, then, conditional on starting, a withdrawal-timing split
(subbed before 60' vs playing 60+). A third, smaller path covers players who did *not* start but
came on as a substitute.

Every other component downstream (goals, assists, clean sheets, defensive contribution) gates and
scales by this model's output — get this wrong and nothing else can compensate.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression

MINUTES_60_PLUS_THRESHOLD = 60

# Per BUILD_PLAN 2.1 "Inputs" table. Position and squad depth are deliberately excluded — see
# "Considered and cut" in the plan (structural redundancy / no clean data source respectively).
FEATURE_COLUMNS = [
    "recent_start_rate",
    "recent_minutes_ewma",
    "fixture_congestion",
    "chance_of_playing_next_round",
    "status_score",
]

# FPL's `status` code -> a coarse numeric availability proxy. `news` free text is deliberately
# NOT parsed into a feature (BUILD_PLAN 2.1: display flag only) — chance_of_playing_next_round
# and status already carry the same information as a clean structured number.
STATUS_SCORE = {"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0}

# Fallback conditional-minutes estimates for a bucket with no fitted training examples (e.g. a
# thin synthetic sample, or a position that never has sub-appearance rows in-sample). Rough
# league-average priors, not asserted as calibrated — real calibration is Phase 3's job.
DEFAULT_MINUTES_60_PLUS = 80.0
DEFAULT_MINUTES_STARTED_UNDER_60 = 45.0
DEFAULT_MINUTES_SUB_APPEARANCE = 20.0


def encode_status(status: str) -> float:
    """FPL `status` (a/d/i/s/u) -> numeric availability proxy for :data:`FEATURE_COLUMNS`."""
    try:
        return STATUS_SCORE[status]
    except KeyError:
        raise ValueError(f"unknown FPL status code: {status!r}") from None


@dataclass(frozen=True)
class MinutesDistribution:
    """The model's full output for one player in one gameweek."""

    p_zero: float
    p_1_to_59: float
    p_60_plus: float
    expected_minutes_given_1_to_59: float
    expected_minutes_given_60_plus: float

    def __post_init__(self) -> None:
        total = self.p_zero + self.p_1_to_59 + self.p_60_plus
        if not np.isclose(total, 1.0, atol=1e-6):
            raise ValueError(f"bucket probabilities must sum to 1, got {total}")
        for name in ("p_zero", "p_1_to_59", "p_60_plus"):
            value = getattr(self, name)
            if not -1e-9 <= value <= 1.0 + 1e-9:
                raise ValueError(f"{name}={value} out of [0, 1]")

    @property
    def expected_minutes(self) -> float:
        """Overall expected minutes — the ``expected_minutes / 90`` scaling factor every
        per-90-rate component downstream multiplies by (BUILD_PLAN 2.2/2.3/2.5)."""
        return (
            self.p_1_to_59 * self.expected_minutes_given_1_to_59
            + self.p_60_plus * self.expected_minutes_given_60_plus
        )

    @property
    def p_60_plus_or_more(self) -> float:
        """Clean-sheet/appearance-point eligibility gate (BUILD_PLAN 2.4/2.6): P(played 60+)."""
        return self.p_60_plus


@dataclass
class _SafeBinaryClassifier:
    """Wraps a :class:`LogisticRegression`, tolerating a single-class training sample — which
    sklearn itself refuses to fit (``ValueError: needs samples of at least 2 classes``), but which
    small/synthetic or early-season-thin data hits easily. Degenerates to "always predict that one
    class with certainty" rather than crashing.
    """

    classifier: LogisticRegression = field(
        default_factory=lambda: LogisticRegression(max_iter=1000)
    )
    _only_class: int | None = field(default=None, init=False, repr=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> _SafeBinaryClassifier:
        unique = np.unique(y)
        if len(unique) == 0:
            # No rows at all for this path (e.g. no "did not start" rows in-sample) — there's no
            # evidence either way, so default to "never" rather than crashing on an empty fit.
            self._only_class = 0
        elif len(unique) == 1:
            self._only_class = int(unique[0])
        else:
            self._only_class = None
            self.classifier.fit(X, y)
        return self

    def positive_proba(self, X: np.ndarray) -> np.ndarray:
        if self._only_class is not None:
            return np.full(X.shape[0], 1.0 if self._only_class == 1 else 0.0)
        positive_index = list(self.classifier.classes_).index(1)
        return self.classifier.predict_proba(X)[:, positive_index]


@dataclass
class MinutesModel:
    """Two-stage minutes model (2.1).

    **Documented simplification** (explainable-over-clever): a substitute appearance is modelled
    as always landing in the 1-59 bucket. A very late introduction can in principle reach 60+
    minutes, but this is rare enough that a third nested stage isn't worth the complexity here —
    revisit only if Phase 3 backtesting shows it materially costs accuracy.
    """

    start_classifier: _SafeBinaryClassifier = field(default_factory=_SafeBinaryClassifier)
    withdrawal_classifier: _SafeBinaryClassifier = field(default_factory=_SafeBinaryClassifier)
    sub_appearance_classifier: _SafeBinaryClassifier = field(default_factory=_SafeBinaryClassifier)
    minutes_given_60_plus: LinearRegression = field(default_factory=LinearRegression)
    minutes_given_started_under_60: LinearRegression = field(default_factory=LinearRegression)
    minutes_given_sub_appearance: LinearRegression = field(default_factory=LinearRegression)

    _fitted: bool = field(default=False, init=False, repr=False)
    _has_60_plus_data: bool = field(default=False, init=False, repr=False)
    _has_started_under_60_data: bool = field(default=False, init=False, repr=False)
    _has_sub_appearance_data: bool = field(default=False, init=False, repr=False)

    def fit(self, features: pd.DataFrame, started: pd.Series, minutes: pd.Series) -> MinutesModel:
        """``started``: bool/0-1, was in the starting XI. ``minutes``: actual minutes played
        (0 if an unused substitute or not in the matchday squad at all)."""
        X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
        started_arr = started.to_numpy().astype(int)
        minutes_arr = minutes.to_numpy(dtype=float)

        self.start_classifier.fit(X, started_arr)

        mask_started = started_arr == 1
        X_started = X[mask_started]
        minutes_started = minutes_arr[mask_started]
        withdrawn_before_60 = (minutes_started < MINUTES_60_PLUS_THRESHOLD).astype(int)
        self.withdrawal_classifier.fit(X_started, withdrawn_before_60)

        mask_60_plus = minutes_started >= MINUTES_60_PLUS_THRESHOLD
        self._has_60_plus_data = bool(mask_60_plus.any())
        if self._has_60_plus_data:
            self.minutes_given_60_plus.fit(X_started[mask_60_plus], minutes_started[mask_60_plus])

        mask_under_60_started = ~mask_60_plus
        self._has_started_under_60_data = bool(mask_under_60_started.any())
        if self._has_started_under_60_data:
            self.minutes_given_started_under_60.fit(
                X_started[mask_under_60_started], minutes_started[mask_under_60_started]
            )

        mask_not_started = started_arr == 0
        X_not_started = X[mask_not_started]
        minutes_not_started = minutes_arr[mask_not_started]
        appeared = (minutes_not_started > 0).astype(int)
        self.sub_appearance_classifier.fit(X_not_started, appeared)

        mask_appeared = appeared == 1
        self._has_sub_appearance_data = bool(mask_appeared.any())
        if self._has_sub_appearance_data:
            self.minutes_given_sub_appearance.fit(
                X_not_started[mask_appeared], minutes_not_started[mask_appeared]
            )

        self._fitted = True
        return self

    def _conditional_minutes(
        self,
        model: LinearRegression,
        has_data: bool,
        default: float,
        X: np.ndarray,
        lo: float,
        hi: float,
    ) -> np.ndarray:
        if has_data:
            predicted = model.predict(X)
        else:
            predicted = np.full(X.shape[0], default)
        return np.clip(predicted, lo, hi)

    def predict(self, features: pd.DataFrame) -> list[MinutesDistribution]:
        if not self._fitted:
            raise RuntimeError("MinutesModel.predict called before fit")

        X = features[FEATURE_COLUMNS].to_numpy(dtype=float)
        n = X.shape[0]

        p_start = self.start_classifier.positive_proba(X)
        p_not_start = 1.0 - p_start
        p_withdrawn_given_start = self.withdrawal_classifier.positive_proba(X)
        p_60_given_start = 1.0 - p_withdrawn_given_start
        p_sub_appearance_given_not_start = self.sub_appearance_classifier.positive_proba(X)

        p_60_plus = p_start * p_60_given_start
        p_1_59_from_start = p_start * p_withdrawn_given_start
        p_1_59_from_sub = p_not_start * p_sub_appearance_given_not_start
        p_1_59 = p_1_59_from_start + p_1_59_from_sub
        p_zero = p_not_start * (1.0 - p_sub_appearance_given_not_start)

        minutes_60_plus = self._conditional_minutes(
            self.minutes_given_60_plus,
            self._has_60_plus_data,
            DEFAULT_MINUTES_60_PLUS,
            X,
            60.0,
            90.0,
        )
        minutes_started_under_60 = self._conditional_minutes(
            self.minutes_given_started_under_60,
            self._has_started_under_60_data,
            DEFAULT_MINUTES_STARTED_UNDER_60,
            X,
            1.0,
            59.0,
        )
        minutes_sub_appearance = self._conditional_minutes(
            self.minutes_given_sub_appearance,
            self._has_sub_appearance_data,
            DEFAULT_MINUTES_SUB_APPEARANCE,
            X,
            1.0,
            59.0,
        )

        results: list[MinutesDistribution] = []
        for i in range(n):
            denom = p_1_59_from_start[i] + p_1_59_from_sub[i]
            if denom > 0:
                minutes_1_59 = (
                    p_1_59_from_start[i] * minutes_started_under_60[i]
                    + p_1_59_from_sub[i] * minutes_sub_appearance[i]
                ) / denom
            else:
                minutes_1_59 = 0.0

            # Renormalize defensively against floating-point drift so probabilities sum to
            # exactly 1 (MinutesDistribution enforces this strictly).
            total = p_zero[i] + p_1_59[i] + p_60_plus[i]
            results.append(
                MinutesDistribution(
                    p_zero=p_zero[i] / total,
                    p_1_to_59=p_1_59[i] / total,
                    p_60_plus=p_60_plus[i] / total,
                    expected_minutes_given_1_to_59=minutes_1_59,
                    expected_minutes_given_60_plus=minutes_60_plus[i],
                )
            )
        return results
