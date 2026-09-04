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

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Cross-validation folds for the isotonic calibration layer (see :func:`_make_classifier`). Fit
# entirely within the training fold, so this stays point-in-time safe.
CALIBRATION_FOLDS = 5

# Fixed seed for the calibration layer's own StratifiedKFold split (see :func:`_make_classifier`).
# Not tuned -- any fixed value works equally well; what matters is that it IS fixed.
CALIBRATION_RANDOM_STATE = 0

logger = logging.getLogger(__name__)

MINUTES_60_PLUS_THRESHOLD = 60

# Per BUILD_PLAN 2.1 "Inputs" table, extended per ENGINE_IMPROVEMENTS.md Tier 1.1 — the real
# single-season backtest found the model starved of signal (AUC 0.81 for "played at all", 6,618
# points misallocated to players who never appeared) because three of the five original features
# were hardcoded/defaulted in the backtest driver, and the single EWMA start-rate level couldn't
# express "recently nailed-on but historically fringe" (or the reverse). Position and squad depth
# remain deliberately excluded — see "Considered and cut" in BUILD_PLAN 2.1 (structural redundancy
# / no clean data source respectively). Every feature below is computed point-in-time-safe by the
# backtest driver's feature-engineering stage (``backtest/run_season.py``), not by this module.
#
# `price`/`ownership_log`/`transfers_out_share`/`transfers_balance_share` (ENGINE_IMPROVEMENTS_2.md
# B.3) were added after a real walk-forward ablation: the shipped 11-feature set alone scored AUC
# 0.8655 for "played at all"; adding these four (already sitting unused in the same archive) raised
# it to 0.8859, clearing the >0.88 target ENGINE_IMPROVEMENTS.md recorded as missed. The single
# largest contributor was `transfers_out_share` alone (+0.012 AUC) — mass transfers-out is the
# crowd reacting in real time to *this week's* injury news, making it a retrospectively-available
# proxy for exactly the two live-only fields (`chance_of_playing_next_round`, `status`) this
# backtest otherwise cannot reconstruct. Kept in the *same* shared FEATURE_COLUMNS list the live
# pipeline also uses (not a backtest-only subset) so there is only ever one feature contract for
# this model; `chance_of_playing_next_round`/`status_score` stay in the list even though they are
# constant (zero standard deviation) throughout this specific backtest, since they carry real
# signal on the live path this same module serves.
FEATURE_COLUMNS = [
    "recent_start_rate",
    "recent_minutes_ewma",
    "fixture_congestion",
    "chance_of_playing_next_round",
    "status_score",
    "days_since_last_appearance",
    "zero_minute_streak_length",
    "start_rate_last_3",
    "start_rate_last_6",
    "start_rate_last_15",
    "team_rotation_propensity",
    "price",
    "ownership_log",
    "transfers_out_share",
    "transfers_balance_share",
    # ENGINE_IMPROVEMENTS_3.md Phase 3 (goalkeepers): BUILD_PLAN 2.1's "Considered and cut:
    # Position" note reasons from the *bonus* model (2.8, which already fits one model per position,
    # so within a single-position model position carries zero information) — that reasoning doesn't
    # transfer to
    # this model, which is fit once across every position. A real multi-season walk-forward backtest
    # found goalkeepers have a qualitatively more binary minutes pattern (nailed-on ever-present or
    # completely out of the squad, far less mid-match substitution/rotation than outfield players)
    # that the other 15 features don't capture: without this feature GK P(60+) was under-predicted
    # by 7.3pp and P(zero) over-predicted by 5.5pp (real 2022/23-2025/26 pooled sample); adding it
    # cut
    # those gaps to 5.1pp/2.7pp with no measurable change to non-GK calibration or AUC (0.8203 both
    # ways). Population is a plain 0.0/1.0 flag callers derive from ``position``, not implicitly
    # derived here, matching every other feature's explicit-column contract.
    "is_goalkeeper",
]

# Ceiling on the magnitude of the two transfer "share" features, applied by
# ``backtest.run_season.engineer_features`` where they are computed. Both are a transfer count
# divided by the player's own owner count, so they are only meaningful when that denominator is a
# real ownership base. It is not always: the archive's `selected` is a genuine raw count (only 2 of
# 14,120 real training rows fall to the divide-by-zero floor), but the live path derives it from
# FPL's `selected_by_percent`, which is reported to one decimal place and so reads exactly 0.0 for
# every player under 0.05% ownership. On a real 2026/27 GW3 pull that was 73 of 486 projected
# players, and the floored denominator turned their "share" into a bare transfer count: one 8-minute
# substitute scored transfers_balance_share 351.0 against a training standard deviation of 0.127,
# which standardizes to 2,756 sigma and saturates the start classifier to a near-certain start.
# Clipping bounds that blow-up on both paths regardless of what the denominator does, the same
# fix-the-input-at-the-source approach as engine.models.goals' MAX_NPXG_PER_90_PER_MATCH. Set
# comfortably above the real training range (observed max 1.5, 99th percentile 0.46) so a genuine
# high-churn gameweek is untouched, and applied symmetrically since transfers_balance_share is
# signed.
MAX_TRANSFER_SHARE = 2.0

# FPL's `status` code -> a coarse numeric availability proxy. `news` free text is deliberately
# NOT parsed into a feature (BUILD_PLAN 2.1: display flag only) — chance_of_playing_next_round
# and status already carry the same information as a clean structured number. `n` (not in squad,
# typically a player out on loan) is included alongside the five documented FPL codes so that a
# loaned-out player scores as fully unavailable rather than crashing feature encoding, per the
# 2026-08-20 engine audit (ENGINE_AUDIT_FIXES-implementation.md T-B).
STATUS_SCORE = {"a": 1.0, "d": 0.75, "i": 0.0, "s": 0.0, "u": 0.0, "n": 0.0}

# Fallback conditional-minutes estimates for a bucket with no fitted training examples (e.g. a
# thin synthetic sample, or a position that never has sub-appearance rows in-sample). Rough
# league-average priors, not asserted as calibrated — real calibration is Phase 3's job.
DEFAULT_MINUTES_60_PLUS = 80.0
DEFAULT_MINUTES_STARTED_UNDER_60 = 45.0
DEFAULT_MINUTES_SUB_APPEARANCE = 20.0

# A player FPL's live data marks as definitely not playing (status_score 0.0, meaning status
# i/s/u/n, or an explicit 0% chance_of_playing_next_round) gets their bucket probabilities
# floored here, after the fitted model's own prediction, rather than trusting the model to have
# learned this relationship. It could not have: chance_of_playing_next_round/status_score are
# live-only fields with zero variance across every real historical training row (that data was
# never retrospectively available, so every backtest row is hardcoded to "fully fit"), so a
# model fit on that data learns essentially no weight for either feature (2026-08-20 engine
# audit, ENGINE_AUDIT_FIXES-implementation.md T-F's verified finding). This floor is a no-op on
# the backtest path, since status_score/chance_of_playing_next_round never take these values
# there. Not floored all the way to exactly 0/1: a small residual probability is kept for the
# rare late fitness reprieve, and to avoid a degenerate all-or-nothing distribution feeding a
# division or log elsewhere downstream.
KNOWN_UNAVAILABLE_P_ZERO = 0.98


def encode_status(status: str) -> float:
    """FPL `status` (a/d/i/s/u/n) -> numeric availability proxy for :data:`FEATURE_COLUMNS`.

    Decision (2026-08-20 engine audit, ENGINE_AUDIT_FIXES-implementation.md T-B): a status code not
    in :data:`STATUS_SCORE` degrades to fully unavailable (0.0) with a logged warning rather than
    raising. FPL has introduced new status codes before and will again, and this feature is only a
    coarse numeric proxy, not the sole availability signal; a single unrecognised code should not be
    able to take down a scheduled projection build. Callers that need to detect the degradation
    should watch the `engine.models.minutes` logger rather than catching an exception.
    """
    try:
        return STATUS_SCORE[status]
    except KeyError:
        logger.warning("unknown FPL status code %r, treating as unavailable (0.0)", status)
        return 0.0


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


def _make_base_classifier() -> Pipeline:
    """Standardize, then fit L2-penalized logistic regression.

    The scaler is not cosmetic. :data:`FEATURE_COLUMNS` mixes 0-1 rates with
    ``days_since_last_appearance`` (0-60), ``price`` (~40-150) and ``ownership_log`` (0-16) — four
    orders of magnitude. Unscaled, L-BFGS did not converge within ``max_iter`` on the real
    2025/26 sample (sklearn emitted a convergence warning on every one of the 35 walk-forward
    refits), so the shipped coefficients were wherever the optimizer happened to stop. Worse, a
    single L2 penalty applied across features on such different scales penalizes the small-valued
    rate features far more heavily than the large-valued crowd features, which is not a modelling
    choice anyone made — it is an artifact of the units the columns happen to be in.
    """
    return Pipeline([("scale", StandardScaler()), ("logit", LogisticRegression(max_iter=1000))])


def _make_classifier() -> CalibratedClassifierCV:
    """The scaled logistic model wrapped in an isotonic calibration layer.

    B2: the shipped model's play probabilities are well calibrated in aggregate (MACE 0.0262) but
    not conditionally — measured on the real 2025/26 walk-forward, the 0.2-0.4 band predicted
    0.301 against a realised 0.219, and the 0.4-0.6 band predicted 0.510 against 0.417, while the
    two top bands were near-exact. That mid-band over-confidence is precisely the population that
    generates predicted points for players who then do not appear, so it shows up downstream as
    the zero-minute predicted-points mass rather than as a headline calibration failure.

    Isotonic (not Platt) because the error is not a monotone squeeze of the whole curve — the
    lowest band is *under*-predicted (0.060 against 0.097) while the middle is over-predicted, an
    S-shape a single sigmoid cannot express. ``cv`` folds are drawn from the training fold only, so
    no future information enters; :class:`_SafeBinaryClassifier` drops back to the bare pipeline
    when a class is too thin to cross-validate.

    ``cv`` is an explicit, seeded, shuffled :class:`~sklearn.model_selection.StratifiedKFold`
    rather than the bare integer ``CALIBRATION_FOLDS`` sklearn would otherwise default to — a
    bare integer defaults to ``shuffle=False``, which assigns folds by the training data's own row
    order. Two builds fit on the same real rows in a different order (e.g. a different Understat/
    vaastav fetch ordering) then land in different folds, so the isotonic calibration curve, and
    therefore every downstream ``p_60_plus``, differs between two builds that are meant to be
    identical. Confirmed directly: fitting on identical rows in shuffled order versus original
    order produced different individual predictions with nothing else changed. Shuffling with a
    fixed ``random_state`` makes the fold assignment (and so the fitted model) depend only on the
    data's *content*, never its incidental arrival order.
    """
    return CalibratedClassifierCV(
        _make_base_classifier(),
        method="isotonic",
        cv=StratifiedKFold(
            n_splits=CALIBRATION_FOLDS, shuffle=True, random_state=CALIBRATION_RANDOM_STATE
        ),
    )


@dataclass
class _SafeBinaryClassifier:
    """Wraps a :class:`LogisticRegression`, tolerating a single-class training sample — which
    sklearn itself refuses to fit (``ValueError: needs samples of at least 2 classes``), but which
    small/synthetic or early-season-thin data hits easily. Degenerates to "always predict that one
    class with certainty" rather than crashing.
    """

    classifier: CalibratedClassifierCV = field(default_factory=_make_classifier)
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
            # Isotonic calibration needs enough of the minority class to cross-validate. A thin or
            # early-season sample can't support it, so fall back to the uncalibrated pipeline
            # rather than raising — the same "degrade, don't crash" contract as the single-class
            # path above.
            if np.bincount(y.astype(int)).min() < CALIBRATION_FOLDS:
                self.classifier = _make_base_classifier()
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
        known_unavailable = (features["status_score"].to_numpy(dtype=float) <= 0.0) | (
            features["chance_of_playing_next_round"].to_numpy(dtype=float) <= 0.0
        )

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

            if known_unavailable[i]:
                # See KNOWN_UNAVAILABLE_P_ZERO's own comment: the fitted model never learned a
                # real coefficient for this live-only signal, so it is applied here directly
                # instead of trusting the model's own (near-zero-weighted) prediction.
                row_p_zero = KNOWN_UNAVAILABLE_P_ZERO
                row_p_1_59 = 1.0 - KNOWN_UNAVAILABLE_P_ZERO
                row_p_60_plus = 0.0
            else:
                # Renormalize defensively against floating-point drift so probabilities sum to
                # exactly 1 (MinutesDistribution enforces this strictly).
                total = p_zero[i] + p_1_59[i] + p_60_plus[i]
                row_p_zero = p_zero[i] / total
                row_p_1_59 = p_1_59[i] / total
                row_p_60_plus = p_60_plus[i] / total

            results.append(
                MinutesDistribution(
                    p_zero=row_p_zero,
                    p_1_to_59=row_p_1_59,
                    p_60_plus=row_p_60_plus,
                    expected_minutes_given_1_to_59=minutes_1_59,
                    expected_minutes_given_60_plus=minutes_60_plus[i],
                )
            )
        return results


# Real 22-a-side (11 vs 11) split for one club's own starting XI. A club normally starts 1 GK and
# 10 outfield players in one fixture — the same real invariant fixture_minutes_coverage already
# checks across BOTH squads combined (target 22). Checking it per club, split by position, catches
# what the pooled fixture check can't: two clubs whose totals happen to average out to 22 (one
# over-covered, one under) still each individually diverge from a real XI. A real 2026/27 GW3 pull
# found club-level outfield p_60_plus sums ranging 8.7 to 15.3 against this target of 10, and two
# of one club's own goalkeepers both individually rated 76% to start the same match.
OUTFIELD_STARTERS_PER_CLUB = 10.0
GOALKEEPER_STARTERS_PER_CLUB = 1.0

# Ceiling an individual player's p_60_plus is allowed to reach when normalize_within_club scales it
# up. Never exactly 1.0 -- even the most nailed-on starter carries some real chance of a late
# fitness call, so a hard ceiling below 1.0 keeps the rescaled distribution honest about that,
# rather than the rescale itself manufacturing false certainty.
NORMALIZE_WITHIN_CLUB_CEILING = 0.97


def _rescale_p_60_plus_to_target(values: list[float], target: float) -> list[float]:
    """Proportionally rescale ``values`` so they sum to ``target``, capping any individual value
    at :data:`NORMALIZE_WITHIN_CLUB_CEILING` and redistributing the remainder among the
    still-uncapped values (a standard bounded water-filling rescale) rather than letting a naive
    single-pass proportional scale push a value above 1 -- MinutesDistribution's own validation
    would reject that outright, and a value close to but under 1 would still overstate how certain
    any real minutes projection should be.

    Values already at or above zero are all treated as eligible for scaling -- callers exclude a
    known-unavailable player from ``values`` entirely before calling this, rather than relying on
    this function to recognise one (see :func:`normalize_within_club`)."""
    n = len(values)
    if n == 0:
        return []
    current_sum = sum(values)
    if current_sum <= 0.0:
        # No signal to redistribute (every eligible player already at exactly 0) -- leave as is
        # rather than dividing by zero or inventing a uniform split with no evidence behind it.
        return list(values)

    remaining_target = target
    uncapped_indices = set(range(n))
    result = [0.0] * n
    # At most n iterations: each pass caps at least one more value, or terminates.
    for _ in range(n):
        if not uncapped_indices:
            break
        uncapped_sum = sum(values[i] for i in uncapped_indices)
        if uncapped_sum <= 0.0:
            break
        scale = remaining_target / uncapped_sum
        newly_capped = [
            i for i in uncapped_indices if values[i] * scale > NORMALIZE_WITHIN_CLUB_CEILING
        ]
        if not newly_capped:
            for i in uncapped_indices:
                result[i] = values[i] * scale
            return result
        for i in newly_capped:
            result[i] = NORMALIZE_WITHIN_CLUB_CEILING
            remaining_target -= NORMALIZE_WITHIN_CLUB_CEILING
            uncapped_indices.discard(i)
    # Every remaining value ended up capped (or the loop otherwise exhausted its passes) -- assign
    # whatever's left proportionally to the still-uncapped set, clipped defensively.
    for i in uncapped_indices:
        result[i] = min(values[i], NORMALIZE_WITHIN_CLUB_CEILING)
    return result


def normalize_within_club(
    distributions: list[MinutesDistribution],
    is_known_unavailable: list[bool],
    target: float = OUTFIELD_STARTERS_PER_CLUB,
) -> list[MinutesDistribution]:
    """Rescale one club's own ``p_60_plus`` values so they sum close to ``target`` rather than
    whatever the per-player minutes model happened to independently predict. Called once per club
    per position group (see :mod:`engine.pipeline`'s caller) — once for outfield players with
    ``target=`` :data:`OUTFIELD_STARTERS_PER_CLUB`, once for goalkeepers with ``target=``
    :data:`GOALKEEPER_STARTERS_PER_CLUB` — since a real starting XI's own goalkeeper/outfield split
    (1 vs 10) is a separate physical constraint from the outfield total.

    Every prior stage of this engine scores one player at a time — the minutes model, every
    downstream component, even :func:`MinutesModel.predict` itself, has no way to see that its own
    predictions, summed across one club's own squad, must respect a real physical constraint (one
    XI, one goalkeeper). A real 2026/27 GW3 pull found this un-checked: club-level outfield
    p_60_plus sums ranged 8.7 to 15.3 against a true 10, and two goalkeepers at the same club both
    individually rated 76% to start.

    ``is_known_unavailable`` (parallel to ``distributions``) marks each player already floored by
    :data:`KNOWN_UNAVAILABLE_P_ZERO` — excluded entirely from the rescale (their ``p_60_plus`` is
    already 0.0 and must stay there; rescaling would otherwise be free to inflate an injured
    player's minutes back up purely to make the club total add up).

    ``p_zero``/``p_1_to_59`` are adjusted alongside ``p_60_plus`` to keep each row's own three
    buckets summing to exactly 1: the probability mass gained or lost by ``p_60_plus`` is taken
    from, or returned to, ``p_zero`` and ``p_1_to_59`` in proportion to their existing split (the
    minimal-assumption redistribution — this function has no basis to prefer moving mass from one
    of those two buckets over the other).
    """
    if len(distributions) != len(is_known_unavailable):
        raise ValueError("distributions and is_known_unavailable must be the same length")
    if not distributions:
        return []

    eligible_indices = [i for i, excluded in enumerate(is_known_unavailable) if not excluded]
    eligible_p_60_plus = [distributions[i].p_60_plus for i in eligible_indices]
    rescaled = _rescale_p_60_plus_to_target(eligible_p_60_plus, target)
    new_p_60_plus_by_index = dict(zip(eligible_indices, rescaled, strict=True))

    results: list[MinutesDistribution] = []
    for i, distribution in enumerate(distributions):
        if i not in new_p_60_plus_by_index:
            results.append(distribution)
            continue
        new_p_60_plus = new_p_60_plus_by_index[i]
        old_total_low = distribution.p_zero + distribution.p_1_to_59
        new_total_low = 1.0 - new_p_60_plus
        if old_total_low > 0.0:
            factor = new_total_low / old_total_low
            new_p_zero = distribution.p_zero * factor
            new_p_1_to_59 = distribution.p_1_to_59 * factor
        else:
            # p_60_plus was already 1.0 (no low-bucket mass to scale) -- split whatever's now
            # freed/needed evenly between the two buckets as a neutral fallback.
            new_p_zero = new_total_low / 2.0
            new_p_1_to_59 = new_total_low / 2.0
        results.append(
            MinutesDistribution(
                p_zero=new_p_zero,
                p_1_to_59=new_p_1_to_59,
                p_60_plus=new_p_60_plus,
                expected_minutes_given_1_to_59=distribution.expected_minutes_given_1_to_59,
                expected_minutes_given_60_plus=distribution.expected_minutes_given_60_plus,
            )
        )
    return results
