"""Engine Definition of Done — the gate to Phase 5 (3.6).

Nothing in the web phase starts until every criterion below holds over a meaningful sample (ideally
a full season — BUILD_PLAN 3.6 notes a good 5-gameweek run tells you almost nothing given
football's variance). No carve-out for defensive contribution or bonus despite their shallower
native history: the statistical tests already feeding this gate (``backtest.baselines``) widen
their own confidence interval to account for a thinner sample on their own, so a genuine signal
still clears the bar and a signal that doesn't clear it is real information — not ready to trust
yet — rather than something to special-case around (BUILD_PLAN 3.6).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backtest.baselines import PairedBootstrapResult, PermutationTestResult
from backtest.metrics import (
    BiasReport,
    CalibrationReport,
    ClubMinutesCoverageReport,
    DecisionSetRankReport,
    FixtureBonusTotalReport,
    FixtureMinutesCoverageReport,
    GoalkeeperSavesReport,
    HorizonMonotonicityReport,
    MeanCalibrationReport,
    MinutesBucketShareReport,
)

# ENGINE_IMPROVEMENTS_3.md B.1: the original 0.15 threshold was never derived from anything —
# every component ever measured has scored between 0.004 and 0.027, six to forty times tighter,
# so the gate could not fail on calibration even at the pre-B.1 defensive-contribution defect
# (0.0306, a 34% under-prediction). Set from the measured distribution with headroom for a
# component that's merely okay rather than excellent, not from a guess.
DEFAULT_MAX_CALIBRATION_ERROR = 0.05

# B3: threshold for the continuous (non-probability) components' mean-calibration relative gap
# (goals/assists/bonus) — the same 5% ENGINE_IMPROVEMENTS_3.md's own re-measurement checklist set
# for bonus specifically, applied uniformly. Must be scored on played rows only (see
# `run_season.score_season`'s `mean_calibrations_played` construction) — scored on all rows this
# is dominated by the minutes model, not the component being checked.
DEFAULT_MAX_MEAN_CALIBRATION_RELATIVE_GAP = 0.05

# T-J (planning/ENGINE_AUDIT_FIXES-implementation.md): aggregate, per-fixture and pool-wide
# plausibility invariants. Every defect this task exists to catch was invisible to the
# per-player metrics above because none of them sum or average across a fixture or the whole
# pool. These tolerances are initial, documented starting points tied directly to the audit's own
# measured baseline table, not a long calibration history the way DEFAULT_MAX_CALIBRATION_ERROR
# was. Tighten them once a season of passing runs establishes real week-to-week noise.
DEFAULT_FIXTURE_MINUTES_COVERAGE_TARGET = 22.0
DEFAULT_FIXTURE_MINUTES_COVERAGE_TOLERANCE = 1.0
# Real 2026/27 GW3 pull: club-level outfield p_60_plus sums ranged 8.7 to 15.3 against a true 10,
# and two goalkeepers at the same club both individually rated 76% to start (a group-of-2 gap of
# 0.52 against a true 1.0) -- a defect the fixture-combined check above cannot isolate (see
# ClubMinutesCoverageReport's own docstring). Set tight enough to catch that real goalkeeper case
# (a group whose target is only 1.0, so even a modest absolute gap is a large relative one) rather
# than the fixture-combined check's own 1.0, which was sized for a target of 22 and would pass
# that same 0.52 gap outright. A single absolute tolerance applied to both the goalkeeper (target
# 1) and outfield (target 10) groups is the same simplification fixture_minutes_coverage's own
# single tolerance already makes; revisit with two separate tolerances if a real walk-forward run
# shows outfield noise alone routinely tripping this.
DEFAULT_CLUB_MINUTES_COVERAGE_TOLERANCE = 0.5
DEFAULT_FIXTURE_BONUS_TOTAL_TARGET = 6.0
DEFAULT_FIXTURE_BONUS_TOTAL_TOLERANCE = 0.5
DEFAULT_MINUTES_BUCKET_SHARE_TOLERANCE = 0.03
# Real 2025-26 season, position-agnostic, every registered player row
# (data_store/season_cache/vaastav/2025-26/merged_gw.parquet).
REAL_2025_26_MINUTES_BUCKET_SHARES = {"p_zero": 0.614, "p_1_to_59": 0.124, "p_60_plus": 0.263}
DEFAULT_GOALKEEPER_SAVES_RELATIVE_TOLERANCE = 0.10
# Real 2025-26 season, position == "GK" and minutes >= 60, same parquet as above.
REAL_2025_26_GOALKEEPER_SAVES_PER_MATCH = 2.78
# Consecutive-gameweek drop in mean p_60_plus, in probability points, allowed before it counts as
# systematic decay rather than ordinary week-to-week noise.
DEFAULT_HORIZON_DECAY_TOLERANCE = 0.01

# ENGINE_IMPROVEMENTS_5.md Tier 0.1: the minimum mean within-shortlist rank correlation
# (:class:`~backtest.metrics.DecisionSetRankReport`) before the engine counts as able to order the
# players it recommends. Set at 0.15 against a measured 0.049 on the real 2025/26 walk-forward, so
# this criterion **fails the tree it was written against** — deliberately. Every other ranking
# number the gate could have read is dominated by the will-they-play axis (pooled Spearman 0.636 is
# matched by `1 - p_zero` alone at 0.643), so no pre-existing criterion could fail on the shortlist
# being unordered. 0.15 is roughly half the honest oracle ceiling for this quantity, not a value
# the current engine is close to; revise it from a measured distribution once a fix lands, the same
# way B.1 replaced the underived 0.15 calibration threshold.
DEFAULT_MIN_DECISION_SET_SPEARMAN = 0.15

# Tier 0.1: shortlist size the criterion above is measured at. 20 is the population a manager
# actually chooses between across captaincy and a transfer shortlist, and is the size the -0.049
# baseline was measured at.
DEFAULT_DECISION_SET_TOP_N = 20

__all__ = [
    "DEFAULT_DECISION_SET_TOP_N",
    "DEFAULT_MIN_DECISION_SET_SPEARMAN",
    "DefinitionOfDoneReport",
    "evaluate_definition_of_done",
]

BaselineResult = PairedBootstrapResult | PermutationTestResult


@dataclass(frozen=True)
class DefinitionOfDoneReport:
    """One row per BUILD_PLAN 3.6 checklist item, plus the overall gate verdict."""

    beats_baselines: bool
    baseline_results: dict[str, BaselineResult]
    no_severe_bias: bool
    bias_reports: dict[str, BiasReport]
    calibration_acceptable: bool
    calibration_reports: dict[str, CalibrationReport]
    predictions_logged: bool
    trusted_by_user: bool
    # B3: continuous (non-probability) components — goals/assists/bonus — scored on played rows.
    # Defaults keep this gate a no-op for callers that don't supply it, so it doesn't retroactively
    # fail a report built before this check existed.
    mean_calibration_acceptable: bool = True
    mean_calibration_reports: dict[str, MeanCalibrationReport] = field(default_factory=dict)
    # T-J: aggregate/fixture-level/pool-wide plausibility invariants. Each defaults to a no-op
    # (True, report None) so a caller that doesn't compute one of these, e.g. no fixture_id column
    # available yet, doesn't retroactively fail a report built before this check existed, the same
    # convention `mean_calibration_reports` established for B3.
    fixture_minutes_coverage_acceptable: bool = True
    fixture_minutes_coverage_report: FixtureMinutesCoverageReport | None = None
    club_minutes_coverage_acceptable: bool = True
    club_minutes_coverage_report: ClubMinutesCoverageReport | None = None
    fixture_bonus_total_acceptable: bool = True
    fixture_bonus_total_report: FixtureBonusTotalReport | None = None
    minutes_bucket_shares_acceptable: bool = True
    minutes_bucket_share_report: MinutesBucketShareReport | None = None
    goalkeeper_saves_acceptable: bool = True
    goalkeeper_saves_report: GoalkeeperSavesReport | None = None
    horizon_minutes_non_decaying: bool = True
    horizon_minutes_report: HorizonMonotonicityReport | None = None
    # ENGINE_IMPROVEMENTS_5.md Tier 0.1: can the engine order its own shortlist, and (Tier 2.1)
    # is E[points | plays] unbiased on the players who did play. Both default to a no-op so a
    # caller that hasn't wired them up doesn't retroactively fail, the same convention every check
    # since B3 has used. `conditional_bias_reports` must score `conditional_expected_points`; see
    # evaluate_definition_of_done's docstring for why scoring `expected_points` there is
    # unpassable by construction rather than informative.
    decision_set_ranking_acceptable: bool = True
    decision_set_rank_report: DecisionSetRankReport | None = None
    no_severe_conditional_bias: bool = True
    conditional_bias_reports: dict[str, BiasReport] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (
            self.beats_baselines
            and self.no_severe_bias
            and self.no_severe_conditional_bias
            and self.calibration_acceptable
            and self.mean_calibration_acceptable
            and self.decision_set_ranking_acceptable
            and self.fixture_minutes_coverage_acceptable
            and self.club_minutes_coverage_acceptable
            and self.fixture_bonus_total_acceptable
            and self.minutes_bucket_shares_acceptable
            and self.goalkeeper_saves_acceptable
            and self.horizon_minutes_non_decaying
            and self.predictions_logged
            and self.trusted_by_user
        )

    def summary(self) -> str:
        checklist = {
            "Beats all baselines, statistically (3.3)": self.beats_baselines,
            "No severe systematic bias in any position/price tier (3.2)": self.no_severe_bias,
            "E[points | plays] unbiased on players who played (5.2.1)": (
                self.no_severe_conditional_bias
            ),
            "Each component reasonably calibrated (3.2)": (
                self.calibration_acceptable and self.mean_calibration_acceptable
            ),
            "Orders its own shortlist better than chance (5.0.1)": (
                self.decision_set_ranking_acceptable
            ),
            "Aggregate/fixture-level plausibility holds (T-J)": (
                self.fixture_minutes_coverage_acceptable
                and self.club_minutes_coverage_acceptable
                and self.fixture_bonus_total_acceptable
                and self.minutes_bucket_shares_acceptable
                and self.goalkeeper_saves_acceptable
                and self.horizon_minutes_non_decaying
            ),
            "Predictions logged immutably, tagged by model version (3.4)": self.predictions_logged,
            "Personally trusted enough to act on (3.6)": self.trusted_by_user,
        }
        lines = [f"[{'x' if ok else ' '}] {label}" for label, ok in checklist.items()]
        lines.append(f"\n{'PASSED' if self.passed else 'NOT YET'} — gate to Phase 5")
        return "\n".join(lines)


def evaluate_definition_of_done(
    baseline_results: dict[str, BaselineResult],
    bias_reports: dict[str, BiasReport],
    calibration_reports: dict[str, CalibrationReport],
    predictions_logged: bool,
    trusted_by_user: bool,
    max_acceptable_calibration_error: float = DEFAULT_MAX_CALIBRATION_ERROR,
    calibration_error_thresholds: Mapping[str, float] | None = None,
    mean_calibration_reports: Mapping[str, MeanCalibrationReport] | None = None,
    max_acceptable_mean_calibration_relative_gap: float = (
        DEFAULT_MAX_MEAN_CALIBRATION_RELATIVE_GAP
    ),
    mean_calibration_relative_gap_thresholds: Mapping[str, float] | None = None,
    fixture_minutes_coverage_report: FixtureMinutesCoverageReport | None = None,
    fixture_minutes_coverage_tolerance: float = DEFAULT_FIXTURE_MINUTES_COVERAGE_TOLERANCE,
    club_minutes_coverage_report: ClubMinutesCoverageReport | None = None,
    club_minutes_coverage_tolerance: float = DEFAULT_CLUB_MINUTES_COVERAGE_TOLERANCE,
    fixture_bonus_total_report: FixtureBonusTotalReport | None = None,
    fixture_bonus_total_tolerance: float = DEFAULT_FIXTURE_BONUS_TOTAL_TOLERANCE,
    minutes_bucket_share_report: MinutesBucketShareReport | None = None,
    minutes_bucket_share_tolerance: float = DEFAULT_MINUTES_BUCKET_SHARE_TOLERANCE,
    goalkeeper_saves_report: GoalkeeperSavesReport | None = None,
    goalkeeper_saves_relative_tolerance: float = DEFAULT_GOALKEEPER_SAVES_RELATIVE_TOLERANCE,
    horizon_minutes_report: HorizonMonotonicityReport | None = None,
    horizon_minutes_decay_tolerance: float = DEFAULT_HORIZON_DECAY_TOLERANCE,
    decision_set_rank_report: DecisionSetRankReport | None = None,
    min_decision_set_spearman: float = DEFAULT_MIN_DECISION_SET_SPEARMAN,
    conditional_bias_reports: Mapping[str, BiasReport] | None = None,
) -> DefinitionOfDoneReport:
    """Roll up backtest results into a pass/fail verdict against every BUILD_PLAN 3.6 criterion.

    ``baseline_results`` keys are baseline names (e.g. "template_captain", "naive_form",
    "pure_xg") mapped to their statistical test result — every one must have already beaten the
    engine's baseline (``beats_baseline`` True) for this gate to pass. ``bias_reports`` and
    ``calibration_reports`` keys are whatever grouping/component the caller checked (e.g. position,
    price tier, "clean_sheet", "defensive_contribution"). ``trusted_by_user`` is BUILD_PLAN 3.6's
    explicit "honest final check" — the one criterion this module cannot compute for you.

    ``calibration_error_thresholds``, if given, overrides ``max_acceptable_calibration_error`` on a
    per-component basis (ENGINE_IMPROVEMENTS_3.md B.1 — "make it per-component with each
    component's own historical value as the reference"), keyed the same way as
    ``calibration_reports``. A component missing from this mapping falls back to
    ``max_acceptable_calibration_error``.

    ``mean_calibration_reports`` (B3) is the continuous-component sibling of ``calibration_reports``
    — goals/assists/bonus aren't bounded-[0,1] probabilities, so they're scored by relative gap
    (:class:`~backtest.metrics.MeanCalibrationReport`) rather than MACE. Must be computed on played
    rows only (see ``run_season.score_season``) — computed on all rows it measures the minutes
    model, not the component. Omitted (the default), this check is a no-op and never fails the
    gate, so existing callers are unaffected.

    The ``*_report`` parameters from ``fixture_minutes_coverage_report`` onward (T-J,
    ``planning/ENGINE_AUDIT_FIXES-implementation.md``) are the aggregate/fixture-level/pool-wide
    plausibility invariants no per-player metric above can see: per-fixture minutes coverage
    (sum of ``p_60_plus`` near 22), per-club minutes coverage split goalkeeper/outfield (sum of
    ``p_60_plus`` near 1/10 respectively — catches a per-club skew, or two players at one club
    both individually reading as likely starters, that the fixture-combined check above cannot
    isolate), per-fixture bonus total (near 6.0), pool-wide minutes-bucket shares against the
    prior season's real empirical split, goalkeeper saves plausibility against the prior season's
    real per-match rate, and horizon minutes non-decay. Each is omitted by default, the same no-op
    convention as ``mean_calibration_reports``. A caller that hasn't wired one of these up yet
    (e.g. no ``fixture_id``/``team`` column available) doesn't retroactively fail a report built
    before this check existed.

    ``decision_set_rank_report`` and ``conditional_bias_reports`` (ENGINE_IMPROVEMENTS_5.md Tier
    0.1) close the gap this gate had between what it measured and what a manager acts on. Every
    other ranking criterion here is pooled across the whole pool, and pooled Spearman of 0.636 is
    matched by ``1 - p_zero`` alone at 0.643, so it measures availability prediction rather than
    points prediction and cannot fail on the engine's own shortlist being unordered (+0.049).
    ``decision_set_rank_report`` must clear ``min_decision_set_spearman``.

    ``conditional_bias_reports`` is keyed and shaped exactly like ``bias_reports`` and checked the
    same way (no group flagged ``severe``), and must be built with
    :func:`~backtest.metrics.bias_by_group`'s ``minutes_col`` set so it reads only rows where the
    player actually played. **It must score ``conditional_expected_points``, not
    ``expected_points``** (Tier 2.1 correction). ``expected_points`` is P(plays) times
    E[points | plays], so restricting to rows where the player did play selects the branch on which
    an unconditional expectation was always going to look low: a simulated model given P(plays) and
    E[points | plays] exactly scores -1.31 on that statistic, worse than the engine's -0.96, which
    makes it unpassable by construction rather than informative. Scored against the conditional
    prediction the same rows read -0.088, and the criterion becomes a real one.
    """
    beats_baselines = bool(baseline_results) and all(
        r.beats_baseline for r in baseline_results.values()
    )
    no_severe_bias = bool(bias_reports) and all(
        not report.by_group["severe"].any() for report in bias_reports.values()
    )
    thresholds = calibration_error_thresholds or {}
    calibration_acceptable = bool(calibration_reports) and all(
        report.mean_absolute_calibration_error
        <= thresholds.get(name, max_acceptable_calibration_error)
        for name, report in calibration_reports.items()
    )

    mean_calibration_reports = dict(mean_calibration_reports or {})
    mean_thresholds = mean_calibration_relative_gap_thresholds or {}
    # Absent entirely -> no-op (True); supplied but every relative_gap NaN (e.g. zero mean actual)
    # also can't be judged, so also no-op rather than an unjudgeable failure.
    mean_calibration_acceptable = not mean_calibration_reports or all(
        np.isnan(report.relative_gap)
        or report.relative_gap
        <= mean_thresholds.get(name, max_acceptable_mean_calibration_relative_gap)
        for name, report in mean_calibration_reports.items()
    )

    # T-J: each check below defaults to a no-op (acceptable=True) when its report isn't supplied,
    # matching mean_calibration_reports' own absent-is-fine convention.
    fixture_minutes_coverage_acceptable = (
        fixture_minutes_coverage_report is None
        or fixture_minutes_coverage_report.mean_absolute_gap <= fixture_minutes_coverage_tolerance
    )
    club_minutes_coverage_acceptable = (
        club_minutes_coverage_report is None
        or club_minutes_coverage_report.mean_absolute_gap <= club_minutes_coverage_tolerance
    )
    fixture_bonus_total_acceptable = (
        fixture_bonus_total_report is None
        or fixture_bonus_total_report.mean_absolute_gap <= fixture_bonus_total_tolerance
    )
    minutes_bucket_shares_acceptable = minutes_bucket_share_report is None or all(
        gap <= minutes_bucket_share_tolerance
        for gap in minutes_bucket_share_report.absolute_gaps.values()
    )
    goalkeeper_saves_acceptable = (
        goalkeeper_saves_report is None
        or np.isnan(goalkeeper_saves_report.relative_gap)
        or goalkeeper_saves_report.relative_gap <= goalkeeper_saves_relative_tolerance
    )
    horizon_minutes_non_decaying = horizon_minutes_report is None or not _horizon_is_decaying(
        horizon_minutes_report.by_gameweek, horizon_minutes_decay_tolerance
    )

    # Tier 0.1: absent -> no-op, matching every check above. A report with no scorable gameweek
    # (mean_spearman NaN) is unjudgeable rather than failing, the same way mean_calibration treats
    # a NaN relative_gap.
    decision_set_ranking_acceptable = (
        decision_set_rank_report is None
        or np.isnan(decision_set_rank_report.mean_spearman)
        or decision_set_rank_report.mean_spearman >= min_decision_set_spearman
    )
    conditional_bias_reports = dict(conditional_bias_reports or {})
    no_severe_conditional_bias = not conditional_bias_reports or all(
        not report.by_group["severe"].any() for report in conditional_bias_reports.values()
    )

    return DefinitionOfDoneReport(
        beats_baselines=beats_baselines,
        baseline_results=baseline_results,
        no_severe_bias=no_severe_bias,
        bias_reports=bias_reports,
        calibration_acceptable=calibration_acceptable,
        calibration_reports=calibration_reports,
        predictions_logged=predictions_logged,
        trusted_by_user=trusted_by_user,
        mean_calibration_acceptable=mean_calibration_acceptable,
        mean_calibration_reports=mean_calibration_reports,
        fixture_minutes_coverage_acceptable=fixture_minutes_coverage_acceptable,
        fixture_minutes_coverage_report=fixture_minutes_coverage_report,
        club_minutes_coverage_acceptable=club_minutes_coverage_acceptable,
        club_minutes_coverage_report=club_minutes_coverage_report,
        fixture_bonus_total_acceptable=fixture_bonus_total_acceptable,
        fixture_bonus_total_report=fixture_bonus_total_report,
        minutes_bucket_shares_acceptable=minutes_bucket_shares_acceptable,
        minutes_bucket_share_report=minutes_bucket_share_report,
        goalkeeper_saves_acceptable=goalkeeper_saves_acceptable,
        goalkeeper_saves_report=goalkeeper_saves_report,
        horizon_minutes_non_decaying=horizon_minutes_non_decaying,
        horizon_minutes_report=horizon_minutes_report,
        decision_set_ranking_acceptable=decision_set_ranking_acceptable,
        decision_set_rank_report=decision_set_rank_report,
        no_severe_conditional_bias=no_severe_conditional_bias,
        conditional_bias_reports=conditional_bias_reports,
    )


def _horizon_is_decaying(by_gameweek: pd.DataFrame, tolerance: float) -> bool:
    """True only if every consecutive gameweek step in ``by_gameweek`` drops by more than
    ``tolerance``. A single noisy dip amid an otherwise flat or rising horizon is ordinary
    football variance, not the systematic decay this check exists to catch (T-J item 5: mean
    P(60+) fell 0.307, 0.214, 0.135 across GW1-3 with no footballing reason playing time should
    fall the further out the horizon looks). ``by_gameweek`` must already be sorted by gameweek and
    carry a ``mean_p_60_plus`` column, matching
    :class:`~backtest.metrics.HorizonMonotonicityReport`'s own shape."""
    if len(by_gameweek) < 2:
        return False
    diffs = by_gameweek["mean_p_60_plus"].diff().dropna()
    return bool(len(diffs) > 0 and (diffs < -tolerance).all())
