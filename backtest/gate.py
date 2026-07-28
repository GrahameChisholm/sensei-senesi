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

from dataclasses import dataclass
from typing import Mapping

from backtest.baselines import PairedBootstrapResult, PermutationTestResult
from backtest.metrics import BiasReport, CalibrationReport

# ENGINE_IMPROVEMENTS_3.md B.1: the original 0.15 threshold was never derived from anything —
# every component ever measured has scored between 0.004 and 0.027, six to forty times tighter,
# so the gate could not fail on calibration even at the pre-B.1 defensive-contribution defect
# (0.0306, a 34% under-prediction). Set from the measured distribution with headroom for a
# component that's merely okay rather than excellent, not from a guess.
DEFAULT_MAX_CALIBRATION_ERROR = 0.05

__all__ = ["DefinitionOfDoneReport", "evaluate_definition_of_done"]

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

    @property
    def passed(self) -> bool:
        return (
            self.beats_baselines
            and self.no_severe_bias
            and self.calibration_acceptable
            and self.predictions_logged
            and self.trusted_by_user
        )

    def summary(self) -> str:
        checklist = {
            "Beats all baselines, statistically (3.3)": self.beats_baselines,
            "No severe systematic bias in any position/price tier (3.2)": self.no_severe_bias,
            "Each component reasonably calibrated (3.2)": self.calibration_acceptable,
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
    return DefinitionOfDoneReport(
        beats_baselines=beats_baselines,
        baseline_results=baseline_results,
        no_severe_bias=no_severe_bias,
        bias_reports=bias_reports,
        calibration_acceptable=calibration_acceptable,
        calibration_reports=calibration_reports,
        predictions_logged=predictions_logged,
        trusted_by_user=trusted_by_user,
    )
