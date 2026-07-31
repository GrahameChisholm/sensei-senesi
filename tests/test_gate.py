"""Tests for the Phase 3.6 Engine Definition-of-Done gate."""

from __future__ import annotations

import pandas as pd

from backtest.baselines import PairedBootstrapResult, PermutationTestResult
from backtest.gate import evaluate_definition_of_done
from backtest.metrics import BiasReport, CalibrationReport, MeanCalibrationReport


def _passing_baseline_results() -> dict:
    return {
        "naive_form": PairedBootstrapResult(
            mean_diff=-1.0, ci_low=-1.5, ci_high=-0.5, n_bootstrap=1000, beats_baseline=True
        ),
        "template_captain": PermutationTestResult(
            observed_diff=0.2, p_value=0.01, n_permutations=1000, beats_baseline=True
        ),
    }


def _unbiased_reports() -> dict:
    return {
        "position": BiasReport(
            by_group=pd.DataFrame(
                {"position": ["MID", "FWD"], "severe": [False, False], "mean_residual": [0.1, -0.1]}
            )
        )
    }


def _well_calibrated_reports() -> dict:
    return {
        "clean_sheet": CalibrationReport(
            by_bin=pd.DataFrame(
                {"bin": ["a"], "predicted_mean": [0.4], "actual_rate": [0.41], "n": [100]}
            ),
            mean_absolute_calibration_error=0.02,
        )
    }


def test_gate_passes_when_every_criterion_holds():
    report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
    )

    assert report.passed
    assert report.beats_baselines
    assert report.no_severe_bias
    assert report.calibration_acceptable
    assert "PASSED" in report.summary()


def test_gate_fails_if_any_baseline_not_beaten():
    baseline_results = _passing_baseline_results()
    baseline_results["template_captain"] = PermutationTestResult(
        observed_diff=0.0, p_value=0.9, n_permutations=1000, beats_baseline=False
    )

    report = evaluate_definition_of_done(
        baseline_results=baseline_results,
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
    )

    assert not report.passed
    assert not report.beats_baselines


def test_gate_fails_on_severe_bias():
    bias_reports = {
        "position": BiasReport(
            by_group=pd.DataFrame({"position": ["FWD"], "severe": [True], "mean_residual": [3.0]})
        )
    }

    report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=bias_reports,
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
    )

    assert not report.passed
    assert not report.no_severe_bias


def test_gate_fails_on_poor_calibration():
    calibration_reports = {
        "clean_sheet": CalibrationReport(
            by_bin=pd.DataFrame(
                {"bin": ["a"], "predicted_mean": [0.8], "actual_rate": [0.3], "n": [100]}
            ),
            mean_absolute_calibration_error=0.5,
        )
    }

    report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=calibration_reports,
        predictions_logged=True,
        trusted_by_user=True,
    )

    assert not report.passed
    assert not report.calibration_acceptable


def test_gate_fails_if_predictions_not_logged_or_not_trusted():
    report_not_logged = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=False,
        trusted_by_user=True,
    )
    assert not report_not_logged.passed

    report_not_trusted = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=False,
    )
    assert not report_not_trusted.passed
    assert "NOT YET" in report_not_trusted.summary()


def test_gate_uses_per_component_calibration_threshold_when_given():
    # ENGINE_IMPROVEMENTS_3.md B.1: a component whose own historical value is looser than the
    # global default should still be checked against its own bar, not silently pass or fail
    # against a one-size-fits-all number.
    calibration_reports = {
        "clean_sheet": CalibrationReport(
            by_bin=pd.DataFrame(
                {"bin": ["a"], "predicted_mean": [0.4], "actual_rate": [0.47], "n": [100]}
            ),
            mean_absolute_calibration_error=0.07,
        )
    }

    default_report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=calibration_reports,
        predictions_logged=True,
        trusted_by_user=True,
    )
    assert not default_report.calibration_acceptable  # 0.07 > the tightened 0.05 default

    per_component_report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=calibration_reports,
        predictions_logged=True,
        trusted_by_user=True,
        calibration_error_thresholds={"clean_sheet": 0.1},
    )
    assert per_component_report.calibration_acceptable


def test_gate_omits_mean_calibration_check_when_not_supplied():
    # B3: a caller that doesn't compute the played-only mean-calibration reports (e.g. no minutes
    # column available) must not have the gate retroactively fail on their absence.
    report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
    )

    assert report.mean_calibration_acceptable
    assert report.passed


def test_gate_fails_on_poor_mean_calibration():
    # B3: goals over-predicting by 24.9% is the all-rows figure dominated by the minutes model;
    # a played-only relative_gap this large is a real component defect and must fail the gate.
    mean_calibration_reports = {
        "goals": MeanCalibrationReport(
            mean_predicted=0.10, mean_actual=0.08, absolute_gap=0.02, relative_gap=0.25
        )
    }

    report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
        mean_calibration_reports=mean_calibration_reports,
    )

    assert not report.passed
    assert not report.mean_calibration_acceptable


def test_gate_uses_per_component_mean_calibration_threshold_when_given():
    mean_calibration_reports = {
        "bonus": MeanCalibrationReport(
            mean_predicted=0.11, mean_actual=0.10, absolute_gap=0.01, relative_gap=0.08
        )
    }

    default_report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
        mean_calibration_reports=mean_calibration_reports,
    )
    assert not default_report.mean_calibration_acceptable  # 0.08 > the 0.05 default

    per_component_report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
        mean_calibration_reports=mean_calibration_reports,
        mean_calibration_relative_gap_thresholds={"bonus": 0.1},
    )
    assert per_component_report.mean_calibration_acceptable


def test_gate_ignores_a_nan_relative_gap_rather_than_failing_unjudgeably():
    # mean_actual == 0 makes relative_gap NaN -- not a defect, just an unjudgeable ratio.
    mean_calibration_reports = {
        "goals": MeanCalibrationReport(
            mean_predicted=0.05, mean_actual=0.0, absolute_gap=0.05, relative_gap=float("nan")
        )
    }

    report = evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
        mean_calibration_reports=mean_calibration_reports,
    )

    assert report.mean_calibration_acceptable


def test_gate_fails_on_empty_inputs_rather_than_vacuously_passing():
    report = evaluate_definition_of_done(
        baseline_results={},
        bias_reports={},
        calibration_reports={},
        predictions_logged=True,
        trusted_by_user=True,
    )

    assert not report.passed
    assert not report.beats_baselines
    assert not report.no_severe_bias
    assert not report.calibration_acceptable
