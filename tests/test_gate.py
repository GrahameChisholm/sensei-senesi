"""Tests for the Phase 3.6 Engine Definition-of-Done gate."""

from __future__ import annotations

import pandas as pd

from backtest.baselines import PairedBootstrapResult, PermutationTestResult
from backtest.gate import evaluate_definition_of_done
from backtest.metrics import BiasReport, CalibrationReport


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
