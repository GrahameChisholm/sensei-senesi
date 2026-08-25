"""Tests for the Phase 3.6 Engine Definition-of-Done gate."""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.baselines import PairedBootstrapResult, PermutationTestResult
from backtest.gate import REAL_2025_26_MINUTES_BUCKET_SHARES, evaluate_definition_of_done
from backtest.metrics import (
    BiasReport,
    CalibrationReport,
    DecisionSetRankReport,
    MeanCalibrationReport,
    fixture_bonus_total,
    fixture_minutes_coverage,
    goalkeeper_saves_plausibility,
    horizon_minutes_monotonicity,
    minutes_bucket_pool_shares,
)


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


# T-J (planning/ENGINE_AUDIT_FIXES-implementation.md): aggregate, per-fixture and pool-wide
# plausibility invariants no per-player metric above can see. Every scenario below is synthetic,
# built to clearly pass or clearly fail rather than reading the real season cache.


def _evaluate_with_defaults(**t_j_kwargs):
    return evaluate_definition_of_done(
        baseline_results=_passing_baseline_results(),
        bias_reports=_unbiased_reports(),
        calibration_reports=_well_calibrated_reports(),
        predictions_logged=True,
        trusted_by_user=True,
        **t_j_kwargs,
    )


def test_fixture_minutes_coverage_report_computes_gap_per_fixture():
    predictions = pd.DataFrame(
        {
            "fixture_id": ["A"] * 22 + ["B"] * 22,
            # fixture A sums to exactly the 22-player target; fixture B mirrors the audit's
            # measured 18.2 shortfall.
            "p_60_plus": [1.0] * 22 + [0.827] * 22,
        }
    )
    report = fixture_minutes_coverage(predictions)
    by_fixture = report.by_fixture.set_index("fixture_id")

    assert by_fixture.loc["A", "sum_p_60_plus"] == 22.0
    assert by_fixture.loc["A", "gap"] == 0.0
    assert by_fixture.loc["B", "sum_p_60_plus"] == pytest.approx(18.194, abs=0.01)
    assert report.mean_absolute_gap > 1.5


def test_gate_fails_on_fixture_minutes_coverage_shortfall():
    predictions = pd.DataFrame({"fixture_id": ["A"] * 22, "p_60_plus": [0.827] * 22})
    report = fixture_minutes_coverage(predictions)

    result = _evaluate_with_defaults(fixture_minutes_coverage_report=report)

    assert not result.passed
    assert not result.fixture_minutes_coverage_acceptable


def test_gate_passes_on_fixture_minutes_coverage_near_target():
    predictions = pd.DataFrame({"fixture_id": ["A"] * 22, "p_60_plus": [1.0] * 22})
    report = fixture_minutes_coverage(predictions)

    result = _evaluate_with_defaults(fixture_minutes_coverage_report=report)

    assert result.fixture_minutes_coverage_acceptable
    assert result.passed


def test_fixture_bonus_total_report_computes_gap_per_fixture():
    predictions = pd.DataFrame({"fixture_id": ["A"] * 22, "bonus": [6.0 / 22] * 22})
    report = fixture_bonus_total(predictions)

    assert report.by_fixture["sum_bonus"].iloc[0] == pytest.approx(6.0)
    assert report.mean_absolute_gap == pytest.approx(0.0, abs=1e-9)


def test_gate_fails_on_fixture_bonus_total_mismatch():
    # A fixture whose total expected bonus is nowhere near the 6.0 points a real match awards.
    predictions = pd.DataFrame({"fixture_id": ["A"] * 22, "bonus": [3.0 / 22] * 22})
    report = fixture_bonus_total(predictions)

    result = _evaluate_with_defaults(fixture_bonus_total_report=report)

    assert not result.passed
    assert not result.fixture_bonus_total_acceptable


def test_gate_passes_on_fixture_bonus_total_near_target():
    predictions = pd.DataFrame({"fixture_id": ["A"] * 22, "bonus": [6.0 / 22] * 22})
    report = fixture_bonus_total(predictions)

    result = _evaluate_with_defaults(fixture_bonus_total_report=report)

    assert result.fixture_bonus_total_acceptable
    assert result.passed


def test_minutes_bucket_pool_shares_flags_the_audit_measured_shortfall():
    n = 1000
    # Mirrors the audit's measured pool-wide shares: p_1_to_59 roughly 2.00x the real 0.124.
    predictions = pd.DataFrame(
        {"p_zero": [0.629] * n, "p_1_to_59": [0.247] * n, "p_60_plus": [0.124] * n}
    )
    report = minutes_bucket_pool_shares(predictions, REAL_2025_26_MINUTES_BUCKET_SHARES)

    assert report.absolute_gaps["p_1_to_59"] > 0.1


def test_gate_fails_on_minutes_bucket_share_mismatch():
    n = 1000
    predictions = pd.DataFrame(
        {"p_zero": [0.629] * n, "p_1_to_59": [0.247] * n, "p_60_plus": [0.124] * n}
    )
    report = minutes_bucket_pool_shares(predictions, REAL_2025_26_MINUTES_BUCKET_SHARES)

    result = _evaluate_with_defaults(minutes_bucket_share_report=report)

    assert not result.passed
    assert not result.minutes_bucket_shares_acceptable


def test_gate_passes_on_minutes_bucket_share_matching_empirical():
    n = 1000
    predictions = pd.DataFrame(
        {
            "p_zero": [REAL_2025_26_MINUTES_BUCKET_SHARES["p_zero"]] * n,
            "p_1_to_59": [REAL_2025_26_MINUTES_BUCKET_SHARES["p_1_to_59"]] * n,
            "p_60_plus": [REAL_2025_26_MINUTES_BUCKET_SHARES["p_60_plus"]] * n,
        }
    )
    report = minutes_bucket_pool_shares(predictions, REAL_2025_26_MINUTES_BUCKET_SHARES)

    result = _evaluate_with_defaults(minutes_bucket_share_report=report)

    assert result.minutes_bucket_shares_acceptable
    assert result.passed


def test_goalkeeper_saves_plausibility_flags_understatement():
    predictions = pd.DataFrame(
        {
            "position": ["GK"] * 5 + ["DEF"] * 5,
            "p_60_plus": [0.9] * 5 + [0.9] * 5,
            # 1.63 is the audit's measured saves-per-90 figure against a real 2.78.
            "expected_saves": [1.63] * 5 + [0.0] * 5,
        }
    )
    report = goalkeeper_saves_plausibility(predictions, empirical_saves_per_match=2.78)

    assert report.n_players == 5
    assert report.relative_gap == pytest.approx((2.78 - 1.63) / 2.78, abs=1e-6)


def test_goalkeeper_saves_plausibility_excludes_rows_below_threshold():
    predictions = pd.DataFrame(
        {
            "position": ["GK", "GK"],
            "p_60_plus": [0.9, 0.1],
            "expected_saves": [2.78, 0.2],
        }
    )
    report = goalkeeper_saves_plausibility(predictions, empirical_saves_per_match=2.78)

    assert report.n_players == 1
    assert report.mean_predicted_saves == pytest.approx(2.78)


def test_goalkeeper_saves_plausibility_raises_when_no_qualifying_rows():
    predictions = pd.DataFrame(
        {"position": ["DEF", "MID"], "p_60_plus": [0.9, 0.9], "expected_saves": [0.0, 0.0]}
    )
    with pytest.raises(ValueError):
        goalkeeper_saves_plausibility(predictions, empirical_saves_per_match=2.78)


def test_gate_fails_on_goalkeeper_saves_understatement():
    predictions = pd.DataFrame(
        {"position": ["GK"] * 5, "p_60_plus": [0.9] * 5, "expected_saves": [1.63] * 5}
    )
    report = goalkeeper_saves_plausibility(predictions, empirical_saves_per_match=2.78)

    result = _evaluate_with_defaults(goalkeeper_saves_report=report)

    assert not result.passed
    assert not result.goalkeeper_saves_acceptable


def test_gate_passes_on_goalkeeper_saves_near_empirical():
    predictions = pd.DataFrame(
        {"position": ["GK"] * 5, "p_60_plus": [0.9] * 5, "expected_saves": [2.78] * 5}
    )
    report = goalkeeper_saves_plausibility(predictions, empirical_saves_per_match=2.78)

    result = _evaluate_with_defaults(goalkeeper_saves_report=report)

    assert result.goalkeeper_saves_acceptable
    assert result.passed


def test_horizon_minutes_monotonicity_report_is_sorted_and_raw():
    predictions = pd.DataFrame(
        {
            "gameweek": [3] * 10 + [1] * 10 + [2] * 10,
            # Mirrors the audit's measured GW1/GW2/GW3 decay: 0.307, 0.214, 0.135.
            "p_60_plus": [0.135] * 10 + [0.307] * 10 + [0.214] * 10,
        }
    )
    report = horizon_minutes_monotonicity(predictions)

    assert list(report.by_gameweek["gameweek"]) == [1, 2, 3]
    means = report.by_gameweek["mean_p_60_plus"].to_numpy()
    assert means[0] == pytest.approx(0.307)
    assert means[1] == pytest.approx(0.214)
    assert means[2] == pytest.approx(0.135)


def test_gate_fails_on_horizon_minutes_decay():
    predictions = pd.DataFrame(
        {
            "gameweek": [1] * 10 + [2] * 10 + [3] * 10,
            "p_60_plus": [0.307] * 10 + [0.214] * 10 + [0.135] * 10,
        }
    )
    report = horizon_minutes_monotonicity(predictions)

    result = _evaluate_with_defaults(horizon_minutes_report=report)

    assert not result.passed
    assert not result.horizon_minutes_non_decaying


def test_gate_passes_on_flat_noisy_horizon_minutes():
    # Ordinary week-to-week noise, not a systematic slide: GW2 dips slightly then GW3 recovers.
    predictions = pd.DataFrame(
        {
            "gameweek": [1] * 10 + [2] * 10 + [3] * 10,
            "p_60_plus": [0.30] * 10 + [0.295] * 10 + [0.31] * 10,
        }
    )
    report = horizon_minutes_monotonicity(predictions)

    result = _evaluate_with_defaults(horizon_minutes_report=report)

    assert result.horizon_minutes_non_decaying
    assert result.passed


def test_gate_omits_t_j_checks_when_not_supplied():
    # No T-J report supplied at all, so every T-J field stays a no-op, matching the existing
    # mean_calibration_reports convention: a caller that hasn't wired these up yet is
    # unaffected.
    result = _evaluate_with_defaults()

    assert result.fixture_minutes_coverage_acceptable
    assert result.fixture_bonus_total_acceptable
    assert result.minutes_bucket_shares_acceptable
    assert result.goalkeeper_saves_acceptable
    assert result.horizon_minutes_non_decaying
    assert result.passed


# =================================================================================================
# ENGINE_IMPROVEMENTS_5.md Tier 0.1 — shortlist ranking and conditional bias
# =================================================================================================


def _decision_set_report(mean_spearman: float) -> DecisionSetRankReport:
    return DecisionSetRankReport(
        top_n=20,
        mean_spearman=mean_spearman,
        median_spearman=mean_spearman,
        std_spearman=0.2,
        share_positive=0.6,
        mean_absolute_error=3.3,
        mean_bias=0.04,
        n_gameweeks=35,
        by_gameweek=pd.DataFrame(
            {"gameweek": [1], "spearman": [mean_spearman], "mae": [3.3], "bias": [0.04], "n": [20]}
        ),
    )


def _conditional_bias_report(severe: bool) -> dict[str, BiasReport]:
    return {
        "position_played_60_plus": BiasReport(
            by_group=pd.DataFrame(
                {
                    "position": ["MID"],
                    "mean_residual": [-0.99 if severe else -0.01],
                    "mean_actual": [4.0],
                    "std_residual": [3.1],
                    "n": [2964],
                    "t_stat": [-18.5 if severe else -0.2],
                    "p_value": [1e-72 if severe else 0.8],
                    "effect_size_floor": [0.4],
                    "severe": [severe],
                }
            )
        )
    }


def test_gate_fails_when_shortlist_ranking_is_no_better_than_chance():
    # The measured 0.0494 on the real 2025/26 walk-forward, against the 0.15 threshold.
    result = _evaluate_with_defaults(decision_set_rank_report=_decision_set_report(0.0494))

    assert not result.decision_set_ranking_acceptable
    assert not result.passed
    assert "Orders its own shortlist better than chance (5.0.1)" in result.summary()


def test_gate_passes_when_shortlist_ranking_clears_the_threshold():
    result = _evaluate_with_defaults(decision_set_rank_report=_decision_set_report(0.31))

    assert result.decision_set_ranking_acceptable
    assert result.passed


def test_gate_treats_an_unscorable_shortlist_report_as_a_no_op():
    # No gameweek could be scored (every actual tied), so the criterion is unjudgeable rather than
    # failing, matching how mean_calibration treats a NaN relative_gap.
    result = _evaluate_with_defaults(decision_set_rank_report=_decision_set_report(float("nan")))

    assert result.decision_set_ranking_acceptable
    assert result.passed


def test_gate_fails_on_severe_conditional_bias_while_unconditional_bias_passes():
    """The whole point of the criterion: `bias_reports` is the unbiased set from
    `_unbiased_reports()` and still passes, while the played-60+ reports fail. Before this existed
    the gate read only the former and could not see a -0.990 bias on every player a manager
    fields."""
    result = _evaluate_with_defaults(conditional_bias_reports=_conditional_bias_report(severe=True))

    assert result.no_severe_bias
    assert not result.no_severe_conditional_bias
    assert not result.passed


def test_gate_passes_when_conditional_bias_is_not_severe():
    result = _evaluate_with_defaults(
        conditional_bias_reports=_conditional_bias_report(severe=False)
    )

    assert result.no_severe_conditional_bias
    assert result.passed


def test_gate_omits_tier_0_1_checks_when_not_supplied():
    result = _evaluate_with_defaults()

    assert result.decision_set_ranking_acceptable
    assert result.no_severe_conditional_bias
    assert result.decision_set_rank_report is None
    assert result.conditional_bias_reports == {}
    assert result.passed
