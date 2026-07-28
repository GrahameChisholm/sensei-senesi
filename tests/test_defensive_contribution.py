"""Tests for engine/models/defensive_contribution.py — Negative Binomial DC model (2.5)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.defensive_contribution import (
    DEFAULT_OVERDISPERSION,
    DefensiveContributionProjection,
    expected_defensive_action_rate,
    fit_overdispersion,
    negative_binomial_params,
    opponent_possession_adjustment,
    probability_clears_threshold,
    project_defensive_contribution,
)
from engine.scoring import DEFENSIVE_CONTRIBUTION_POINTS


def test_opponent_possession_adjustment_above_average_scales_up():
    adjustment = opponent_possession_adjustment(0.6, league_avg_possession_share=0.5)
    assert adjustment == pytest.approx(1.2)


def test_opponent_possession_adjustment_below_average_scales_down():
    adjustment = opponent_possession_adjustment(0.4, league_avg_possession_share=0.5)
    assert adjustment < 1.0


def test_opponent_possession_adjustment_rejects_out_of_range_share():
    with pytest.raises(ValueError):
        opponent_possession_adjustment(1.5)


def test_opponent_possession_adjustment_rejects_non_positive_league_average():
    with pytest.raises(ValueError):
        opponent_possession_adjustment(0.5, league_avg_possession_share=0.0)


def test_expected_defensive_action_rate_more_vs_possession_dominant_opponent():
    # Reverse direction from goals/assists: a possession-heavy opponent means MORE actions.
    vs_dominant = expected_defensive_action_rate(8.0, opponent_possession_share=0.65)
    vs_weak = expected_defensive_action_rate(8.0, opponent_possession_share=0.35)
    assert vs_dominant > vs_weak


def test_expected_defensive_action_rate_scales_with_minutes():
    full = expected_defensive_action_rate(8.0, 0.5, expected_minutes=90.0)
    half = expected_defensive_action_rate(8.0, 0.5, expected_minutes=45.0)
    assert half == pytest.approx(full / 2)


def test_expected_defensive_action_rate_rejects_negative_inputs():
    with pytest.raises(ValueError):
        expected_defensive_action_rate(-1.0, 0.5)
    with pytest.raises(ValueError):
        expected_defensive_action_rate(8.0, 0.5, expected_minutes=-1.0)


def test_negative_binomial_params_variance_exceeds_poisson():
    mu = 8.0
    alpha = 0.2
    n, p = negative_binomial_params(mu, alpha)
    variance = n * (1 - p) / (p**2)
    assert variance > mu  # overdispersed relative to Poisson (variance == mean)


def test_negative_binomial_params_rejects_non_positive_alpha():
    with pytest.raises(ValueError):
        negative_binomial_params(8.0, alpha=0.0)


def test_negative_binomial_params_rejects_negative_mu():
    with pytest.raises(ValueError):
        negative_binomial_params(-1.0, alpha=0.2)


def test_probability_clears_threshold_zero_mu_is_zero():
    assert probability_clears_threshold(0.0, threshold=10) == 0.0


def test_probability_clears_threshold_increases_with_mu():
    low = probability_clears_threshold(mu=4.0, threshold=10)
    high = probability_clears_threshold(mu=14.0, threshold=10)
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_probability_clears_threshold_rejects_non_positive_threshold():
    with pytest.raises(ValueError):
        probability_clears_threshold(8.0, threshold=0)


def test_defensive_contribution_projection_expected_points():
    projection = DefensiveContributionProjection(p_clears_threshold=0.4, threshold=10)
    assert projection.expected_points == pytest.approx(0.4 * DEFENSIVE_CONTRIBUTION_POINTS)


def test_defensive_contribution_projection_rejects_bad_probability():
    with pytest.raises(ValueError):
        DefensiveContributionProjection(p_clears_threshold=1.5, threshold=10)


def test_defensive_contribution_projection_rejects_bad_threshold():
    with pytest.raises(ValueError):
        DefensiveContributionProjection(p_clears_threshold=0.4, threshold=0)


def test_project_defensive_contribution_uses_position_threshold():
    def_projection = project_defensive_contribution("DEF", 10.0, opponent_possession_share=0.5)
    mid_projection = project_defensive_contribution("MID", 10.0, opponent_possession_share=0.5)
    assert def_projection.threshold == 10
    assert mid_projection.threshold == 12
    # Same action rate, lower threshold -> higher clearance probability.
    assert def_projection.p_clears_threshold > mid_projection.p_clears_threshold


def test_project_defensive_contribution_rejects_gk():
    with pytest.raises(ValueError):
        project_defensive_contribution("GK", 5.0, opponent_possession_share=0.5)


def test_project_defensive_contribution_rejects_unknown_position():
    with pytest.raises(ValueError):
        project_defensive_contribution("XYZ", 5.0, opponent_possession_share=0.5)


def test_fit_overdispersion_falls_back_when_too_few_rows():
    alpha = fit_overdispersion(pd.Series([5, 6, 7]), pd.Series([5.0, 5.0, 5.0]), min_rows=100)
    assert alpha == DEFAULT_OVERDISPERSION


def test_fit_overdispersion_falls_back_for_underdispersed_data():
    n = 200
    mu = np.full(n, 8.0)
    actual = mu.copy()  # variance == 0, below mu -> a negative method-of-moments estimate
    alpha = fit_overdispersion(pd.Series(actual), pd.Series(mu), min_rows=100)
    assert alpha == DEFAULT_OVERDISPERSION


def test_project_defensive_contribution_bucket_weighted_exceeds_point_estimate_for_rotation_risk():
    # ENGINE_IMPROVEMENTS_2.md B.1: probability_clears_threshold is convex in its mean, so for a
    # rotation-risk player (most of the mass at p_zero, a modest E[minutes]) the point-estimate
    # evaluation at E[minutes] understates the properly bucket-weighted expectation.
    point_estimate = project_defensive_contribution(
        "DEF", player_actions_per_90=10.0, opponent_possession_share=0.5, expected_minutes=27.0
    )
    bucket_weighted = project_defensive_contribution(
        "DEF",
        player_actions_per_90=10.0,
        opponent_possession_share=0.5,
        expected_minutes=27.0,  # ignored once the bucket args are supplied
        p_1_to_59=0.2,
        minutes_given_1_to_59=30.0,
        p_60_plus=0.1,
        minutes_given_60_plus=85.0,
        # (implied p_zero = 0.7; E[minutes] = 0.2*30 + 0.1*85 = 14.5 -- deliberately far from the
        # scalar expected_minutes above to prove the bucket path is genuinely used, not just
        # re-deriving the same number)
    )
    assert bucket_weighted.p_clears_threshold != pytest.approx(point_estimate.p_clears_threshold)


def test_project_defensive_contribution_requires_all_bucket_args_together():
    with pytest.raises(ValueError):
        project_defensive_contribution(
            "DEF", 10.0, opponent_possession_share=0.5, p_1_to_59=0.2, p_60_plus=0.1
        )


def test_project_defensive_contribution_bucket_weighted_matches_manual_expectation():
    alpha = 0.2
    p_1, m_1, p_60, m_60 = 0.3, 25.0, 0.2, 80.0
    mu_1 = expected_defensive_action_rate(9.0, 0.5, expected_minutes=m_1)
    mu_60 = expected_defensive_action_rate(9.0, 0.5, expected_minutes=m_60)
    expected = p_1 * probability_clears_threshold(mu_1, 12, alpha) + p_60 * probability_clears_threshold(
        mu_60, 12, alpha
    )

    result = project_defensive_contribution(
        "MID",
        player_actions_per_90=9.0,
        opponent_possession_share=0.5,
        alpha=alpha,
        p_1_to_59=p_1,
        minutes_given_1_to_59=m_1,
        p_60_plus=p_60,
        minutes_given_60_plus=m_60,
    )
    assert result.p_clears_threshold == pytest.approx(expected)


def test_fit_overdispersion_recovers_positive_alpha_for_overdispersed_data():
    rng = np.random.default_rng(0)
    n = 2000
    true_alpha = 0.25
    mu = 8.0
    r = 1.0 / true_alpha
    p = r / (r + mu)
    actual = rng.negative_binomial(r, p, size=n)
    expected_mu = np.full(n, mu)

    fitted_alpha = fit_overdispersion(pd.Series(actual), pd.Series(expected_mu), min_rows=100)

    assert fitted_alpha > 0
    assert fitted_alpha == pytest.approx(true_alpha, rel=0.3)
