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
