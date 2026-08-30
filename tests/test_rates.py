"""Tests for engine/rates.py — the shared EWMA per-90 rate utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.rates import (
    EwmaRateConfig,
    effective_sample_minutes,
    ewma_rate_asof,
    fit_rate_ratio_prior,
    latest_ewma_rate,
    league_average_rate,
    rate_ratio_posterior,
    shrink_toward_prior,
)


def _matches(stat_values: list[float], minutes_values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"npxG": stat_values, "time": minutes_values})


def test_ewma_rate_asof_first_row_is_nan():
    matches = _matches([0.5, 0.3, 0.4], [90, 90, 90])
    result = ewma_rate_asof(matches, "npxG")
    assert np.isnan(result.iloc[0])


def test_ewma_rate_asof_uses_only_prior_rows_no_leakage():
    matches = _matches([0.1, 0.1, 0.1, 99.0], [90, 90, 90, 90])
    result = ewma_rate_asof(matches, "npxG")
    # The huge final-row value must not leak into any earlier row's "as of" rate.
    assert (result.iloc[1:3] < 1.0).all()


def test_ewma_rate_asof_constant_rate_converges_to_that_rate():
    # A long, constant per-90 rate history should converge (as-of, i.e. excluding the row itself)
    # to that same rate regardless of the exact match count/minutes pattern.
    matches = _matches([0.4] * 50, [90] * 50)
    result = ewma_rate_asof(matches, "npxG")
    assert result.iloc[-1] == pytest.approx(0.4, abs=1e-6)


def test_ewma_rate_asof_low_minutes_cameo_does_not_dominate():
    # A 5-minute cameo with a fluke goal (per-match rate = 18/90) should barely move the rate
    # relative to a long history of a much lower true rate, because minutes are weighted too.
    matches = _matches([0.3] * 20 + [1.0], [90] * 20 + [5])
    result = latest_ewma_rate(matches, "npxG")
    assert result < 0.5


def test_latest_ewma_rate_empty_history_is_nan():
    matches = _matches([], [])
    assert np.isnan(latest_ewma_rate(matches, "npxG"))


def test_latest_ewma_rate_zero_minutes_history_is_nan():
    matches = _matches([0.0, 0.0], [0, 0])
    assert np.isnan(latest_ewma_rate(matches, "npxG"))


def test_ewma_rate_asof_respects_custom_halflife():
    matches = _matches([1.0, 0.0, 0.0, 0.0, 0.0], [90] * 5)
    short = ewma_rate_asof(matches, "npxG", config=EwmaRateConfig(halflife_matches=1.0))
    long = ewma_rate_asof(matches, "npxG", config=EwmaRateConfig(halflife_matches=20.0))
    # A short halflife should have forgotten the early spike faster than a long one.
    assert short.iloc[-1] < long.iloc[-1]


def test_latest_ewma_rate_all_cameo_history_without_cap_is_physically_implausible():
    # A player whose entire sample is short cameos, one of which had a fluke high-xG kick (0.68 xG
    # in 1 minute = 61.2 per-90), has no long "normal" history to dilute it against, unlike the
    # low_minutes_cameo test above. Uncapped, the resulting rate is implausibly high for any real
    # striker, the exact failure mode a real Understat pull surfaced.
    matches = _matches([0.0, 0.0, 0.0, 0.68], [15, 10, 2, 1])
    result = latest_ewma_rate(matches, "npxG")
    assert result > 2.0


def test_latest_ewma_rate_max_rate_per_90_caps_all_cameo_history():
    # Same all-cameo history as above, but with a cap: the fluke 1-minute kick can contribute at
    # most `max_rate_per_90` worth of evidence, not the raw 61.2/90 the match literally implies.
    matches = _matches([0.0, 0.0, 0.0, 0.68], [15, 10, 2, 1])
    result = latest_ewma_rate(matches, "npxG", max_rate_per_90=2.5)
    assert result < 2.5


def test_ewma_rate_asof_max_rate_per_90_caps_a_fluke_cameo_row():
    matches = _matches([0.0, 0.0, 0.68, 0.1], [15, 10, 1, 90])
    uncapped = ewma_rate_asof(matches, "npxG")
    capped = ewma_rate_asof(matches, "npxG", max_rate_per_90=2.5)
    assert capped.iloc[-1] < uncapped.iloc[-1]


def test_ewma_rate_asof_max_rate_per_90_first_row_still_nan():
    matches = _matches([0.68, 0.1], [1, 90])
    result = ewma_rate_asof(matches, "npxG", max_rate_per_90=2.5)
    assert np.isnan(result.iloc[0])


def test_latest_ewma_rate_max_rate_per_90_none_matches_uncapped():
    matches = _matches([0.3] * 20 + [1.0], [90] * 20 + [5])
    assert latest_ewma_rate(matches, "npxG", max_rate_per_90=None) == latest_ewma_rate(
        matches, "npxG"
    )


def test_latest_ewma_rate_rejects_non_positive_max_rate_per_90():
    matches = _matches([0.3], [90])
    with pytest.raises(ValueError):
        latest_ewma_rate(matches, "npxG", max_rate_per_90=0.0)


def test_effective_sample_minutes_grows_with_more_matches():
    thin = _matches([0.3], [90])
    thick = _matches([0.3] * 30, [90] * 30)
    assert effective_sample_minutes(thin) < effective_sample_minutes(thick)


def test_effective_sample_minutes_empty_is_zero():
    assert effective_sample_minutes(_matches([], [])) == 0.0


def test_shrink_toward_prior_no_individual_weight_returns_prior():
    assert shrink_toward_prior(0.9, 0.0, 0.2, shrinkage_k=10.0) == 0.2


def test_shrink_toward_prior_nan_individual_returns_prior():
    assert shrink_toward_prior(float("nan"), 50.0, 0.2, shrinkage_k=10.0) == 0.2


def test_shrink_toward_prior_large_weight_stays_close_to_individual():
    shrunk = shrink_toward_prior(0.9, 10_000.0, 0.2, shrinkage_k=10.0)
    assert shrunk == pytest.approx(0.9, abs=1e-3)


def test_shrink_toward_prior_small_weight_moves_toward_prior():
    shrunk = shrink_toward_prior(0.9, 1.0, 0.2, shrinkage_k=10.0)
    assert 0.2 < shrunk < 0.9
    assert shrunk == pytest.approx((1.0 * 0.9 + 10.0 * 0.2) / 11.0)


def test_shrink_toward_prior_rejects_negative_k():
    with pytest.raises(ValueError):
        shrink_toward_prior(0.9, 10.0, 0.2, shrinkage_k=-1.0)


def test_shrink_toward_prior_nan_weight_returns_prior():
    # ENGINE_IMPROVEMENTS_2.md C.2: a NaN weight must fall back to the prior, not silently
    # NaN-poison the blended result.
    assert shrink_toward_prior(0.9, float("nan"), 0.2, shrinkage_k=10.0) == 0.2


def test_league_average_rate_computes_mean():
    assert league_average_rate({"A": 1.0, "B": 2.0, "C": 3.0}) == pytest.approx(2.0)


def test_league_average_rate_rejects_empty():
    with pytest.raises(ValueError):
        league_average_rate({})


# --- Gamma-Poisson rate ratios ------------------------------------------------------------------


@pytest.mark.parametrize("true_k", [2.0, 5.0, 20.0])
def test_fit_rate_ratio_prior_recovers_a_known_k(true_k):
    """The estimator's core claim: given data actually generated by a gamma-Poisson process with
    a known prior strength, method of moments recovers it. Without this the fitted k is just an
    unvalidated number driving every ratio on the page."""
    rng = np.random.default_rng(7)
    exposures = rng.uniform(0.5, 12.0, 4000)
    theta = rng.gamma(true_k, 1.0 / true_k, 4000)
    actuals = rng.poisson(exposures * theta)

    assert fit_rate_ratio_prior(actuals, exposures) == pytest.approx(true_k, rel=0.15)


def test_fit_rate_ratio_prior_reports_no_detectable_heterogeneity():
    """Data generated with theta == 1 for everyone has no between-player skill to find, so the
    spread is pure Poisson noise. Returning inf (every ratio collapses to 1.0) is the correct
    and honest answer, not a degenerate failure."""
    rng = np.random.default_rng(11)
    exposures = rng.uniform(0.5, 12.0, 4000)
    actuals = rng.poisson(exposures)

    assert fit_rate_ratio_prior(actuals, exposures) == float("inf")


def test_fit_rate_ratio_prior_ignores_zero_exposure_players():
    # A player with no expected involvements carries no information about the population's spread.
    assert fit_rate_ratio_prior([0.0, 0.0], [0.0, 0.0]) == float("inf")


def test_fit_rate_ratio_prior_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        fit_rate_ratio_prior([1.0, 2.0], [1.0])


def test_rate_ratio_posterior_matches_the_closed_form_mean():
    result = rate_ratio_posterior(actual=12.0, exposure=6.0, k=5.0)
    assert result is not None
    assert result.ratio == pytest.approx((12.0 + 5.0) / (6.0 + 5.0))
    assert result.low < result.ratio < result.high


def test_rate_ratio_posterior_shrinks_a_thin_sample_fluke_toward_one():
    """The case that motivated using a shrunk ratio over a raw per-90 difference: one goal from
    0.08 xGI is a +3.3 raw per-90 gap and would top any sort. It must land near 1.0 with an
    interval that still contains 1.0, i.e. "not distinguishable from chance"."""
    fluke = rate_ratio_posterior(actual=1.0, exposure=0.08, k=5.0)
    assert fluke is not None
    assert fluke.ratio < 1.25
    assert fluke.low < 1.0 < fluke.high
    assert not fluke.is_hot


def test_rate_ratio_posterior_interval_narrows_as_exposure_grows():
    """Same observed ratio (2x expected), more evidence behind it: the point estimate must move
    away from the prior and the interval must tighten, which is what makes the column safe to
    sort on."""
    thin = rate_ratio_posterior(actual=2.0, exposure=1.0, k=5.0)
    thick = rate_ratio_posterior(actual=40.0, exposure=20.0, k=5.0)
    assert thin is not None and thick is not None

    assert (thick.high - thick.low) < (thin.high - thin.low)
    assert thick.ratio > thin.ratio
    assert thick.is_hot and not thin.is_hot


def test_rate_ratio_posterior_flags_sustained_underperformance_as_cold():
    cold = rate_ratio_posterior(actual=2.0, exposure=20.0, k=5.0)
    assert cold is not None
    assert cold.is_cold and not cold.is_hot


def test_rate_ratio_posterior_returns_none_without_exposure():
    # "No expected involvements at all" is a real state, never a ratio of zero.
    assert rate_ratio_posterior(actual=0.0, exposure=0.0, k=5.0) is None


def test_rate_ratio_posterior_degenerates_when_no_heterogeneity_was_detected():
    # An infinite k asserts there is no real spread, so every player sits exactly at expectation.
    result = rate_ratio_posterior(actual=12.0, exposure=6.0, k=float("inf"))
    assert result is not None
    assert (result.ratio, result.low, result.high) == (1.0, 1.0, 1.0)
    assert not result.is_hot and not result.is_cold
