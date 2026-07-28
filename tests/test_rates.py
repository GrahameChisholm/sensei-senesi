"""Tests for engine/rates.py — the shared EWMA per-90 rate utility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.rates import (
    EwmaRateConfig,
    effective_sample_minutes,
    ewma_rate_asof,
    latest_ewma_rate,
    league_average_rate,
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
