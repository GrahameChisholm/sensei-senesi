"""Tests for market_overlay/divergence.py — engine-vs-market probability comparison (BUILD_PLAN
4b.2)."""

from __future__ import annotations

import pytest

from market_overlay.divergence import (
    LARGE_DIVERGENCE_THRESHOLD,
    SMALL_DIVERGENCE_THRESHOLD,
    compare_probabilities,
    implied_probability,
    remove_overround,
)


def test_implied_probability_matches_one_over_odds():
    assert implied_probability(2.0) == pytest.approx(0.5)


def test_implied_probability_rejects_odds_at_or_below_one():
    with pytest.raises(ValueError):
        implied_probability(1.0)
    with pytest.raises(ValueError):
        implied_probability(0.5)


def test_remove_overround_normalizes_to_one():
    # Raw implied probabilities: 1/1.8 + 1/3.6 + 1/4.5 = 0.556 + 0.278 + 0.222 = 1.056 (overround)
    probs = remove_overround([1.8, 3.6, 4.5])
    assert sum(probs) == pytest.approx(1.0)


def test_remove_overround_preserves_relative_ordering():
    probs = remove_overround([1.8, 3.6, 4.5])
    assert probs[0] > probs[1] > probs[2]  # shortest odds -> highest probability


def test_remove_overround_rejects_empty_input():
    with pytest.raises(ValueError):
        remove_overround([])


def test_compare_probabilities_rejects_out_of_range_inputs():
    with pytest.raises(ValueError):
        compare_probabilities("x", 1.5, 0.5)
    with pytest.raises(ValueError):
        compare_probabilities("x", 0.5, -0.1)


def test_compare_probabilities_none_within_small_threshold():
    flag = compare_probabilities(
        "Haaland anytime scorer", 0.50, 0.50 + SMALL_DIVERGENCE_THRESHOLD / 2
    )
    assert flag.severity == "none"
    assert flag.gap == pytest.approx(-SMALL_DIVERGENCE_THRESHOLD / 2)


def test_compare_probabilities_small_between_thresholds():
    midpoint = (SMALL_DIVERGENCE_THRESHOLD + LARGE_DIVERGENCE_THRESHOLD) / 2
    flag = compare_probabilities("x", 0.5 + midpoint, 0.5)
    assert flag.severity == "small"


def test_compare_probabilities_large_at_or_above_threshold():
    flag = compare_probabilities("x", 0.5 + LARGE_DIVERGENCE_THRESHOLD, 0.5)
    assert flag.severity == "large"


def test_compare_probabilities_gap_sign_reflects_direction():
    bullish = compare_probabilities("x", 0.7, 0.5)
    bearish = compare_probabilities("x", 0.3, 0.5)
    assert bullish.gap > 0
    assert bearish.gap < 0


def test_compare_probabilities_never_mutates_engine_probability():
    flag = compare_probabilities("x", 0.42, 0.10)
    assert flag.engine_probability == 0.42
