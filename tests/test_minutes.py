"""Tests for engine/models/minutes.py — the two-stage minutes model (2.1)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.minutes import (
    FEATURE_COLUMNS,
    MinutesDistribution,
    MinutesModel,
    encode_status,
)


def test_encode_status_known_codes():
    assert encode_status("a") == 1.0
    assert encode_status("i") == 0.0


def test_encode_status_unknown_raises():
    with pytest.raises(ValueError):
        encode_status("x")


def test_minutes_distribution_rejects_probabilities_not_summing_to_one():
    with pytest.raises(ValueError):
        MinutesDistribution(
            p_zero=0.5,
            p_1_to_59=0.5,
            p_60_plus=0.5,
            expected_minutes_given_1_to_59=30.0,
            expected_minutes_given_60_plus=80.0,
        )


def test_minutes_distribution_expected_minutes_blends_buckets():
    dist = MinutesDistribution(
        p_zero=0.2,
        p_1_to_59=0.3,
        p_60_plus=0.5,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )
    assert dist.expected_minutes == pytest.approx(0.3 * 30.0 + 0.5 * 90.0)


def _synthetic_training_data(
    n: int = 200, seed: int = 0
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    rng = np.random.default_rng(seed)
    recent_start_rate = rng.uniform(0, 1, n)
    recent_minutes_ewma = rng.uniform(0, 90, n)
    fixture_congestion = rng.integers(0, 4, n).astype(float)
    chance_of_playing = rng.choice([0, 25, 50, 75, 100], n).astype(float)
    status_score = rng.choice([0.0, 0.75, 1.0], n)

    features = pd.DataFrame(
        {
            "recent_start_rate": recent_start_rate,
            "recent_minutes_ewma": recent_minutes_ewma,
            "fixture_congestion": fixture_congestion,
            "chance_of_playing_next_round": chance_of_playing,
            "status_score": status_score,
        }
    )

    # A player with a high recent start rate and full fitness starts; low-fitness/low-usage
    # players don't. This gives the classifiers genuine, learnable signal.
    fitness_component = status_score * 0.4 + (chance_of_playing / 100) * 0.3
    start_propensity = recent_start_rate * 0.6 + fitness_component
    started = (start_propensity + rng.normal(0, 0.1, n) > 0.5).astype(int)

    minutes = np.zeros(n)
    for i in range(n):
        if started[i]:
            # Higher fitness -> more likely to play the full 90.
            plays_full_90 = rng.uniform(0, 1) < 0.2 + 0.6 * status_score[i]
            minutes[i] = 90 if plays_full_90 else rng.integers(15, 59)
        else:
            minutes[i] = rng.integers(1, 30) if rng.uniform(0, 1) < 0.3 else 0

    return features, pd.Series(started), pd.Series(minutes)


def test_minutes_model_fit_predict_end_to_end():
    features, started, minutes = _synthetic_training_data()
    model = MinutesModel().fit(features, started, minutes)
    results = model.predict(features)

    assert len(results) == len(features)
    for dist in results:
        assert isinstance(dist, MinutesDistribution)
        assert 0.0 <= dist.p_zero <= 1.0
        assert 0.0 <= dist.p_1_to_59 <= 1.0
        assert 0.0 <= dist.p_60_plus <= 1.0
        assert dist.expected_minutes_given_60_plus >= 60.0
        assert 0.0 <= dist.expected_minutes <= 90.0


def test_minutes_model_high_fitness_high_start_rate_favours_60_plus():
    features, started, minutes = _synthetic_training_data()
    model = MinutesModel().fit(features, started, minutes)

    nailed_on = pd.DataFrame(
        [
            {
                "recent_start_rate": 1.0,
                "recent_minutes_ewma": 90.0,
                "fixture_congestion": 0.0,
                "chance_of_playing_next_round": 100.0,
                "status_score": 1.0,
            }
        ]
    )
    fringe = pd.DataFrame(
        [
            {
                "recent_start_rate": 0.0,
                "recent_minutes_ewma": 5.0,
                "fixture_congestion": 3.0,
                "chance_of_playing_next_round": 0.0,
                "status_score": 0.0,
            }
        ]
    )

    nailed_dist = model.predict(nailed_on)[0]
    fringe_dist = model.predict(fringe)[0]

    assert nailed_dist.p_60_plus > fringe_dist.p_60_plus
    assert nailed_dist.expected_minutes > fringe_dist.expected_minutes


def test_minutes_model_predict_before_fit_raises():
    features = pd.DataFrame([{col: 0.0 for col in FEATURE_COLUMNS}])
    with pytest.raises(RuntimeError):
        MinutesModel().predict(features)


def test_minutes_model_handles_single_class_training_data():
    # Every row starts and plays 60+ -- degenerate, but must not crash (small early-season
    # samples can look like this in practice).
    n = 10
    features = pd.DataFrame(
        {
            "recent_start_rate": [1.0] * n,
            "recent_minutes_ewma": [90.0] * n,
            "fixture_congestion": [0.0] * n,
            "chance_of_playing_next_round": [100.0] * n,
            "status_score": [1.0] * n,
        }
    )
    started = pd.Series([1] * n)
    minutes = pd.Series([90.0] * n)

    model = MinutesModel().fit(features, started, minutes)
    results = model.predict(features)
    assert results[0].p_60_plus == pytest.approx(1.0, abs=1e-6)
    assert results[0].p_zero == pytest.approx(0.0, abs=1e-6)
