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
    days_since_last_appearance = rng.integers(0, 30, n).astype(float)
    zero_minute_streak_length = rng.integers(0, 6, n).astype(float)
    start_rate_last_3 = rng.uniform(0, 1, n)
    start_rate_last_6 = rng.uniform(0, 1, n)
    start_rate_last_15 = rng.uniform(0, 1, n)
    team_rotation_propensity = rng.uniform(0, 1, n)
    price = rng.uniform(4.0, 14.0, n)
    ownership_log = rng.uniform(0, 5, n)
    transfers_out_share = rng.uniform(0, 0.1, n)
    transfers_balance_share = rng.uniform(-0.1, 0.1, n)
    is_goalkeeper = np.zeros(n)

    features = pd.DataFrame(
        {
            "recent_start_rate": recent_start_rate,
            "recent_minutes_ewma": recent_minutes_ewma,
            "fixture_congestion": fixture_congestion,
            "chance_of_playing_next_round": chance_of_playing,
            "status_score": status_score,
            "days_since_last_appearance": days_since_last_appearance,
            "zero_minute_streak_length": zero_minute_streak_length,
            "start_rate_last_3": start_rate_last_3,
            "start_rate_last_6": start_rate_last_6,
            "start_rate_last_15": start_rate_last_15,
            "team_rotation_propensity": team_rotation_propensity,
            "price": price,
            "ownership_log": ownership_log,
            "transfers_out_share": transfers_out_share,
            "transfers_balance_share": transfers_balance_share,
            "is_goalkeeper": is_goalkeeper,
        }
    )

    # A player with a high recent start rate and full fitness starts; low-fitness/low-usage
    # players don't, and a long zero-minute streak further suppresses starting. This gives the
    # classifiers genuine, learnable signal across the extended feature set too.
    fitness_component = status_score * 0.4 + (chance_of_playing / 100) * 0.3
    start_propensity = (
        recent_start_rate * 0.4
        + fitness_component
        + start_rate_last_3 * 0.2
        - (zero_minute_streak_length / 6) * 0.3
    )
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
                "days_since_last_appearance": 0.0,
                "zero_minute_streak_length": 0.0,
                "start_rate_last_3": 1.0,
                "start_rate_last_6": 1.0,
                "start_rate_last_15": 1.0,
                "team_rotation_propensity": 0.1,
                "price": 9.0,
                "ownership_log": 3.0,
                "transfers_out_share": 0.01,
                "transfers_balance_share": 0.0,
                "is_goalkeeper": 0.0,
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
                "days_since_last_appearance": 30.0,
                "zero_minute_streak_length": 5.0,
                "start_rate_last_3": 0.0,
                "start_rate_last_6": 0.0,
                "start_rate_last_15": 0.0,
                "team_rotation_propensity": 0.9,
                "price": 4.5,
                "ownership_log": 0.5,
                "transfers_out_share": 0.05,
                "transfers_balance_share": -0.05,
                "is_goalkeeper": 0.0,
            }
        ]
    )

    nailed_dist = model.predict(nailed_on)[0]
    fringe_dist = model.predict(fringe)[0]

    assert nailed_dist.p_60_plus > fringe_dist.p_60_plus
    assert nailed_dist.expected_minutes > fringe_dist.expected_minutes


def test_minutes_model_zero_minute_streak_suppresses_p_60_plus():
    # A player mid a long zero-minute streak is categorically different from one rested for a
    # single match (ENGINE_IMPROVEMENTS.md 1.1) -- holding every other feature fixed, a longer
    # streak should predict materially lower p_60_plus.
    features, started, minutes = _synthetic_training_data()
    model = MinutesModel().fit(features, started, minutes)

    base_row = {
        "recent_start_rate": 0.6,
        "recent_minutes_ewma": 60.0,
        "fixture_congestion": 1.0,
        "chance_of_playing_next_round": 75.0,
        "status_score": 0.75,
        "days_since_last_appearance": 7.0,
        "start_rate_last_3": 0.6,
        "start_rate_last_6": 0.6,
        "start_rate_last_15": 0.6,
        "team_rotation_propensity": 0.5,
        "price": 7.0,
        "ownership_log": 2.0,
        "transfers_out_share": 0.02,
        "transfers_balance_share": 0.0,
        "is_goalkeeper": 0.0,
    }
    rested_one_match = pd.DataFrame([{**base_row, "zero_minute_streak_length": 1.0}])
    long_streak = pd.DataFrame([{**base_row, "zero_minute_streak_length": 5.0}])

    rested_dist = model.predict(rested_one_match)[0]
    streak_dist = model.predict(long_streak)[0]

    assert rested_dist.p_60_plus > streak_dist.p_60_plus


def test_minutes_model_is_goalkeeper_feature_captures_distinct_gk_pattern():
    # Real multi-season evidence (ENGINE_IMPROVEMENTS_3.md Phase 3): goalkeepers have a
    # qualitatively more binary minutes pattern the other 15 features don't capture on their own --
    # nailed-on ever-present or entirely out of the squad, with far less of the mid-match
    # substitution/rotation variance that shapes those features for outfield players. Synthesize
    # that pattern directly (GK start reliably regardless of recent_start_rate; outfield players'
    # recent_start_rate is the real signal) and confirm the model actually uses the is_goalkeeper
    # flag to tell them apart at an identical recent_start_rate, not just ignoring the extra column.
    rng = np.random.default_rng(42)
    n = 400
    is_goalkeeper = rng.choice([0.0, 1.0], n)
    recent_start_rate = rng.uniform(0.4, 0.6, n)
    start_propensity = np.where(is_goalkeeper == 1.0, 0.95, recent_start_rate)
    started = (rng.uniform(0, 1, n) < start_propensity).astype(int)
    minutes = np.where(started == 1, 90.0, 0.0)

    features = pd.DataFrame(
        {
            "recent_start_rate": recent_start_rate,
            "recent_minutes_ewma": np.full(n, 70.0),
            "fixture_congestion": np.zeros(n),
            "chance_of_playing_next_round": np.full(n, 100.0),
            "status_score": np.ones(n),
            "days_since_last_appearance": np.full(n, 7.0),
            "zero_minute_streak_length": np.zeros(n),
            "start_rate_last_3": recent_start_rate,
            "start_rate_last_6": recent_start_rate,
            "start_rate_last_15": recent_start_rate,
            "team_rotation_propensity": np.full(n, 0.3),
            "price": np.full(n, 6.0),
            "ownership_log": np.full(n, 2.0),
            "transfers_out_share": np.full(n, 0.02),
            "transfers_balance_share": np.zeros(n),
            "is_goalkeeper": is_goalkeeper,
        }
    )
    model = MinutesModel().fit(features, pd.Series(started), pd.Series(minutes))

    same_start_rate = {
        "recent_start_rate": 0.5,
        "recent_minutes_ewma": 70.0,
        "fixture_congestion": 0.0,
        "chance_of_playing_next_round": 100.0,
        "status_score": 1.0,
        "days_since_last_appearance": 7.0,
        "zero_minute_streak_length": 0.0,
        "start_rate_last_3": 0.5,
        "start_rate_last_6": 0.5,
        "start_rate_last_15": 0.5,
        "team_rotation_propensity": 0.3,
        "price": 6.0,
        "ownership_log": 2.0,
        "transfers_out_share": 0.02,
        "transfers_balance_share": 0.0,
    }
    gk_dist = model.predict(pd.DataFrame([{**same_start_rate, "is_goalkeeper": 1.0}]))[0]
    outfield_dist = model.predict(pd.DataFrame([{**same_start_rate, "is_goalkeeper": 0.0}]))[0]

    assert gk_dist.p_60_plus > outfield_dist.p_60_plus


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
            "days_since_last_appearance": [0.0] * n,
            "zero_minute_streak_length": [0.0] * n,
            "start_rate_last_3": [1.0] * n,
            "start_rate_last_6": [1.0] * n,
            "start_rate_last_15": [1.0] * n,
            "team_rotation_propensity": [0.2] * n,
            "price": [8.0] * n,
            "ownership_log": [2.5] * n,
            "transfers_out_share": [0.01] * n,
            "transfers_balance_share": [0.0] * n,
            "is_goalkeeper": [0.0] * n,
        }
    )
    started = pd.Series([1] * n)
    minutes = pd.Series([90.0] * n)

    model = MinutesModel().fit(features, started, minutes)
    results = model.predict(features)
    assert results[0].p_60_plus == pytest.approx(1.0, abs=1e-6)
    assert results[0].p_zero == pytest.approx(0.0, abs=1e-6)
