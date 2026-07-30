"""Tests for engine/models/bonus.py — regression proxy bonus model (2.6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.bonus import (
    FEATURE_COLUMNS,
    BonusModel,
    BonusProjection,
    build_features,
    expected_bonus_from_fixture_strengths,
    plackett_luce_rank_probabilities,
)


def test_build_features_one_hot_encodes_position():
    row = build_features(
        expected_goals=0.3,
        expected_assists=0.1,
        clean_sheet_probability=0.4,
        defensive_action_rate=8.0,
        position="DEF",
        expected_minutes=90.0,
    )
    assert row["position_DEF"] == 1.0
    assert row["position_MID"] == 0.0
    assert row["expected_minutes"] == 90.0
    assert set(row) == set(FEATURE_COLUMNS)


def test_build_features_rejects_unknown_position():
    with pytest.raises(ValueError):
        build_features(0.3, 0.1, 0.4, 8.0, "XYZ", expected_minutes=90.0)


def test_bonus_projection_rejects_out_of_range():
    with pytest.raises(ValueError):
        BonusProjection(expected_bonus=3.5)
    with pytest.raises(ValueError):
        BonusProjection(expected_bonus=-0.1)


def test_bonus_projection_expected_points_equals_expected_bonus():
    projection = BonusProjection(expected_bonus=1.2)
    assert projection.expected_points == pytest.approx(1.2)


def _synthetic_bonus_data(n: int = 100, seed: int = 0) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    rows = []
    bonus = []
    positions = ["GK", "DEF", "MID", "FWD"]
    for _ in range(n):
        position = rng.choice(positions)
        expected_goals = rng.uniform(0, 1)
        expected_assists = rng.uniform(0, 1)
        clean_sheet_probability = rng.uniform(0, 1)
        defensive_action_rate = rng.uniform(0, 15)
        expected_minutes = rng.uniform(0, 90)
        rows.append(
            build_features(
                expected_goals,
                expected_assists,
                clean_sheet_probability,
                defensive_action_rate,
                position,
                expected_minutes=expected_minutes,
            )
        )
        # A simple synthetic "actual bonus" driven mostly by attacking output + clean sheets,
        # scaled by minutes (ENGINE_IMPROVEMENTS_3.md A.2) -- a player who barely appears cannot
        # realistically earn bonus regardless of their other inputs.
        raw = (1.5 * expected_goals + 1.0 * expected_assists + 1.0 * clean_sheet_probability) * (
            expected_minutes / 90.0
        )
        bonus.append(float(np.clip(raw, 0, 3)))
    return pd.DataFrame(rows), pd.Series(bonus)


def test_bonus_model_fit_predict_end_to_end():
    features, actual_bonus = _synthetic_bonus_data()
    model = BonusModel().fit(features, actual_bonus)
    projections = model.predict(features)

    assert len(projections) == len(features)
    for projection in projections:
        assert isinstance(projection, BonusProjection)
        assert 0.0 <= projection.expected_bonus <= 3.0


def test_bonus_model_higher_attacking_output_predicts_more_bonus():
    features, actual_bonus = _synthetic_bonus_data()
    model = BonusModel().fit(features, actual_bonus)

    prolific = pd.DataFrame([build_features(0.9, 0.5, 0.8, 12.0, "MID", expected_minutes=90.0)])
    quiet = pd.DataFrame([build_features(0.0, 0.0, 0.0, 0.0, "MID", expected_minutes=90.0)])

    assert model.predict(prolific)[0].expected_bonus > model.predict(quiet)[0].expected_bonus


def test_bonus_model_near_zero_minutes_predicts_near_zero_bonus():
    # ENGINE_IMPROVEMENTS_3.md A.2: a player the minutes model is confident won't appear should not
    # receive meaningful bonus on the strength of team-level clean-sheet probability or a raw
    # defensive-action rate alone -- both are independent of whether this player actually plays.
    features, actual_bonus = _synthetic_bonus_data()
    model = BonusModel().fit(features, actual_bonus)

    full_match = pd.DataFrame([build_features(0.9, 0.5, 0.8, 12.0, "MID", expected_minutes=90.0)])
    barely_plays = pd.DataFrame([build_features(0.9, 0.5, 0.8, 12.0, "MID", expected_minutes=2.0)])

    assert (
        model.predict(barely_plays)[0].expected_bonus < model.predict(full_match)[0].expected_bonus
    )


def test_bonus_model_predict_before_fit_raises():
    features = pd.DataFrame([build_features(0.3, 0.1, 0.4, 8.0, "DEF", expected_minutes=90.0)])
    with pytest.raises(RuntimeError):
        BonusModel().predict(features)


def _brute_force_rank_probabilities(strengths, max_rank):
    """Reference implementation via exhaustive permutation enumeration (ENGINE_IMPROVEMENTS_3.md
    D.2) -- only tractable for small n, used purely to cross-check the exact recursive
    formulation in plackett_luce_rank_probabilities."""
    import itertools

    w = np.asarray(strengths, dtype=float)
    n = len(w)
    probs = np.zeros((n, max_rank))
    for perm in itertools.permutations(range(n)):
        remaining = float(w.sum())
        p = 1.0
        for item in perm:
            p *= w[item] / remaining
            remaining -= w[item]
        for rank, item in enumerate(perm[:max_rank]):
            probs[item, rank] += p
    return probs


def test_plackett_luce_rank_probabilities_matches_brute_force_permutation_enumeration():
    rng = np.random.default_rng(0)
    strengths = rng.uniform(0.1, 5.0, size=5)

    exact = plackett_luce_rank_probabilities(strengths, max_rank=3)
    brute_force = _brute_force_rank_probabilities(strengths, max_rank=3)

    np.testing.assert_allclose(exact, brute_force, atol=1e-10)


def test_plackett_luce_rank_probabilities_rows_sum_to_at_most_one():
    strengths = np.array([5.0, 3.0, 1.0, 1.0, 0.5, 0.2])
    probs = plackett_luce_rank_probabilities(strengths, max_rank=3)
    assert np.all(probs >= 0)
    assert np.all(probs.sum(axis=1) <= 1.0 + 1e-9)
    # every rank's probabilities across all items sum to exactly 1 (someone takes each rank)
    assert probs[:, 0].sum() == pytest.approx(1.0)
    assert probs[:, 1].sum() == pytest.approx(1.0)
    assert probs[:, 2].sum() == pytest.approx(1.0)


def test_plackett_luce_rank_probabilities_highest_strength_most_likely_first():
    probs = plackett_luce_rank_probabilities(np.array([10.0, 1.0, 1.0, 1.0]), max_rank=1)
    assert probs[0, 0] > probs[1, 0]


def test_plackett_luce_rank_probabilities_rejects_negative_strengths():
    with pytest.raises(ValueError):
        plackett_luce_rank_probabilities(np.array([1.0, -1.0]))


def test_plackett_luce_rank_probabilities_rejects_all_zero_strengths():
    with pytest.raises(ValueError):
        plackett_luce_rank_probabilities(np.array([0.0, 0.0]))


def test_expected_bonus_from_fixture_strengths_sums_to_at_most_six():
    # 3 + 2 + 1 total bonus points are handed out per fixture; expected value across all players
    # can't exceed that.
    strengths = np.array([8.0, 4.0, 2.0, 1.0, 1.0, 0.5, 0.5, 0.2])
    expected = expected_bonus_from_fixture_strengths(strengths)
    assert expected.sum() == pytest.approx(6.0)
    assert np.all(expected >= 0)


def test_expected_bonus_from_fixture_strengths_ranks_by_strength():
    strengths = np.array([10.0, 5.0, 1.0, 1.0, 1.0])
    expected = expected_bonus_from_fixture_strengths(strengths)
    assert expected[0] > expected[1] > expected[2]
