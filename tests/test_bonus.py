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
)


def test_build_features_one_hot_encodes_position():
    row = build_features(
        expected_goals=0.3,
        expected_assists=0.1,
        clean_sheet_probability=0.4,
        defensive_action_rate=8.0,
        position="DEF",
    )
    assert row["position_DEF"] == 1.0
    assert row["position_MID"] == 0.0
    assert set(row) == set(FEATURE_COLUMNS)


def test_build_features_rejects_unknown_position():
    with pytest.raises(ValueError):
        build_features(0.3, 0.1, 0.4, 8.0, "XYZ")


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
        rows.append(
            build_features(
                expected_goals,
                expected_assists,
                clean_sheet_probability,
                defensive_action_rate,
                position,
            )
        )
        # A simple synthetic "actual bonus" driven mostly by attacking output + clean sheets.
        raw = 1.5 * expected_goals + 1.0 * expected_assists + 1.0 * clean_sheet_probability
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

    prolific = pd.DataFrame([build_features(0.9, 0.5, 0.8, 12.0, "MID")])
    quiet = pd.DataFrame([build_features(0.0, 0.0, 0.0, 0.0, "MID")])

    assert model.predict(prolific)[0].expected_bonus > model.predict(quiet)[0].expected_bonus


def test_bonus_model_predict_before_fit_raises():
    features = pd.DataFrame([build_features(0.3, 0.1, 0.4, 8.0, "DEF")])
    with pytest.raises(RuntimeError):
        BonusModel().predict(features)
