"""Tests for engine/models/saves.py — GK saves model (2.6)."""

from __future__ import annotations

import pytest

from engine.models.saves import (
    SavesProjection,
    expected_saves,
    expected_shots_faced,
    project_saves,
)
from engine.scoring import PENALTY_SAVE_POINTS


def test_expected_shots_faced_weaker_defence_faces_more_shots():
    weak_defence = expected_shots_faced(
        4.0, team_xga_per_90=2.0, league_avg_xga_per_90=1.4, is_home=True
    )
    strong_defence = expected_shots_faced(
        4.0, team_xga_per_90=0.8, league_avg_xga_per_90=1.4, is_home=True
    )
    assert weak_defence > strong_defence


def test_expected_shots_faced_away_faces_more_shots_than_home():
    home = expected_shots_faced(4.0, 1.4, 1.4, is_home=True)
    away = expected_shots_faced(4.0, 1.4, 1.4, is_home=False)
    assert away > home


def test_expected_shots_faced_scales_with_minutes():
    full = expected_shots_faced(4.0, 1.4, 1.4, is_home=True, expected_minutes=90.0)
    half = expected_shots_faced(4.0, 1.4, 1.4, is_home=True, expected_minutes=45.0)
    assert half == pytest.approx(full / 2)


def test_expected_shots_faced_rejects_negative_inputs():
    with pytest.raises(ValueError):
        expected_shots_faced(-1.0, 1.4, 1.4, is_home=True)


def test_expected_shots_faced_rejects_non_positive_league_average():
    with pytest.raises(ValueError):
        expected_shots_faced(4.0, 1.4, 0.0, is_home=True)


def test_expected_saves_applies_conversion_rate():
    assert expected_saves(6.0, save_conversion_rate=0.7) == pytest.approx(4.2)


def test_expected_saves_rejects_out_of_range_conversion_rate():
    with pytest.raises(ValueError):
        expected_saves(6.0, save_conversion_rate=1.5)


def test_saves_projection_expected_points_includes_penalty_bonus():
    without_penalties = SavesProjection(expected_saves=6.0, expected_penalties_faced=0.0)
    with_penalties = SavesProjection(
        expected_saves=6.0, expected_penalties_faced=0.1, penalty_save_rate=0.25
    )
    penalty_component = 0.1 * 0.25 * PENALTY_SAVE_POINTS
    assert with_penalties.expected_points == pytest.approx(
        without_penalties.expected_points + penalty_component
    )


def test_saves_projection_rejects_negative_fields():
    with pytest.raises(ValueError):
        SavesProjection(expected_saves=-1.0, expected_penalties_faced=0.0)
    with pytest.raises(ValueError):
        SavesProjection(expected_saves=1.0, expected_penalties_faced=-1.0)


def test_saves_projection_rejects_bad_penalty_save_rate():
    with pytest.raises(ValueError):
        SavesProjection(expected_saves=1.0, expected_penalties_faced=0.0, penalty_save_rate=1.5)


def test_project_saves_end_to_end():
    projection = project_saves(
        opponent_shots_on_target_per_90=4.0,
        team_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        is_home=True,
        expected_penalties_faced=0.05,
    )
    assert isinstance(projection, SavesProjection)
    assert projection.expected_saves > 0
    assert projection.expected_points > 0
