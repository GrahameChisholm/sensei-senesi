"""Tests for engine/models/cards.py — historical card-rate model (2.6)."""

from __future__ import annotations

import pytest

from engine.models.cards import CardsProjection, project_cards
from engine.scoring import RED_CARD_POINTS, YELLOW_CARD_POINTS


def test_project_cards_scales_with_minutes():
    full = project_cards(0.3, 0.02, expected_minutes=90.0)
    half = project_cards(0.3, 0.02, expected_minutes=45.0)
    assert half.expected_yellow_cards == pytest.approx(full.expected_yellow_cards / 2)
    assert half.expected_red_cards == pytest.approx(full.expected_red_cards / 2)


def test_project_cards_rejects_negative_rates():
    with pytest.raises(ValueError):
        project_cards(-0.1, 0.02)
    with pytest.raises(ValueError):
        project_cards(0.3, -0.02)


def test_project_cards_rejects_negative_minutes():
    with pytest.raises(ValueError):
        project_cards(0.3, 0.02, expected_minutes=-1.0)


def test_cards_projection_expected_points_is_negative():
    projection = project_cards(0.3, 0.02, expected_minutes=90.0)
    assert projection.expected_points < 0
    assert projection.expected_points == pytest.approx(
        0.3 * YELLOW_CARD_POINTS + 0.02 * RED_CARD_POINTS
    )


def test_cards_projection_rejects_negative_fields():
    with pytest.raises(ValueError):
        CardsProjection(expected_yellow_cards=-0.1, expected_red_cards=0.0)
    with pytest.raises(ValueError):
        CardsProjection(expected_yellow_cards=0.1, expected_red_cards=-0.02)
