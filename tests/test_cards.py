"""Tests for engine/models/cards.py — historical card-rate model (2.6)."""

from __future__ import annotations

import pytest

from engine.models.cards import (
    CardsProjection,
    OwnGoalProjection,
    project_cards,
    project_own_goals,
)
from engine.scoring import OWN_GOAL_POINTS, RED_CARD_POINTS, YELLOW_CARD_POINTS


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


def test_project_own_goals_scales_with_minutes():
    full = project_own_goals(0.02, expected_minutes=90.0)
    half = project_own_goals(0.02, expected_minutes=45.0)
    assert half.expected_own_goals == pytest.approx(full.expected_own_goals / 2)


def test_project_own_goals_rejects_negative_rate():
    with pytest.raises(ValueError):
        project_own_goals(-0.01)


def test_project_own_goals_rejects_negative_minutes():
    with pytest.raises(ValueError):
        project_own_goals(0.02, expected_minutes=-1.0)


def test_own_goal_projection_expected_points_is_negative():
    projection = project_own_goals(0.02, expected_minutes=90.0)
    assert projection.expected_points < 0
    assert projection.expected_points == pytest.approx(0.02 * OWN_GOAL_POINTS)


def test_own_goal_projection_rejects_negative_field():
    with pytest.raises(ValueError):
        OwnGoalProjection(expected_own_goals=-0.01)


def test_project_cards_shrinkage_disabled_by_default():
    # A thin-sample outlier rate (one red card from a 3-minute cameo) is used unmodified when the
    # caller doesn't opt into shrinkage -- matching every pre-A.3 caller's behavior unchanged.
    projection = project_cards(0.0, 22.9, expected_minutes=90.0)
    assert projection.expected_red_cards == pytest.approx(22.9)


def test_project_cards_shrinkage_pulls_a_thin_sample_toward_the_league_prior():
    # ENGINE_IMPROVEMENTS_3.md A.3: one dismissal from a single 3-minute cameo should not be taken
    # at face value.
    shrunk = project_cards(
        0.0,
        22.9,
        expected_minutes=90.0,
        individual_weight=3.0,
        league_avg_red_card_rate_per_90=0.02,
        red_shrinkage_k=1000.0,
    )
    # weight mostly on the prior: (3*22.9 + 1000*0.02) / 1003
    assert shrunk.expected_red_cards == pytest.approx((3 * 22.9 + 1000 * 0.02) / 1003)
    assert shrunk.expected_red_cards < 1.0


def test_project_cards_shrinkage_yellow_and_red_are_independent():
    shrunk = project_cards(
        1.0,
        1.0,
        expected_minutes=90.0,
        individual_weight=5.0,
        league_avg_yellow_card_rate_per_90=0.2,
        league_avg_red_card_rate_per_90=0.02,
        yellow_shrinkage_k=50.0,
        red_shrinkage_k=500.0,
    )
    # Red shrinks much harder toward its (lower) prior than yellow does toward its own.
    assert shrunk.expected_red_cards < shrunk.expected_yellow_cards


def test_project_own_goals_shrinkage_disabled_by_default():
    projection = project_own_goals(0.5, expected_minutes=90.0)
    assert projection.expected_own_goals == pytest.approx(0.5)


def test_project_own_goals_shrinkage_pulls_toward_the_league_prior():
    shrunk = project_own_goals(
        0.5,
        expected_minutes=90.0,
        individual_weight=3.0,
        league_avg_own_goal_rate_per_90=0.005,
        shrinkage_k=500.0,
    )
    assert shrunk.expected_own_goals == pytest.approx((3 * 0.5 + 500 * 0.005) / 503)
    assert shrunk.expected_own_goals < 0.1
