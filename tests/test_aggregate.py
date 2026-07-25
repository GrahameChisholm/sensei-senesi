"""Tests for engine/aggregate.py — summing components into one projection (2.7)."""

from __future__ import annotations

import pytest

from engine.aggregate import (
    ComponentBreakdown,
    aggregate_gameweek,
    aggregate_horizon,
)
from engine.models.assists import AssistProjection
from engine.models.bonus import BonusProjection
from engine.models.cards import CardsProjection
from engine.models.clean_sheets import CleanSheetProjection
from engine.models.defensive_contribution import DefensiveContributionProjection
from engine.models.goals import GoalProjection
from engine.models.minutes import MinutesDistribution
from engine.models.saves import SavesProjection


def _minutes(p_zero=0.1, p_1_59=0.2, p_60=0.7) -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=p_zero,
        p_1_to_59=p_1_59,
        p_60_plus=p_60,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=88.0,
    )


def _goals() -> GoalProjection:
    return GoalProjection(
        non_penalty_goal_rate=0.3, expected_penalty_goals=0.05, expected_penalty_misses=0.01
    )


def _assists() -> AssistProjection:
    return AssistProjection(assist_rate=0.2)


def _clean_sheet() -> CleanSheetProjection:
    return CleanSheetProjection(
        clean_sheet_probability=0.35,
        expected_goals_conceded_penalty=-0.2,
        team_for_lambda=1.3,
        team_against_lambda=1.2,
    )


def _bonus() -> BonusProjection:
    return BonusProjection(expected_bonus=0.5)


def _cards() -> CardsProjection:
    return CardsProjection(expected_yellow_cards=0.1, expected_red_cards=0.01)


def _defensive_contribution() -> DefensiveContributionProjection:
    return DefensiveContributionProjection(p_clears_threshold=0.4, threshold=10)


def _saves() -> SavesProjection:
    return SavesProjection(expected_saves=3.0, expected_penalties_faced=0.05)


def test_aggregate_gameweek_def_includes_defensive_contribution_and_goals_conceded():
    breakdown = aggregate_gameweek(
        "DEF",
        _minutes(),
        _goals(),
        _assists(),
        _clean_sheet(),
        _bonus(),
        _cards(),
        defensive_contribution=_defensive_contribution(),
    )
    assert isinstance(breakdown, ComponentBreakdown)
    assert breakdown.defensive_contribution == pytest.approx(0.4 * 2)
    assert breakdown.goals_conceded == pytest.approx(-0.2)
    assert breakdown.saves == 0.0


def test_aggregate_gameweek_fwd_has_no_goals_conceded_but_has_zero_clean_sheet_points():
    breakdown = aggregate_gameweek(
        "FWD",
        _minutes(),
        _goals(),
        _assists(),
        _clean_sheet(),
        _bonus(),
        _cards(),
        defensive_contribution=_defensive_contribution(),
    )
    assert breakdown.goals_conceded == 0.0
    assert breakdown.clean_sheet == 0.0  # FWD clean sheet points are 0 by rule


def test_aggregate_gameweek_gk_includes_saves_not_defensive_contribution():
    breakdown = aggregate_gameweek(
        "GK", _minutes(), _goals(), _assists(), _clean_sheet(), _bonus(), _cards(), saves=_saves()
    )
    assert breakdown.saves == pytest.approx(_saves().expected_points)
    assert breakdown.defensive_contribution == 0.0


def test_aggregate_gameweek_gk_requires_saves():
    with pytest.raises(ValueError):
        aggregate_gameweek(
            "GK", _minutes(), _goals(), _assists(), _clean_sheet(), _bonus(), _cards()
        )


def test_aggregate_gameweek_gk_rejects_defensive_contribution():
    with pytest.raises(ValueError):
        aggregate_gameweek(
            "GK",
            _minutes(),
            _goals(),
            _assists(),
            _clean_sheet(),
            _bonus(),
            _cards(),
            defensive_contribution=_defensive_contribution(),
            saves=_saves(),
        )


def test_aggregate_gameweek_outfield_requires_defensive_contribution():
    with pytest.raises(ValueError):
        aggregate_gameweek(
            "DEF", _minutes(), _goals(), _assists(), _clean_sheet(), _bonus(), _cards()
        )


def test_aggregate_gameweek_outfield_rejects_saves():
    with pytest.raises(ValueError):
        aggregate_gameweek(
            "DEF",
            _minutes(),
            _goals(),
            _assists(),
            _clean_sheet(),
            _bonus(),
            _cards(),
            defensive_contribution=_defensive_contribution(),
            saves=_saves(),
        )


def test_component_breakdown_total_sums_all_lines():
    breakdown = ComponentBreakdown(
        appearance=2.0,
        goals=4.0,
        assists=0.6,
        clean_sheet=1.4,
        goals_conceded=-0.3,
        defensive_contribution=0.8,
        saves=0.0,
        bonus=0.5,
        cards=-0.1,
        penalty_misses=-0.02,
    )
    expected_total = 2.0 + 4.0 + 0.6 + 1.4 - 0.3 + 0.8 + 0.0 + 0.5 - 0.1 - 0.02
    assert breakdown.total == pytest.approx(expected_total)


def test_aggregate_gameweek_appearance_points_blend_buckets():
    minutes = _minutes(p_zero=0.1, p_1_59=0.2, p_60=0.7)
    breakdown = aggregate_gameweek(
        "MID",
        minutes,
        _goals(),
        _assists(),
        _clean_sheet(),
        _bonus(),
        _cards(),
        defensive_contribution=_defensive_contribution(),
    )
    expected_appearance = 0.2 * 1 + 0.7 * 2
    assert breakdown.appearance == pytest.approx(expected_appearance)


def test_aggregate_horizon_rolls_up_multiple_gameweeks():
    breakdown_1 = aggregate_gameweek(
        "MID",
        _minutes(),
        _goals(),
        _assists(),
        _clean_sheet(),
        _bonus(),
        _cards(),
        defensive_contribution=_defensive_contribution(),
    )
    breakdown_2 = aggregate_gameweek(
        "MID",
        _minutes(p_zero=0.3, p_1_59=0.3, p_60=0.4),
        _goals(),
        _assists(),
        _clean_sheet(),
        _bonus(),
        _cards(),
        defensive_contribution=_defensive_contribution(),
    )
    projection = aggregate_horizon(101, "MID", {1: breakdown_1, 2: breakdown_2})
    assert projection.per_gameweek_points == {1: breakdown_1.total, 2: breakdown_2.total}
    assert projection.horizon_total_points == pytest.approx(breakdown_1.total + breakdown_2.total)


def test_aggregate_horizon_rejects_empty_breakdowns():
    with pytest.raises(ValueError):
        aggregate_horizon(101, "MID", {})
