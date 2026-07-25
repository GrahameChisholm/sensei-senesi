"""Tests for engine/projections.py — the top-level player projection entry point."""

from __future__ import annotations

import numpy as np
import pytest

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerGameweekProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.simulate import PlayerSimulationSummary


def _minutes() -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.2,
        p_60_plus=0.7,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=88.0,
    )


def _breakdown(total_hint: float = 0.0) -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=1.6,
        goals=1.2 + total_hint,
        assists=0.4,
        clean_sheet=0.9,
        goals_conceded=-0.2,
        defensive_contribution=0.6,
        saves=0.0,
        bonus=0.5,
        cards=-0.05,
        penalty_misses=-0.01,
    )


def _simulation(player_id: int) -> PlayerSimulationSummary:
    return PlayerSimulationSummary(
        player_id=player_id,
        mean=4.5,
        median=4.0,
        floor=1.0,
        ceiling=10.0,
        prob_big_haul=0.08,
        raw_points=np.array([4.0, 5.0, 3.0]),
    )


def test_project_player_gameweek_expected_points_matches_breakdown_total():
    breakdown = _breakdown()
    projection = project_player_gameweek(101, "MID", 5, _minutes(), breakdown)
    assert projection.expected_points == pytest.approx(breakdown.total)
    assert projection.simulation is None


def test_project_player_gameweek_attaches_simulation():
    projection = project_player_gameweek(
        101, "MID", 5, _minutes(), _breakdown(), simulation=_simulation(101)
    )
    assert projection.simulation is not None
    assert projection.simulation.mean == pytest.approx(4.5)


def test_project_player_gameweek_rejects_mismatched_simulation_player_id():
    with pytest.raises(ValueError):
        project_player_gameweek(
            101, "MID", 5, _minutes(), _breakdown(), simulation=_simulation(999)
        )


def test_project_player_horizon_rolls_up_points():
    gw1 = project_player_gameweek(101, "MID", 1, _minutes(), _breakdown(total_hint=0.0))
    gw2 = project_player_gameweek(101, "MID", 2, _minutes(), _breakdown(total_hint=0.5))
    horizon = project_player_horizon(101, "MID", {1: gw1, 2: gw2})

    assert horizon.per_gameweek_points == {1: gw1.expected_points, 2: gw2.expected_points}
    assert horizon.horizon_total_points == pytest.approx(gw1.expected_points + gw2.expected_points)


def test_project_player_horizon_rejects_empty_gameweeks():
    with pytest.raises(ValueError):
        project_player_horizon(101, "MID", {})


def test_player_horizon_projection_rejects_mismatched_player_id():
    gw1 = project_player_gameweek(101, "MID", 1, _minutes(), _breakdown())
    with pytest.raises(ValueError):
        project_player_horizon(999, "MID", {1: gw1})


def test_player_horizon_projection_rejects_mismatched_position():
    gw1 = project_player_gameweek(101, "MID", 1, _minutes(), _breakdown())
    with pytest.raises(ValueError):
        project_player_horizon(101, "DEF", {1: gw1})


def test_player_gameweek_projection_is_a_dataclass_instance():
    projection = project_player_gameweek(101, "MID", 1, _minutes(), _breakdown())
    assert isinstance(projection, PlayerGameweekProjection)
