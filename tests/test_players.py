"""Tests for features/players.py — search/comparison over AppState.projections (BUILD_PLAN 5.2)."""

from __future__ import annotations

import numpy as np
import pytest

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection, PlayerHorizonProjection
from engine.simulate import PlayerSimulationSummary
from features.players import get_player_detail, search_players

MINUTES = MinutesDistribution(
    p_zero=0.05,
    p_1_to_59=0.10,
    p_60_plus=0.85,
    expected_minutes_given_1_to_59=30.0,
    expected_minutes_given_60_plus=88.0,
)


def _breakdown(appearance: float = 2.0) -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=appearance,
        goals=0.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )


def _gameweek_projection(
    player_id: int,
    position: str,
    gameweek: int,
    expected_points: float,
    with_simulation: bool = False,
) -> PlayerGameweekProjection:
    breakdown = _breakdown(expected_points)
    simulation = None
    if with_simulation:
        simulation = PlayerSimulationSummary(
            player_id=player_id,
            mean=expected_points,
            median=expected_points,
            floor=expected_points - 1,
            ceiling=expected_points + 5,
            prob_big_haul=0.2,
            raw_points=np.array([expected_points]),
        )
    return PlayerGameweekProjection(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        minutes=MINUTES,
        breakdown=breakdown,
        simulation=simulation,
    )


def _horizon(
    player_id: int, position: str, points_by_gameweek: dict[int, float], with_simulation=False
) -> PlayerHorizonProjection:
    return PlayerHorizonProjection(
        player_id=player_id,
        position=position,
        gameweeks={
            gw: _gameweek_projection(player_id, position, gw, pts, with_simulation)
            for gw, pts in points_by_gameweek.items()
        },
    )


@pytest.fixture
def projections():
    return {
        1: _horizon(1, "MID", {1: 6.0}, with_simulation=True),
        2: _horizon(2, "FWD", {1: 4.0}),
        3: _horizon(3, "DEF", {1: 8.0}),
    }


@pytest.fixture
def player_names():
    return {1: "Bruno Fernandes", 2: "Erling Haaland", 3: "Virgil van Dijk"}


@pytest.fixture
def buy_prices():
    return {1: 85, 2: 145, 3: 65}


def test_search_players_ranks_by_expected_points_descending(projections, player_names, buy_prices):
    results = search_players(projections, player_names, buy_prices)

    assert [r.player_id for r in results] == [3, 1, 2]


def test_search_players_filters_by_name_case_insensitively(projections, player_names, buy_prices):
    results = search_players(projections, player_names, buy_prices, search="haaland")

    assert [r.player_id for r in results] == [2]


def test_search_players_filters_by_position(projections, player_names, buy_prices):
    results = search_players(projections, player_names, buy_prices, position="DEF")

    assert [r.player_id for r in results] == [3]


def test_search_players_filters_by_max_price(projections, player_names, buy_prices):
    results = search_players(projections, player_names, buy_prices, max_price=90)

    assert {r.player_id for r in results} == {1, 3}


def test_search_players_excludes_player_missing_from_the_requested_gameweek(
    projections, player_names, buy_prices
):
    results = search_players(projections, player_names, buy_prices, gameweek=2)

    assert results == []


def test_search_players_defaults_to_each_players_earliest_gameweek(player_names, buy_prices):
    projections = {1: _horizon(1, "MID", {3: 5.0, 5: 9.0})}

    results = search_players(projections, player_names, buy_prices)

    assert results[0].gameweek == 3
    assert results[0].expected_points == pytest.approx(5.0)


def test_get_player_detail_returns_full_breakdown_and_simulation(
    projections, player_names, buy_prices
):
    detail = get_player_detail(1, projections, player_names, buy_prices)

    assert detail.name == "Bruno Fernandes"
    assert detail.position == "MID"
    assert detail.price == 85
    assert detail.expected_points == pytest.approx(6.0)
    assert detail.floor == pytest.approx(5.0)
    assert detail.ceiling == pytest.approx(11.0)
    assert detail.prob_big_haul == pytest.approx(0.2)
    assert isinstance(detail.breakdown, ComponentBreakdown)


def test_get_player_detail_has_null_simulation_fields_when_not_simulated(
    projections, player_names, buy_prices
):
    detail = get_player_detail(2, projections, player_names, buy_prices)

    assert detail.floor is None
    assert detail.ceiling is None
    assert detail.prob_big_haul is None


def test_get_player_detail_raises_for_unknown_player(projections, player_names, buy_prices):
    with pytest.raises(KeyError):
        get_player_detail(999, projections, player_names, buy_prices)


def test_get_player_detail_raises_for_a_gameweek_the_player_has_no_projection_for(
    projections, player_names, buy_prices
):
    with pytest.raises(KeyError):
        get_player_detail(1, projections, player_names, buy_prices, gameweek=99)
