"""Tests for GET /players/stats (PLAYER_STATS_PLAN Phase 3) -- the thin FastAPI wiring over
features.player_stats/api.player_stats_panel. Business-rule correctness (summing, small-sample
flagging, the G4 per-gameweek conversion order) is already covered at the features/ level
(test_player_stats_features.py, test_player_history.py); these tests check that the endpoint
wires the request to the right functions and returns the right shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.state as state_module
from engine.aggregate import ComponentBreakdown
from engine.data.player_history import PlayerGameweekActual
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import FWD, MID
from tests.conftest import UPCOMING_DEADLINE


def _minutes() -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )


def _breakdown() -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=2.0,
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


def _horizon(player_id: int, position: str, gameweeks=(1, 2, 3)):
    return project_player_horizon(
        player_id,
        position,
        {
            gw: project_player_gameweek(player_id, position, gw, _minutes(), _breakdown())
            for gw in gameweeks
        },
    )


def _actual(gameweek: int, **overrides) -> PlayerGameweekActual:
    base = dict(
        gameweek=gameweek,
        minutes=90,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        saves=0,
        yellow_cards=0,
        red_cards=0,
        bonus=0,
        defensive_contribution=0,
        total_points=2,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        expected_goals_conceded=0.0,
    )
    base.update(overrides)
    return PlayerGameweekActual(**base)


def _fixture_app_state():
    from api.state import AppState

    players = {
        1: {
            "web_name": "Player1",
            "full_name": "Player One",
            "team_id": 100,
            "position": FWD,
            "price": 90,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
            "selected_by_percent": 12.5,
            "penalties_order": 1,
        },
        2: {
            "web_name": "Player2",
            "full_name": "Player Two",
            "team_id": 101,
            "position": MID,
            "price": 70,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
            "selected_by_percent": None,
            "penalties_order": None,
        },
    }
    teams = {
        100: {"name": "Team 100", "short_name": "T100"},
        101: {"name": "Team 101", "short_name": "T101"},
    }
    projections = {1: _horizon(1, FWD), 2: _horizon(2, MID)}
    player_history = {
        1: [_actual(1, goals_scored=1, total_points=8), _actual(2, total_points=2)],
        # Player 2 has no history yet in gameweeks 1-2 (transferred in later).
        2: [_actual(5, total_points=2)],
    }

    return AppState(
        season="2026-27",
        gameweek=1,
        horizon_gameweeks=[1, 2, 3],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=UPCOMING_DEADLINE,
        model_version="test",
        projections=projections,
        players=players,
        teams=teams,
        fixtures=[],
        diagnostics={},
        player_history=player_history,
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    state_module.set_app_state(_fixture_app_state())
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state_module.reset_state()


def test_players_stats_returns_only_players_with_data_in_range(client: TestClient):
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 2})

    assert response.status_code == 200
    body = response.json()
    assert [row["player_id"] for row in body["rows"]] == [1]


def test_players_stats_sums_actuals_over_the_range(client: TestClient):
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 2})

    row = response.json()["rows"][0]
    assert row["actuals"]["goals_scored"] == 1
    assert row["actuals"]["total_points"] == 10
    assert row["actuals"]["gameweek_from"] == 1
    assert row["actuals"]["gameweek_to"] == 2


def test_players_stats_includes_points_breakdown(client: TestClient):
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 2})

    row = response.json()["rows"][0]
    assert row["actuals"]["points_breakdown"]["goals"] == pytest.approx(4.0)  # FWD, 1 goal


def test_players_stats_reports_why_league_ownership_is_missing(client: TestClient):
    """No mini league is configured in this fixture, so Own% must come back empty with the reason
    stated, never silently backfilled with FPL's population-wide selected_by_percent (12.5 here)
    under the same heading."""
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 2})

    body = response.json()
    assert body["ownership_status"] == "not_configured"
    assert body["rows"][0]["actuals"]["ownership_percent"] is None


def test_players_stats_marks_the_designated_penalty_taker(client: TestClient):
    """penalties_order lives on the snapshot's player records, so this asserts the whole path from
    there to the response rather than trusting the flag is wired up."""
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 10})

    takers = {
        row["player_id"]: row["actuals"]["is_penalty_taker"] for row in response.json()["rows"]
    }
    assert takers == {1: True, 2: False}


def test_players_stats_includes_fixture_cells_for_the_full_horizon(client: TestClient):
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 2})

    row = response.json()["rows"][0]
    assert [cell["gameweek"] for cell in row["fixtures"]] == [1, 2, 3]


def test_players_stats_includes_player_whose_range_covers_their_only_data(client: TestClient):
    response = client.get("/players/stats", params={"gameweek_from": 1, "gameweek_to": 10})

    assert {row["player_id"] for row in response.json()["rows"]} == {1, 2}


def test_players_stats_rejects_an_inverted_range(client: TestClient):
    response = client.get("/players/stats", params={"gameweek_from": 5, "gameweek_to": 1})

    assert response.status_code == 400


def test_players_stats_requires_both_range_params(client: TestClient):
    response = client.get("/players/stats")

    assert response.status_code == 422
