"""Tests for GET /teams/fixture-swing -- the thin FastAPI wiring over
features.fixture_swing/api.fixture_swing_panel. Swing-rating correctness is already covered at the
features/ level (test_fixture_swing.py); these tests check window resolution against the app's own
current gameweek, the owned-squad marker, and the no-rate-data degrade path.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.state as state_module
from api.squad_state import SquadState
from api.state import AppState
from features.fixtures import TeamRates
from features.team_state import SquadPlayer

TEAM_A, TEAM_B, TEAM_C = 1, 2, 3


def _fixture_app_state(team_rates: dict[int, TeamRates] | None = None) -> AppState:
    teams = {
        TEAM_A: {"name": "Team A", "short_name": "TMA"},
        TEAM_B: {"name": "Team B", "short_name": "TMB"},
        TEAM_C: {"name": "Team C", "short_name": "TMC"},
    }
    fixtures = [
        # Team A faces the strong-defense side (B) near-term, the leaky side (C) further out --
        # an improving attacking run.
        {"team_id": TEAM_A, "opponent_id": TEAM_B, "gameweek": 1, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_B, "gameweek": 2, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_B, "gameweek": 3, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_C, "gameweek": 4, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_C, "gameweek": 5, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_C, "gameweek": 6, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_C, "gameweek": 7, "is_home": True},
        {"team_id": TEAM_A, "opponent_id": TEAM_C, "gameweek": 8, "is_home": True},
    ]
    players = {99: {"web_name": "Owned Player", "team_id": TEAM_B, "position": "MID", "price": 55}}
    return AppState(
        season="2026-27",
        gameweek=1,
        horizon_gameweeks=[1, 2, 3],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        model_version="test",
        projections={},
        players=players,
        teams=teams,
        fixtures=fixtures,
        diagnostics={},
        team_rates=team_rates or {},
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    team_rates = {
        TEAM_B: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=0.5, away_xga_per_90=0.5
        ),
        TEAM_C: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=3.0, away_xga_per_90=3.0
        ),
    }
    state_module.set_app_state(_fixture_app_state(team_rates))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state_module.reset_state()


def _row_for(rows: list[dict], team_id: int) -> dict:
    return next(row for row in rows if row["team_id"] == team_id)


class TestFixtureSwing:
    def test_default_windows_are_the_locked_in_three_and_five(self, client):
        response = client.get("/teams/fixture-swing")
        assert response.status_code == 200
        body = response.json()
        assert body["near_gameweeks"] == [1, 2, 3]
        assert body["far_gameweeks"] == [4, 5, 6, 7, 8]

    def test_explicit_windows_resolve_relative_to_the_current_gameweek(self, client):
        response = client.get("/teams/fixture-swing", params={"near": 2, "far": 2})
        assert response.status_code == 200
        body = response.json()
        assert body["near_gameweeks"] == [1, 2]
        assert body["far_gameweeks"] == [3, 4]

    def test_improving_run_has_a_positive_attack_swing(self, client):
        rows = client.get("/teams/fixture-swing").json()["rows"]
        team_a = _row_for(rows, TEAM_A)
        assert team_a["attack_swing"] > 0
        assert team_a["near"]["attack_rating"] > team_a["far"]["attack_rating"]

    def test_team_with_no_fixtures_gets_null_windows_and_swings(self, client):
        rows = client.get("/teams/fixture-swing").json()["rows"]
        # Team B has no fixture rows of its own anywhere in this fixture list.
        team_b = _row_for(rows, TEAM_B)
        assert team_b["near"] is None
        assert team_b["far"] is None
        assert team_b["attack_swing"] is None
        assert team_b["defense_swing"] is None

    def test_every_team_produces_exactly_one_row(self, client):
        rows = client.get("/teams/fixture-swing").json()["rows"]
        assert {row["team_id"] for row in rows} == {TEAM_A, TEAM_B, TEAM_C}

    def test_owned_player_marks_their_team(self, client):
        squad_state = SquadState(squad=(SquadPlayer(player_id=99, position="MID", price=55),))
        state_module.set_squad_state(squad_state)

        rows = client.get("/teams/fixture-swing").json()["rows"]

        assert _row_for(rows, TEAM_B)["has_owned_player"] is True
        assert _row_for(rows, TEAM_A)["has_owned_player"] is False
        assert _row_for(rows, TEAM_C)["has_owned_player"] is False

    def test_no_owned_players_marks_every_team_false(self, client):
        rows = client.get("/teams/fixture-swing").json()["rows"]
        assert all(row["has_owned_player"] is False for row in rows)

    def test_zero_near_is_rejected(self, client):
        response = client.get("/teams/fixture-swing", params={"near": 0})
        assert response.status_code == 400

    def test_zero_far_is_rejected(self, client):
        response = client.get("/teams/fixture-swing", params={"far": 0})
        assert response.status_code == 400


def test_no_team_rates_yet_returns_an_empty_row_list():
    state_module.reset_state()
    state_module.set_app_state(_fixture_app_state(team_rates={}))
    from api.main import app

    with TestClient(app) as test_client:
        response = test_client.get("/teams/fixture-swing")

    assert response.status_code == 200
    assert response.json()["rows"] == []
    state_module.reset_state()
