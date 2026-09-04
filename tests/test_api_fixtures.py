"""Tests for the GET /fixtures endpoint -- the thin API wiring over api.fixtures_view. The
difficulty numbers themselves are just FPL's own values carried straight through, so these tests
check request/response wiring (gameweek window handling, blank/double gameweek shape, team
exclusion), not any rating math. The expected-goals fields are covered similarly: wiring only, the
underlying formula is exercised by tests/test_fixtures.py.
"""

from __future__ import annotations

import dataclasses
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.state as state_module
from api.state import AppState
from features.fixtures import TeamFixture, TeamRates, fixture_expected_goals, league_average_rate
from tests.conftest import UPCOMING_DEADLINE

TEAM_A, TEAM_B, TEAM_C = 1, 2, 3


def _fixture_app_state() -> AppState:
    teams = {
        TEAM_A: {"name": "Team A", "short_name": "TMA"},
        TEAM_B: {"name": "Team B", "short_name": "TMB"},
        TEAM_C: {"name": "Team C", "short_name": "TMC"},
    }
    fixtures = [
        # Team A: a normal fixture, a blank in GW2, a double in GW3, then two more normals.
        {"team_id": TEAM_A, "opponent_id": TEAM_B, "gameweek": 1, "is_home": True, "difficulty": 2},
        {"team_id": TEAM_A, "opponent_id": TEAM_B, "gameweek": 3, "is_home": True, "difficulty": 3},
        {
            "team_id": TEAM_A,
            "opponent_id": TEAM_C,
            "gameweek": 3,
            "is_home": False,
            "difficulty": 4,
        },
        {"team_id": TEAM_A, "opponent_id": TEAM_C, "gameweek": 4, "is_home": True, "difficulty": 5},
        {
            "team_id": TEAM_A,
            "opponent_id": TEAM_B,
            "gameweek": 5,
            "is_home": False,
            "difficulty": 1,
        },
        # Team C: one fixture, in GW2.
        {"team_id": TEAM_C, "opponent_id": TEAM_B, "gameweek": 2, "is_home": True, "difficulty": 3},
        # Team B has no fixture rows of its own anywhere in the horizon.
    ]
    return AppState(
        season="2026-27",
        gameweek=1,
        horizon_gameweeks=[1, 2, 3],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=UPCOMING_DEADLINE,
        model_version="test",
        projections={},
        players={},
        teams=teams,
        fixtures=fixtures,
        diagnostics={},
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    state_module.set_app_state(_fixture_app_state())
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state_module.reset_state()


@pytest.fixture()
def client_with_team_rates(tmp_path):
    """Team A and C have live rate snapshots; Team B deliberately doesn't, exercising the "no live
    source yet" degrade-gracefully path for any fixture involving it."""
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    app_state = dataclasses.replace(
        _fixture_app_state(),
        team_rates={
            TEAM_A: TeamRates(
                home_xg_per_90=1.6, away_xg_per_90=1.2, home_xga_per_90=1.0, away_xga_per_90=1.3
            ),
            TEAM_C: TeamRates(
                home_xg_per_90=1.5, away_xg_per_90=1.5, home_xga_per_90=0.9, away_xga_per_90=0.9
            ),
        },
    )
    state_module.set_app_state(app_state)
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state_module.reset_state()


def _row_for(rows: list[dict], team_id: int) -> dict:
    return next(row for row in rows if row["team_id"] == team_id)


class TestFixtureTicker:
    def test_default_horizon_is_five_gameweeks(self, client):
        response = client.get("/fixtures")
        assert response.status_code == 200
        rows = response.json()
        team_a = _row_for(rows, TEAM_A)
        assert [cell["gameweek"] for cell in team_a["gameweeks"]] == [1, 2, 3, 4, 5]

    def test_explicit_range_limits_gameweek_columns(self, client):
        response = client.get("/fixtures", params={"gameweek_from": 1, "gameweek_to": 2})
        assert response.status_code == 200
        rows = response.json()
        team_a = _row_for(rows, TEAM_A)
        assert [cell["gameweek"] for cell in team_a["gameweeks"]] == [1, 2]

    def test_explicit_range_need_not_start_at_current_gameweek(self, client):
        response = client.get("/fixtures", params={"gameweek_from": 4, "gameweek_to": 5})
        assert response.status_code == 200
        rows = response.json()
        team_a = _row_for(rows, TEAM_A)
        assert [cell["gameweek"] for cell in team_a["gameweeks"]] == [4, 5]

    def test_gameweek_to_defaults_from_explicit_gameweek_from(self, client):
        response = client.get("/fixtures", params={"gameweek_from": 3})
        assert response.status_code == 200
        rows = response.json()
        team_a = _row_for(rows, TEAM_A)
        # Same default 5-gameweek span as the no-params case, just chained off gameweek_from=3.
        assert [cell["gameweek"] for cell in team_a["gameweeks"]] == [3, 4, 5, 6, 7]

    def test_blank_gameweek_has_no_fixtures(self, client):
        rows = client.get("/fixtures").json()
        team_a = _row_for(rows, TEAM_A)
        gw2 = next(cell for cell in team_a["gameweeks"] if cell["gameweek"] == 2)
        assert gw2["fixtures"] == []

    def test_double_gameweek_has_two_fixtures(self, client):
        rows = client.get("/fixtures").json()
        team_a = _row_for(rows, TEAM_A)
        gw3 = next(cell for cell in team_a["gameweeks"] if cell["gameweek"] == 3)
        assert len(gw3["fixtures"]) == 2
        assert {entry["opponent_id"] for entry in gw3["fixtures"]} == {TEAM_B, TEAM_C}

    def test_average_difficulty_covers_every_fixture_including_the_double(self, client):
        rows = client.get("/fixtures").json()
        team_a = _row_for(rows, TEAM_A)
        # difficulties 2, 3, 4, 5, 1 across GW1-5 (the double gameweek contributes two values).
        assert team_a["average_difficulty"] == pytest.approx(3.0)

    def test_team_with_no_fixtures_has_null_average_and_empty_cells(self, client):
        rows = client.get("/fixtures").json()
        team_b = _row_for(rows, TEAM_B)
        assert team_b["average_difficulty"] is None
        assert all(cell["fixtures"] == [] for cell in team_b["gameweeks"])

    def test_every_team_produces_exactly_one_row(self, client):
        rows = client.get("/fixtures").json()
        assert {row["team_id"] for row in rows} == {TEAM_A, TEAM_B, TEAM_C}

    def test_gameweek_to_before_gameweek_from_is_rejected(self, client):
        response = client.get("/fixtures", params={"gameweek_from": 3, "gameweek_to": 2})
        assert response.status_code == 400

    def test_gameweek_from_below_one_is_rejected(self, client):
        response = client.get("/fixtures", params={"gameweek_from": 0, "gameweek_to": 2})
        assert response.status_code == 400


class TestFixtureTickerExpectedGoals:
    def test_expected_goals_omitted_when_no_team_rates_in_app_state(self, client):
        # Plain `client` fixture has no team_rates at all -- every fixture's expected-goals fields
        # stay None, matching this endpoint's pre-expected-goals behaviour exactly.
        rows = client.get("/fixtures").json()
        team_a = _row_for(rows, TEAM_A)
        gw4 = next(cell for cell in team_a["gameweeks"] if cell["gameweek"] == 4)
        assert gw4["fixtures"][0]["expected_goals_for"] is None
        assert gw4["fixtures"][0]["expected_goals_against"] is None

    def test_expected_goals_present_when_both_teams_have_a_rate_snapshot(
        self, client_with_team_rates
    ):
        # GW4: Team A (home) vs Team C (away) -- both have rate snapshots.
        rows = client_with_team_rates.get("/fixtures").json()
        team_a = _row_for(rows, TEAM_A)
        gw4_entry = next(cell for cell in team_a["gameweeks"] if cell["gameweek"] == 4)["fixtures"][
            0
        ]

        team_rates = {
            TEAM_A: TeamRates(1.6, 1.2, 1.0, 1.3),
            TEAM_C: TeamRates(1.5, 1.5, 0.9, 0.9),
        }
        league_avg_xga = league_average_rate(team_rates, "home_xga_per_90", "away_xga_per_90")
        expected = fixture_expected_goals(
            TeamFixture(team_id=TEAM_A, opponent_id=TEAM_C, gameweek=4, is_home=True),
            team_rates[TEAM_A],
            team_rates[TEAM_C],
            league_avg_xga,
        )
        assert gw4_entry["expected_goals_for"] == pytest.approx(expected.expected_goals_for)
        assert gw4_entry["expected_goals_against"] == pytest.approx(expected.expected_goals_against)

    def test_expected_goals_null_when_opponent_has_no_rate_snapshot(self, client_with_team_rates):
        # GW1: Team A (home) vs Team B -- Team B has no rate snapshot in client_with_team_rates.
        rows = client_with_team_rates.get("/fixtures").json()
        team_a = _row_for(rows, TEAM_A)
        gw1_entry = next(cell for cell in team_a["gameweeks"] if cell["gameweek"] == 1)["fixtures"][
            0
        ]
        assert gw1_entry["expected_goals_for"] is None
        assert gw1_entry["expected_goals_against"] is None
