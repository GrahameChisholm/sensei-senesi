"""Tests for GET /mini-league/{league_id} -- the thin FastAPI wiring over
api.mini_league_panel.build_mini_league_panel plus a live (here, stubbed) FPL fetch. Panel-assembly
correctness is already covered at the api.mini_league_panel level (test_mini_league_panel.py);
these tests check the plumbing: the two required-configuration guards, the FPLClientError ->
ValueError -> 400 translation, the response shape, and that the cache/refresh wiring reaches the
real client.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.state as state_module
from api.mini_league_panel import reset_snapshot_cache
from api.state import AppState
from engine.aggregate import ComponentBreakdown
from engine.data.fpl_client import FPLClientError
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]
ALL_IDS = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]
MY_ENTRY_ID = 555


def _position_for(player_id: int) -> str:
    if player_id in (GK1, GK2):
        return GK
    if player_id in DEF_IDS:
        return DEF
    if player_id in MID_IDS:
        return MID
    return FWD


def _horizon(player_id: int, points: float = 4.0, gameweeks=(1,)):
    position = _position_for(player_id)
    minutes = MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )
    breakdown = ComponentBreakdown(
        appearance=2.0,
        goals=points - 2.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )
    return project_player_horizon(
        player_id,
        position,
        {
            gw: project_player_gameweek(player_id, position, gw, minutes, breakdown)
            for gw in gameweeks
        },
    )


def _fixture_app_state() -> AppState:
    # Spread the 15 squad players across 5 clubs (3 each) so MAX_PER_CLUB is never accidentally
    # violated -- team_id = 100 + (index // 3), matching test_api_squad.py's own fixture.
    players = {}
    teams = {}
    for index, player_id in enumerate(ALL_IDS):
        team_id = 100 + (index // 3)
        teams.setdefault(team_id, {"name": f"Team {team_id}", "short_name": f"T{team_id}"})
        players[player_id] = {
            "web_name": f"Player{player_id}",
            "team_id": team_id,
            "position": _position_for(player_id),
            "price": 40,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
        }
    projections = {pid: _horizon(pid) for pid in ALL_IDS}
    return AppState(
        season="2026-27",
        gameweek=1,
        horizon_gameweeks=[1],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        model_version="test",
        projections=projections,
        players=players,
        teams=teams,
        fixtures=[],
        diagnostics={},
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    state_module.set_app_state(_fixture_app_state())
    reset_snapshot_cache()
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_snapshot_cache()
    state_module.reset_state()


def _build_full_squad(client: TestClient) -> None:
    for player_id in ALL_IDS:
        response = client.post(
            "/squad/players",
            json={"player_id": player_id, "position": _position_for(player_id), "price": 40},
        )
        assert response.status_code == 200, response.json()


def _configure_fpl_team_id(client: TestClient, team_id: int = MY_ENTRY_ID) -> None:
    response = client.post(
        "/mini-league/leagues", json={"fpl_team_id": team_id, "mini_league_ids": []}
    )
    assert response.status_code == 200, response.json()


_DEFAULT_STANDINGS = [
    {
        "entry": MY_ENTRY_ID,
        "player_name": "Me",
        "entry_name": "My Team",
        "rank": 1,
        "total": 100,
        "event_total": 0,
    },
    {
        "entry": 1,
        "player_name": "Dave",
        "entry_name": "Dave's Team",
        "rank": 2,
        "total": 90,
        "event_total": 0,
    },
]


class _StubFPLClient:
    def __init__(self, standings=None, error=None, on_standings_call=None):
        self._standings = standings if standings is not None else _DEFAULT_STANDINGS
        self._error = error
        self._on_standings_call = on_standings_call

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_league_standings(self, league_id, page=1):
        if self._on_standings_call is not None:
            self._on_standings_call(page)
        if self._error is not None:
            raise self._error
        return {
            "league": {"id": league_id, "name": "Test League"},
            "standings": {"has_next": False, "results": self._standings},
        }

    def get_entry(self, entry_id):
        return {"id": entry_id, "current_event": 1}

    def get_entry_picks(self, entry_id, gameweek):
        return {"picks": []}

    def get_entry_history(self, entry_id):
        return {"chips": []}


class TestGetMiniLeague:
    def test_requires_an_fpl_team_id_to_be_configured(self, client):
        _build_full_squad(client)
        response = client.get("/mini-league/999")
        assert response.status_code == 400

    def test_requires_a_complete_squad(self, client):
        _configure_fpl_team_id(client)
        response = client.get("/mini-league/999")
        assert response.status_code == 400

    def test_returns_a_full_panel(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_fpl_team_id(client)
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())

        response = client.get("/mini-league/999")
        assert response.status_code == 200, response.json()
        body = response.json()

        assert body["league_id"] == 999
        assert body["league_name"] == "Test League"
        assert body["gameweek"] == 1
        assert {rival["entry_id"] for rival in body["rivals"]} == {1}
        assert len(body["captain_options"]) == 11
        assert len(body["template_xi"]) <= 11
        assert isinstance(body["insights"], list)
        assert all(insight["kind"] in ("edge", "drag", "captain") for insight in body["insights"])

    def test_my_own_entry_never_appears_in_rivals(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_fpl_team_id(client)
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())

        body = client.get("/mini-league/999").json()
        assert MY_ENTRY_ID not in {rival["entry_id"] for rival in body["rivals"]}

    def test_unknown_league_becomes_a_400_not_a_500(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_fpl_team_id(client)
        monkeypatch.setattr(
            api_main, "FPLClient", lambda: _StubFPLClient(error=FPLClientError("boom"))
        )

        response = client.get("/mini-league/424242")
        assert response.status_code == 400

    def test_second_request_within_ttl_does_not_refetch(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_fpl_team_id(client)
        calls: list[int] = []
        monkeypatch.setattr(
            api_main,
            "FPLClient",
            lambda: _StubFPLClient(on_standings_call=lambda page: calls.append(page)),
        )

        client.get("/mini-league/999")
        client.get("/mini-league/999")
        assert len(calls) == 1

    def test_refresh_query_param_bypasses_the_cache(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_fpl_team_id(client)
        calls: list[int] = []
        monkeypatch.setattr(
            api_main,
            "FPLClient",
            lambda: _StubFPLClient(on_standings_call=lambda page: calls.append(page)),
        )

        client.get("/mini-league/999")
        client.get("/mini-league/999?refresh=true")
        assert len(calls) == 2
