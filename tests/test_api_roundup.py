"""Tests for GET /roundup/{league_id} -- the thin FastAPI wiring over
api.roundup_panel.build_roundup_panel plus a live (here, stubbed) FPL fetch. Panel-assembly
correctness is already covered at the features.roundup level (test_roundup.py); these tests check
the plumbing: no fpl_team_id/squad requirement, gameweek auto-resolution from data_checked, the
FPLClientError -> ValueError -> 400 translation, the response shape, and cache/refresh behaviour.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.state as state_module
from api.roundup_panel import reset_roundup_cache
from engine.data.fpl_client import FPLClientError

GK_IDS = (1, 2)
DEF_IDS = (11, 12, 13, 14, 15)
MID_IDS = (21, 22, 23, 24, 25)
FWD_IDS = (31, 32, 33)
ALL_IDS = (*GK_IDS, *DEF_IDS, *MID_IDS, *FWD_IDS)


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    reset_roundup_cache()
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_roundup_cache()
    state_module.reset_state()


def _element_type(player_id: int) -> int:
    if player_id in GK_IDS:
        return 1
    if player_id in DEF_IDS:
        return 2
    if player_id in MID_IDS:
        return 3
    return 4


def _bootstrap(events=None):
    events = (
        events
        if events is not None
        else [
            {"id": 1, "data_checked": True},
            {"id": 2, "data_checked": True},
            {"id": 3, "data_checked": False},
        ]
    )
    return {
        "events": events,
        "elements": [
            {"id": pid, "web_name": f"Player{pid}", "team": 100, "element_type": _element_type(pid)}
            for pid in ALL_IDS
        ],
        "teams": [{"id": 100, "name": "Test Town", "short_name": "TST"}],
    }


def _live(points: dict[int, int]):
    return {
        "elements": [{"id": pid, "stats": {"total_points": pts}} for pid, pts in points.items()]
    }


_DEFAULT_STANDINGS = [
    {
        "entry": 1,
        "player_name": "Dave",
        "entry_name": "Dave's Team",
        "rank": 1,
        "total": 100,
        "event_total": 60,
    },
    {
        "entry": 2,
        "player_name": "Sam",
        "entry_name": "Sam's Team",
        "rank": 2,
        "total": 90,
        "event_total": 50,
    },
]


def _full_squad_picks(captain_id: int) -> dict:
    """A full, legal 15-man squad in a (3, 4, 3) formation -- enough position depth for
    league_template_xi's formation requirement, matching every real rival's actual squad shape.
    ``captain_id`` (must be a starter) gets multiplier 2; the bench (position 12-15, multiplier
    0) is the second goalkeeper plus the lowest-priority DEF/MID."""
    starting_xi = (GK_IDS[0], *DEF_IDS[:3], *MID_IDS[:4], *FWD_IDS[:3])
    bench = (GK_IDS[1], *DEF_IDS[3:5], MID_IDS[4])
    picks = []
    for position, pid in enumerate(starting_xi, start=1):
        multiplier = 2 if pid == captain_id else 1
        picks.append({"element": pid, "position": position, "multiplier": multiplier})
    for offset, pid in enumerate(bench, start=1):
        picks.append({"element": pid, "position": 11 + offset, "multiplier": 0})
    return {"picks": picks}


class _StubFPLClient:
    def __init__(
        self,
        bootstrap=None,
        live=None,
        standings=None,
        error=None,
        on_standings_call=None,
    ):
        self._bootstrap = bootstrap if bootstrap is not None else _bootstrap()
        self._live = (
            live
            if live is not None
            else _live({GK_IDS[0]: 4, DEF_IDS[0]: 6, MID_IDS[0]: 8, FWD_IDS[0]: 15})
        )
        self._standings = standings if standings is not None else _DEFAULT_STANDINGS
        self._error = error
        self._on_standings_call = on_standings_call

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_bootstrap_static(self):
        return self._bootstrap

    def get_event_live(self, gameweek):
        return self._live

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
        return {"id": entry_id, "current_event": 2}

    def get_entry_picks(self, entry_id, gameweek):
        captain_id = MID_IDS[0] if entry_id == 1 else FWD_IDS[0]
        return _full_squad_picks(captain_id)

    def get_entry_history(self, entry_id):
        return {
            "current": [
                {"event": 1, "points": 50, "total_points": 50},
                {"event": 2, "points": 60, "total_points": 110},
            ],
            "chips": [],
        }


class TestGetRoundup:
    def test_requires_no_squad_or_fpl_team_id(self, client, monkeypatch):
        """Unlike /mini-league/{league_id}, no squad and no saved fpl_team_id at all."""
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())
        response = client.get("/roundup/999")
        assert response.status_code == 200, response.json()

    def test_defaults_to_the_latest_data_checked_gameweek(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())
        body = client.get("/roundup/999").json()
        assert body["gameweek"] == 2

    def test_explicit_gameweek_overrides_the_default(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())
        body = client.get("/roundup/999?gameweek=1").json()
        assert body["gameweek"] == 1

    def test_no_data_checked_gameweek_becomes_a_400_not_a_500(self, client, monkeypatch):
        monkeypatch.setattr(
            api_main,
            "FPLClient",
            lambda: _StubFPLClient(bootstrap=_bootstrap(events=[{"id": 1, "data_checked": False}])),
        )
        response = client.get("/roundup/999")
        assert response.status_code == 400

    def test_response_includes_player_and_team_reference_data(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())
        body = client.get("/roundup/999").json()
        assert body["players"][str(GK_IDS[0])]["web_name"] == f"Player{GK_IDS[0]}"
        assert body["teams"]["100"]["name"] == "Test Town"

    def test_top_player_and_captain_returns_are_populated(self, client, monkeypatch):
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())
        body = client.get("/roundup/999").json()
        assert body["top_player"]["player_id"] == FWD_IDS[0]  # highest live points, 15
        assert {row["entry_id"] for row in body["captain_returns"]} == {1, 2}

    def test_unknown_league_becomes_a_400_not_a_500(self, client, monkeypatch):
        monkeypatch.setattr(
            api_main, "FPLClient", lambda: _StubFPLClient(error=FPLClientError("boom"))
        )
        response = client.get("/roundup/424242")
        assert response.status_code == 400

    def test_second_request_is_served_from_the_indefinite_cache(self, client, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(
            api_main,
            "FPLClient",
            lambda: _StubFPLClient(on_standings_call=lambda page: calls.append(page)),
        )
        client.get("/roundup/999")
        client.get("/roundup/999")
        assert len(calls) == 1

    def test_refresh_query_param_bypasses_the_cache(self, client, monkeypatch):
        calls: list[int] = []
        monkeypatch.setattr(
            api_main,
            "FPLClient",
            lambda: _StubFPLClient(on_standings_call=lambda page: calls.append(page)),
        )
        client.get("/roundup/999")
        client.get("/roundup/999?refresh=true")
        assert len(calls) == 2
