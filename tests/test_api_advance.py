"""Tests for POST /squad/advance -- Season Replay's "advance to the next gameweek" endpoint: score
the committed squad against the current gameweek's real result, log it, and roll the app state on
to the next gameweek's cache.

Uses season "2099-00" throughout, deliberately: it can never collide with a real
data_store/replay/{season}/results.json this repo might actually have on disk (unlike e.g.
"2025-26", which this repo's own Season Replay batch job really does populate), so
load_projection_cache's real, un-mocked results lookup stays reliably None for these fixtures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.state as state_module
from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID

SEASON = "2099-00"
GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]
ALL_IDS = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]


def _position_for(player_id: int) -> str:
    if player_id in (GK1, GK2):
        return GK
    if player_id in DEF_IDS:
        return DEF
    if player_id in MID_IDS:
        return MID
    return FWD


def _horizon(player_id: int, gameweeks: tuple[int, ...]):
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
        goals=2.0,
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


def _app_state_for_gameweek(gameweek: int, results):
    from api.state import AppState

    players = {}
    teams = {}
    for index, player_id in enumerate(ALL_IDS):
        team_id = 100 + (index // 3)
        teams.setdefault(team_id, {"name": f"Team {team_id}", "short_name": f"T{team_id}"})
        players[player_id] = {
            "web_name": f"Player{player_id}",
            "full_name": f"Player {player_id}",
            "team_id": team_id,
            "position": _position_for(player_id),
            "price": 40,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
        }
    projections = {pid: _horizon(pid, (gameweek,)) for pid in ALL_IDS}

    return AppState(
        season=SEASON,
        gameweek=gameweek,
        horizon_gameweeks=[gameweek],
        deadline_passed=False,
        generated_at=datetime(2025, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=datetime(2025, 8, 21, 17, 30, tzinfo=UTC),
        model_version="test-replay",
        projections=projections,
        players=players,
        teams=teams,
        fixtures=[],
        diagnostics={},
        results=results,
    )


def _all_played_90(points: float = 4.0) -> dict:
    return {pid: {"minutes": 90, "total_points": points} for pid in ALL_IDS}


def _write_minimal_cache(path, gameweek: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "season": SEASON,
                "gameweek": gameweek,
                "horizon_gameweeks": [gameweek],
                "deadline_passed": False,
                "generated_at": "2025-08-01T00:00:00+00:00",
                "deadline_time": "2025-08-15T17:30:00+00:00",
                "model_version": "test-replay",
                "projections": {},
                "players": {},
                "teams": {},
                "fixtures": [],
                "diagnostics": {},
            }
        )
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    results = {1: _all_played_90(4.0), 2: _all_played_90(3.0)}
    state_module.set_app_state(_app_state_for_gameweek(1, results))
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state_module.reset_state()


def _build_and_confirm(client: TestClient) -> dict:
    for player_id in ALL_IDS:
        response = client.post(
            "/squad/build/players",
            json={"player_id": player_id, "position": _position_for(player_id), "price": 40},
        )
        assert response.status_code == 200, response.json()
    starting_xi = [GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]]
    bench_order = [DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2]
    response = client.post(
        "/squad/build/confirm",
        json={
            "player_ids": ALL_IDS,
            "starting_xi": starting_xi,
            "bench_order": bench_order,
            "captain_id": MID_IDS[0],
            "vice_captain_id": MID_IDS[1],
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


class TestAdvanceHappyPath:
    def test_scores_the_gameweek_and_moves_to_the_next_one(self, client, tmp_path, monkeypatch):
        import api.main as main_module

        cache_dir = tmp_path / "projections"
        monkeypatch.setattr(main_module, "DEFAULT_PROJECTION_CACHE_DIR", cache_dir)
        _write_minimal_cache(cache_dir / SEASON / "gw02.json", 2)

        _build_and_confirm(client)
        response = client.post("/squad/advance")
        assert response.status_code == 200, response.json()
        body = response.json()

        # 11 starters at 4.0 + captain (MID_IDS[0]) doubled once more (+4.0), no hits, no chip.
        assert body["points"] == pytest.approx(11 * 4.0 + 4.0)
        assert body["running_total"] == pytest.approx(body["points"])
        assert body["gameweek"] == 1
        assert body["season_complete"] is False
        assert len(body["season_log"]) == 1
        assert body["season_log"][0]["running_total"] == pytest.approx(body["points"])
        # The app has moved on: /squad/advance already swapped the process-wide app state.
        gw_response = client.get("/gameweek")
        assert gw_response.json()["gameweek"] == 2

    def test_season_complete_when_no_next_gameweek_cache_exists(
        self, client, tmp_path, monkeypatch
    ):
        import api.main as main_module

        monkeypatch.setattr(main_module, "DEFAULT_PROJECTION_CACHE_DIR", tmp_path / "nonexistent")

        _build_and_confirm(client)
        response = client.post("/squad/advance")
        assert response.status_code == 200, response.json()
        body = response.json()
        assert body["season_complete"] is True
        # gameweek stays put -- nothing to advance to.
        gw_response = client.get("/gameweek")
        assert gw_response.json()["gameweek"] == 1

    def test_running_total_builds_on_an_existing_season_log(self, client, tmp_path, monkeypatch):
        import api.main as main_module

        monkeypatch.setattr(main_module, "DEFAULT_PROJECTION_CACHE_DIR", tmp_path / "nonexistent")
        _build_and_confirm(client)
        state_module.set_season_log(
            [{"gameweek": 0, "points": 50.0, "running_total": 50.0, "chip_played": None}]
        )

        response = client.post("/squad/advance")
        assert response.status_code == 200, response.json()
        body = response.json()
        assert body["running_total"] == pytest.approx(50.0 + body["points"])
        assert len(body["season_log"]) == 2


class TestAdvanceRejections:
    def test_no_committed_squad_yet(self, client):
        response = client.post("/squad/advance")
        assert response.status_code == 400

    def test_open_draft_is_rejected(self, client):
        _build_and_confirm(client)
        open_response = client.post("/squad/draft")
        assert open_response.status_code == 200
        response = client.post("/squad/advance")
        assert response.status_code == 400
        assert "draft" in response.json()["detail"].lower()

    def test_no_results_means_not_a_replay_season(self, client):
        state_module.set_app_state(_app_state_for_gameweek(1, results=None))
        _build_and_confirm(client)
        response = client.post("/squad/advance")
        assert response.status_code == 400
        assert "results" in response.json()["detail"].lower()
