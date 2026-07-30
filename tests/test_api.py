"""Tests for api/main.py — FastAPI endpoints over the demo state (BUILD_PLAN Phase 5.1)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.demo_data import load_demo_state
from api.main import app
from api.state import get_state
from features.team_state import compute_sell_price


@pytest.fixture
def client():
    state = load_demo_state()
    app.dependency_overrides[get_state] = lambda: state
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_health():
    with TestClient(app) as test_client:
        response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_team_returns_full_squad_and_sorted_chips(client):
    response = client.get("/team")
    assert response.status_code == 200
    body = response.json()
    assert len(body["squad"]) == 15
    assert body["chips_remaining"] == sorted(body["chips_remaining"])
    # Player 1 rose from 4.5m to 5.5m -> sell price banks half the profit, rounded down.
    player_1 = next(p for p in body["squad"] if p["player_id"] == 1)
    assert player_1["sell_price"] == compute_sell_price(45, 55)


def test_get_fixtures_filters_by_gameweek(client):
    response = client.get("/fixtures", params={"gameweek": 4})
    assert response.status_code == 200
    body = response.json()
    # GW4 only has the 1v2 match in the demo schedule -> exactly 2 rows (one per team).
    assert len(body) == 2
    assert {row["team_id"] for row in body} == {1, 2}


def test_get_fixtures_without_gameweek_returns_every_row(client):
    response = client.get("/fixtures")
    assert response.status_code == 200
    assert len(response.json()) == 2 * 9  # 9 matches in the demo schedule, 2 rows each


def test_get_captaincy_ranks_full_pool(client):
    response = client.get("/captaincy", params={"gameweek": 1})
    assert response.status_code == 200
    body = response.json()
    assert len(body["ranked_pool"]) == 40  # 15 squad + 25 pool
    assert body["top_ev_pick"] is not None
    assert body["top_ev_pick"]["is_eligible"] is True


def test_get_captaincy_rejects_unknown_gameweek(client):
    response = client.get("/captaincy", params={"gameweek": 99})
    assert response.status_code == 400


def test_get_transfers_recommends_forced_sell_for_injured_player(client):
    response = client.get("/transfers")
    assert response.status_code == 200
    body = response.json()
    assert body["recommended"] is not None
    assert body["recommended"]["is_forced"] is True
    assert body["recommended"]["sell_player_id"] == 5  # the demo's deliberately-injured player


def test_get_bench_boost_returns_a_verdict(client):
    response = client.get("/chips/bench-boost", params={"gameweek": 1})
    assert response.status_code == 200
    assert response.json()["recommendation"] in {"play_now", "wait"}


def test_get_triple_captain_returns_a_verdict(client):
    response = client.get("/chips/triple-captain", params={"gameweek": 1})
    assert response.status_code == 200
    assert response.json()["recommendation"] in {"play_now", "wait"}


def test_get_free_hit_rejects_blocked_gameweek(client):
    response = client.get("/chips/free-hit", params={"gameweek": 1})
    assert response.status_code == 400


def test_get_free_hit_recommends_the_deliberate_blank_gameweek(client):
    # GW4 is the demo schedule's deliberate blank for teams 3 & 4 -> worst exposure in horizon.
    response = client.get("/chips/free-hit", params={"gameweek": 4})
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"] == "play_now"
    assert body["best_gameweek"] == 4

    response_elsewhere = client.get("/chips/free-hit", params={"gameweek": 2})
    assert response_elsewhere.json()["recommendation"] == "wait"
    assert response_elsewhere.json()["best_gameweek"] == 4


def test_get_wildcard_returns_a_verdict(client):
    response = client.get("/chips/wildcard")
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation"] in {"play_now", "hold"}
    assert body["squad_uplift"] >= 0.0
    assert body["upgradeable_slots"] >= 0
