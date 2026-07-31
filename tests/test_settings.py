"""Tests for api/settings.py — the persistent (BUILD_PLAN 5.2) team-id/mini-league/planning-
horizon store, and its /settings API endpoints. Every test uses an isolated tmp-path SQLite
engine, never the real data_store/fpl.sqlite.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.settings import (
    DEFAULT_PLANNING_HORIZON_GAMEWEEKS,
    AppSettingsData,
    get_db_engine,
    get_settings,
    save_settings,
)
from engine.data import storage


@pytest.fixture
def db_engine(tmp_path):
    return storage.init_db(str(tmp_path / "test.sqlite"))


def test_get_settings_returns_defaults_when_nothing_saved(db_engine):
    settings = get_settings(db_engine)

    assert settings.fpl_team_id is None
    assert settings.mini_league_ids == ()
    assert settings.planning_horizon_gameweeks == DEFAULT_PLANNING_HORIZON_GAMEWEEKS


def test_save_and_get_settings_round_trips(db_engine):
    save_settings(
        db_engine,
        AppSettingsData(fpl_team_id=12345, mini_league_ids=(1, 2, 3), planning_horizon_gameweeks=3),
    )

    settings = get_settings(db_engine)

    assert settings.fpl_team_id == 12345
    assert settings.mini_league_ids == (1, 2, 3)
    assert settings.planning_horizon_gameweeks == 3


def test_save_settings_upserts_the_singleton_row(db_engine):
    save_settings(db_engine, AppSettingsData(1, (1,), 5))
    save_settings(db_engine, AppSettingsData(2, (2, 3), 4))

    settings = get_settings(db_engine)

    assert settings.fpl_team_id == 2
    assert settings.mini_league_ids == (2, 3)
    assert settings.planning_horizon_gameweeks == 4


def test_save_settings_handles_empty_mini_league_ids(db_engine):
    save_settings(db_engine, AppSettingsData(1, (), 5))

    assert get_settings(db_engine).mini_league_ids == ()


@pytest.fixture
def client(db_engine):
    app.dependency_overrides[get_db_engine] = lambda: db_engine
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_get_settings_endpoint_returns_defaults(client):
    response = client.get("/settings")

    assert response.status_code == 200
    body = response.json()
    assert body["fpl_team_id"] is None
    assert body["mini_league_ids"] == []
    assert body["planning_horizon_gameweeks"] == DEFAULT_PLANNING_HORIZON_GAMEWEEKS


def test_put_settings_endpoint_persists_and_returns_the_new_values(client):
    response = client.put(
        "/settings",
        json={
            "fpl_team_id": 999,
            "mini_league_ids": [10, 20],
            "planning_horizon_gameweeks": 4,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body == {
        "fpl_team_id": 999,
        "mini_league_ids": [10, 20],
        "planning_horizon_gameweeks": 4,
    }

    # And it's really persisted, not just echoed back.
    follow_up = client.get("/settings")
    assert follow_up.json() == body
