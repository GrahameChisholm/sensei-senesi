"""Tests for engine.data.understat_client — all HTTP is mocked; no network."""

import json
from pathlib import Path

import httpx
import pytest

from engine.data.understat_client import (
    DEFAULT_HEADERS,
    UnderstatClient,
    UnderstatClientError,
    league_data_to_dataframes,
    player_data_to_dataframe,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LEAGUE_DATA = json.loads((FIXTURES_DIR / "understat_league_data.json").read_text())
PLAYER_DATA = json.loads((FIXTURES_DIR / "understat_player_data.json").read_text())


def _client(handler, headers: dict | None = None) -> UnderstatClient:
    return UnderstatClient(
        client=httpx.Client(transport=httpx.MockTransport(handler), headers=headers or {})
    )


def test_get_league_data_hits_correct_path_and_sends_browser_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/getLeagueData/EPL/2023"
        assert "Chrome" in request.headers["User-Agent"]
        assert request.headers["X-Requested-With"] == "XMLHttpRequest"
        return httpx.Response(200, json=LEAGUE_DATA)

    # exercise the real default headers (production behaviour), not the bare test client
    client = _client(handler, headers=DEFAULT_HEADERS)
    data = client.get_league_data(2023)
    assert set(data) == {"teams", "players", "dates"}


def test_get_league_data_custom_league():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/getLeagueData/La_liga/2023"
        return httpx.Response(200, json=LEAGUE_DATA)

    client = _client(handler)
    client.get_league_data(2023, league="La_liga")


def test_get_league_data_missing_key_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"teams": {}, "players": []})  # no 'dates'

    client = _client(handler)
    with pytest.raises(UnderstatClientError):
        client.get_league_data(2023)


def test_empty_body_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={})

    client = _client(handler)
    with pytest.raises(UnderstatClientError):
        client.get_league_data(2023)


def test_get_player_data_hits_correct_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/getPlayerData/1250"
        return httpx.Response(200, json=PLAYER_DATA)

    client = _client(handler)
    data = client.get_player_data(1250)
    assert len(data["matches"]) == 10


def test_get_player_data_missing_matches_key_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"player": {"id": "1250"}})

    client = _client(handler)
    with pytest.raises(UnderstatClientError):
        client.get_player_data(1250)


def test_http_error_raises_understatclienterror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    client = _client(handler)
    with pytest.raises(UnderstatClientError):
        client.get_league_data(2023)


def test_league_data_to_dataframes_shapes():
    tables = league_data_to_dataframes(LEAGUE_DATA)
    assert set(tables) == {"players", "teams_history", "dates"}
    assert len(tables["players"]) == 50
    assert "xG" in tables["players"].columns
    assert {"team_id", "team_title", "xG", "xGA"}.issubset(tables["teams_history"].columns)
    assert len(tables["dates"]) == 20


def test_player_data_to_dataframe_shape():
    df = player_data_to_dataframe(PLAYER_DATA)
    assert len(df) == 10
    assert "xG" in df.columns and "xA" in df.columns
