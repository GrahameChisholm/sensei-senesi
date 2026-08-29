"""Tests for engine.data.fpl_client — all HTTP is mocked via httpx.MockTransport; no network."""

import json
from pathlib import Path

import httpx
import pytest

from engine.data.fpl_client import (
    FPLClient,
    FPLClientError,
    bootstrap_to_dataframes,
    fixtures_to_dataframe,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BOOTSTRAP = json.loads((FIXTURES_DIR / "fpl_bootstrap_static.json").read_text())
FIXTURES = json.loads((FIXTURES_DIR / "fpl_fixtures.json").read_text())


def _client(handler) -> FPLClient:
    return FPLClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def test_get_bootstrap_static_returns_parsed_json():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/bootstrap-static/"
        return httpx.Response(200, json=BOOTSTRAP)

    client = _client(handler)
    data = client.get_bootstrap_static()
    assert len(data["elements"]) == 30
    assert len(data["teams"]) == 20


def test_get_fixtures_without_event_hits_plain_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/fixtures/"
        assert request.url.query == b""
        return httpx.Response(200, json=FIXTURES)

    client = _client(handler)
    data = client.get_fixtures()
    assert len(data) == 10


def test_get_fixtures_with_event_filters_by_query_param():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/fixtures/"
        assert request.url.query == b"event=3"
        return httpx.Response(200, json=FIXTURES)

    client = _client(handler)
    client.get_fixtures(event=3)


def test_get_element_summary_hits_correct_path():
    summary = {"fixtures": [], "history": [], "history_past": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/element-summary/42/"
        return httpx.Response(200, json=summary)

    client = _client(handler)
    assert client.get_element_summary(42) == summary


def test_http_error_raises_fplclienterror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    client = _client(handler)
    with pytest.raises(FPLClientError):
        client.get_bootstrap_static()


def test_non_json_body_raises_fplclienterror():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = _client(handler)
    with pytest.raises(FPLClientError):
        client.get_bootstrap_static()


def test_iter_element_summaries_fetches_one_request_per_player():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, json={"fixtures": [], "history": [], "history_past": []})

    client = _client(handler)
    result = client.iter_element_summaries([1, 2, 3])
    assert set(result) == {1, 2, 3}
    assert calls == [
        "/api/element-summary/1/",
        "/api/element-summary/2/",
        "/api/element-summary/3/",
    ]


def test_get_entry_hits_correct_path():
    entry = {"id": 123, "name": "Test Team", "last_deadline_bank": 5, "last_deadline_value": 1000}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/entry/123/"
        return httpx.Response(200, json=entry)

    client = _client(handler)
    assert client.get_entry(123) == entry


def test_get_entry_picks_hits_correct_path():
    picks = {
        "active_chip": None,
        "entry_history": {"event": 5, "points": 60, "bank": 5, "value": 1000},
        "picks": [{"element": 1, "position": 1, "multiplier": 2, "is_captain": True}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/entry/123/event/5/picks/"
        return httpx.Response(200, json=picks)

    client = _client(handler)
    assert client.get_entry_picks(123, 5) == picks


def test_get_entry_transfers_hits_correct_path():
    transfers = [
        {
            "element_in": 2,
            "element_in_cost": 55,
            "element_out": 1,
            "element_out_cost": 50,
            "event": 3,
        }
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/entry/123/transfers/"
        return httpx.Response(200, json=transfers)

    client = _client(handler)
    assert client.get_entry_transfers(123) == transfers


def test_get_entry_history_hits_correct_path():
    history = {"current": [{"event": 1, "points": 60}], "past": [], "chips": []}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/entry/123/history/"
        return httpx.Response(200, json=history)

    client = _client(handler)
    assert client.get_entry_history(123) == history


def test_get_league_standings_defaults_to_page_one():
    standings = {
        "league": {"id": 999, "name": "Sunday League"},
        "standings": {"has_next": False, "results": [{"id": 1, "rank": 1, "total": 100}]},
    }

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/leagues-classic/999/standings/"
        assert request.url.query == b"page_standings=1"
        return httpx.Response(200, json=standings)

    client = _client(handler)
    assert client.get_league_standings(999) == standings


def test_get_league_standings_with_explicit_page():
    standings = {"league": {"id": 999}, "standings": {"has_next": False, "results": []}}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/leagues-classic/999/standings/"
        assert request.url.query == b"page_standings=2"
        return httpx.Response(200, json=standings)

    client = _client(handler)
    client.get_league_standings(999, page=2)


def test_bootstrap_to_dataframes_shapes():
    tables = bootstrap_to_dataframes(BOOTSTRAP)
    assert set(tables) == {"elements", "teams", "element_types", "events"}
    assert len(tables["elements"]) == 30
    assert len(tables["teams"]) == 20
    assert "defensive_contribution" in tables["elements"].columns


def test_fixtures_to_dataframe_shape():
    df = fixtures_to_dataframe(FIXTURES)
    assert len(df) == 10
    assert "team_h" in df.columns and "team_a" in df.columns
