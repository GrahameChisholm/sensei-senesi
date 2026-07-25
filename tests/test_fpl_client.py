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
