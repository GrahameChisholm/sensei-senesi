"""Tests for market_overlay/odds_client.py — all HTTP is mocked via httpx.MockTransport; no
network (BUILD_PLAN 4b.1)."""

from __future__ import annotations

import httpx
import pytest

from market_overlay.odds_client import (
    ANYTIME_SCORER_MARKET,
    AnytimeScorerOdds,
    MatchResultOdds,
    OddsClient,
    OddsClientError,
    parse_anytime_scorer_odds,
    parse_match_result_odds,
)

MATCH_EVENTS = [
    {
        "id": "evt1",
        "home_team": "Arsenal",
        "away_team": "Chelsea",
        "bookmakers": [
            {
                "key": "bookieA",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 1.8},
                            {"name": "Chelsea", "price": 4.5},
                            {"name": "Draw", "price": 3.6},
                        ],
                    }
                ],
            },
            {
                "key": "bookieB",
                "markets": [
                    {
                        "key": "h2h",
                        "outcomes": [
                            {"name": "Arsenal", "price": 2.0},
                            {"name": "Chelsea", "price": 4.3},
                            {"name": "Draw", "price": 3.4},
                        ],
                    }
                ],
            },
        ],
    },
    {
        "id": "evt2",
        "home_team": "Everton",
        "away_team": "Fulham",
        "bookmakers": [],  # not priced yet -- should be skipped, not raise
    },
]

ANYTIME_SCORER_EVENT = {
    "id": "evt1",
    "bookmakers": [
        {
            "key": "bookieA",
            "markets": [
                {
                    "key": ANYTIME_SCORER_MARKET,
                    "outcomes": [
                        {"name": "Bukayo Saka", "price": 2.5},
                        {"name": "Cole Palmer", "price": 3.0},
                    ],
                }
            ],
        }
    ],
}


def _client(handler) -> OddsClient:
    return OddsClient(
        api_key="test-key", client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def test_odds_client_requires_api_key_when_not_injected(monkeypatch):
    monkeypatch.delenv("ODDS_API_KEY", raising=False)
    with pytest.raises(OddsClientError):
        OddsClient()


def test_get_match_odds_hits_correct_path_and_params():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v4/sports/soccer_epl/odds"
        assert b"apiKey=test-key" in request.url.query
        assert b"markets=h2h%2Ctotals" in request.url.query
        return httpx.Response(200, json=MATCH_EVENTS)

    client = _client(handler)
    data = client.get_match_odds()
    assert len(data) == 2


def test_get_anytime_scorer_odds_hits_correct_path():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v4/sports/soccer_epl/events/evt1/odds"
        assert b"player_goal_scorer_anytime" in request.url.query
        return httpx.Response(200, json=ANYTIME_SCORER_EVENT)

    client = _client(handler)
    data = client.get_anytime_scorer_odds("evt1")
    assert data["id"] == "evt1"


def test_odds_client_raises_on_http_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    client = _client(handler)
    with pytest.raises(OddsClientError):
        client.get_match_odds()


def test_odds_client_raises_on_non_json_body():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = _client(handler)
    with pytest.raises(OddsClientError):
        client.get_match_odds()


# --- parsing --------------------------------------------------------------------------------


def test_parse_match_result_odds_averages_across_bookmakers():
    parsed = parse_match_result_odds(MATCH_EVENTS)
    assert len(parsed) == 1  # evt2 has no bookmakers, skipped
    result = parsed[0]
    assert result == MatchResultOdds(
        fixture_id="evt1",
        home_team="Arsenal",
        away_team="Chelsea",
        home_odds=pytest.approx(1.9),
        draw_odds=pytest.approx(3.5),
        away_odds=pytest.approx(4.4),
    )


def test_parse_match_result_odds_skips_fixtures_missing_a_market():
    events = [{"id": "evt3", "home_team": "A", "away_team": "B", "bookmakers": []}]
    assert parse_match_result_odds(events) == []


def test_parse_anytime_scorer_odds_returns_one_entry_per_player():
    parsed = parse_anytime_scorer_odds(ANYTIME_SCORER_EVENT)
    assert set(parsed) == {
        AnytimeScorerOdds(fixture_id="evt1", player_name="Bukayo Saka", odds=2.5),
        AnytimeScorerOdds(fixture_id="evt1", player_name="Cole Palmer", odds=3.0),
    }


def test_parse_anytime_scorer_odds_empty_when_market_unavailable():
    event = {"id": "evt9", "bookmakers": []}
    assert parse_anytime_scorer_odds(event) == []
