"""Tests for engine.data.league_state_builder -- all HTTP is mocked via httpx.MockTransport, the
same convention tests/test_fpl_client.py uses; no network.
"""

from __future__ import annotations

import httpx
import pytest

from engine.data.fpl_client import FPLClient, FPLClientError
from engine.data.league_state_builder import build_league_snapshot


def _client(handler) -> FPLClient:
    return FPLClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _standings_page(results: list[dict], has_next: bool = False) -> dict:
    return {
        "league": {"id": 999, "name": "Sunday League"},
        "standings": {"has_next": has_next, "results": results},
    }


def _entry_result(entry_id: int, rank: int, **overrides) -> dict:
    """``entry`` is the real, usable FPL entry ID -- a live standings response also carries an
    ``id`` field that is a *different*, unrelated per-row identifier and never a usable entry ID
    (a real production bug: the module used to read ``id`` here by mistake). ``id`` is deliberately
    set to a decoy value distinct from ``entry_id`` so a regression back to reading ``id`` fails
    loudly (an unhandled URL, not a silently-passing test)."""
    base = {
        "id": entry_id + 900_000,
        "entry": entry_id,
        "player_name": f"Manager {entry_id}",
        "entry_name": f"Team {entry_id}",
        "rank": rank,
        "total": 300 + rank,
        "event_total": 60,
    }
    base.update(overrides)
    return base


def _picks_payload(picks: list[tuple[int, int]]) -> dict:
    return {
        "active_chip": None,
        "entry_history": {"event": 7, "points": 60, "bank": 5, "value": 1000},
        "picks": [
            {"element": player_id, "position": i + 1, "multiplier": multiplier}
            for i, (player_id, multiplier) in enumerate(picks)
        ],
    }


def test_single_page_league_returns_every_entry():
    results = [_entry_result(1, rank=1), _entry_result(2, rank=2)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            assert request.url.query == b"page_standings=1"
            return httpx.Response(200, json=_standings_page(results))
        if path == "/api/entry/1/":
            return httpx.Response(200, json={"id": 1, "current_event": 7})
        if path.startswith("/api/entry/") and path.endswith("/picks/"):
            return httpx.Response(200, json=_picks_payload([(10, 2), (11, 1), (12, 0)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    assert snapshot.league_id == 999
    assert snapshot.league_name == "Sunday League"
    assert snapshot.picks_gameweek == 7
    assert {entry.entry_id for entry in snapshot.entries} == {1, 2}


def test_picks_gameweek_resolves_from_probe_entrys_current_event():
    """MINI_LEAGUE_PLAN M1: before a deadline passes, ``current_event`` lags the app's own current
    gameweek -- the snapshot must reflect that lag rather than the requested gameweek, since it is
    resolved from FPL's own record of what picks actually exist, not asked for by the caller."""
    results = [_entry_result(1, rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page(results))
        if path == "/api/entry/1/":
            return httpx.Response(200, json={"id": 1, "current_event": 6})
        if path == "/api/entry/1/event/6/picks/":
            return httpx.Response(200, json=_picks_payload([(10, 1)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    assert snapshot.picks_gameweek == 6


def test_picks_map_keeps_every_pick_including_zero_multiplier_bench():
    results = [_entry_result(1, rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page(results))
        if path == "/api/entry/1/":
            return httpx.Response(200, json={"id": 1, "current_event": 7})
        if path.endswith("/picks/"):
            return httpx.Response(200, json=_picks_payload([(10, 2), (11, 0)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    entry = snapshot.entries[0]
    assert entry.picks == {10: 2, 11: 0}


def test_chips_are_carried_through_verbatim_including_unrecognised_names():
    results = [_entry_result(1, rank=1)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page(results))
        if path == "/api/entry/1/":
            return httpx.Response(200, json={"id": 1, "current_event": 7})
        if path.endswith("/picks/"):
            return httpx.Response(200, json=_picks_payload([(10, 1)]))
        if path.endswith("/history/"):
            return httpx.Response(
                200,
                json={
                    "current": [],
                    "past": [],
                    "chips": [
                        {"name": "bboost", "event": 5},
                        {"name": "some_future_chip", "event": 12},
                    ],
                },
            )
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    entry = snapshot.entries[0]
    assert entry.chips[0].name == "bboost"
    assert entry.chips[0].gameweek == 5
    assert entry.chips[1].name == "some_future_chip"


def test_pagination_stops_once_limit_is_reached():
    page_one = [_entry_result(i, rank=i) for i in range(1, 51)]
    page_two = [_entry_result(51, rank=51)]
    calls = {"standings_pages": []}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            page = int(request.url.params["page_standings"])
            calls["standings_pages"].append(page)
            if page == 1:
                return httpx.Response(200, json=_standings_page(page_one, has_next=True))
            return httpx.Response(200, json=_standings_page(page_two, has_next=False))
        if path.endswith("/picks/"):
            return httpx.Response(200, json=_picks_payload([(10, 1)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        if path.startswith("/api/entry/"):
            return httpx.Response(200, json={"id": 1, "current_event": 7})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999, limit=50)

    assert len(snapshot.entries) == 50
    assert 2 not in calls["standings_pages"]


def test_pagination_follows_has_next_across_pages_when_under_the_limit():
    page_one = [_entry_result(1, rank=1)]
    page_two = [_entry_result(2, rank=2)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            page = int(request.url.params["page_standings"])
            if page == 1:
                return httpx.Response(200, json=_standings_page(page_one, has_next=True))
            return httpx.Response(200, json=_standings_page(page_two, has_next=False))
        if path == "/api/entry/1/":
            return httpx.Response(200, json={"id": 1, "current_event": 7})
        if path.endswith("/picks/"):
            return httpx.Response(200, json=_picks_payload([(10, 1)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999, limit=50)

    assert {entry.entry_id for entry in snapshot.entries} == {1, 2}


def test_league_with_no_entries_returns_empty_snapshot_without_a_probe_call():
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page([], has_next=False))
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    assert snapshot.entries == ()
    assert snapshot.picks_gameweek == 0
    assert snapshot.league_name == "Sunday League"


def test_unknown_league_id_raises_fplclienterror_uncaught():
    """MINI_LEAGUE_PLAN M19: this module does not translate the error -- the API layer decides how
    to turn an unknown/private league into a caller-facing 400, matching
    engine.data.team_state_builder's own convention."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404)

    with pytest.raises(FPLClientError):
        build_league_snapshot(_client(handler), league_id=123456789)


def test_picks_gameweek_probe_falls_through_when_the_first_ranked_entry_is_inaccessible():
    """Regression: a real manager account can be deleted/banned/private and 404 on ``get_entry``
    even though it still appears in old standings. That must not take the whole league fetch down
    just because that entry happens to be ranked first -- the probe should fall through to the
    next entry instead."""
    results = [_entry_result(1, rank=1), _entry_result(2, rank=2)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page(results))
        if path == "/api/entry/1/":
            return httpx.Response(404)
        if path == "/api/entry/2/":
            return httpx.Response(200, json={"id": 2, "current_event": 7})
        if path.endswith("/picks/"):
            return httpx.Response(200, json=_picks_payload([(10, 1)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    assert snapshot.picks_gameweek == 7


def test_picks_gameweek_probe_raises_only_once_every_entry_has_failed():
    results = [_entry_result(1, rank=1), _entry_result(2, rank=2)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page(results))
        if path in ("/api/entry/1/", "/api/entry/2/"):
            return httpx.Response(404)
        raise AssertionError(f"unexpected path {path}")

    with pytest.raises(FPLClientError):
        build_league_snapshot(_client(handler), league_id=999)


def test_an_entry_whose_picks_cannot_be_fetched_is_dropped_not_fatal():
    """The same 'one inaccessible manager' failure, discovered later: this entry's ``current_event``
    probe would have worked, but its picks/history calls 404 anyway -- it should simply be absent
    from the snapshot rather than aborting every other rival's fetch."""
    results = [_entry_result(1, rank=1), _entry_result(2, rank=2)]

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/leagues-classic/999/standings/":
            return httpx.Response(200, json=_standings_page(results))
        if path == "/api/entry/1/":
            return httpx.Response(200, json={"id": 1, "current_event": 7})
        if path == "/api/entry/1/event/7/picks/":
            return httpx.Response(404)
        if path == "/api/entry/2/event/7/picks/":
            return httpx.Response(200, json=_picks_payload([(10, 1)]))
        if path.endswith("/history/"):
            return httpx.Response(200, json={"current": [], "past": [], "chips": []})
        raise AssertionError(f"unexpected path {path}")

    snapshot = build_league_snapshot(_client(handler), league_id=999)

    assert {entry.entry_id for entry in snapshot.entries} == {2}
