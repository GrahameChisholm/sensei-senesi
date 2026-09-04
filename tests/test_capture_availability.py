"""Tests for scripts/capture_availability.py -- all HTTP is mocked via httpx.MockTransport, no
network, mirroring tests/test_fpl_client.py's own convention."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd
import pytest

from engine.data.availability_log import load_observations
from engine.data.fpl_client import FPLClient
from scripts.capture_availability import capture_availability, resolve_gameweek

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BOOTSTRAP = json.loads((FIXTURES_DIR / "fpl_bootstrap_static.json").read_text())


def _client(handler) -> FPLClient:
    return FPLClient(client=httpx.Client(transport=httpx.MockTransport(handler)))


def _bootstrap_handler(request: httpx.Request) -> httpx.Response:
    assert request.url.path == "/api/bootstrap-static/"
    return httpx.Response(200, json=BOOTSTRAP)


def test_resolve_gameweek_picks_the_is_next_event():
    events = pd.DataFrame(BOOTSTRAP["events"])
    assert resolve_gameweek(events) == 1  # fixture's GW1 carries is_next=True


def test_resolve_gameweek_override_wins_over_is_next():
    events = pd.DataFrame(BOOTSTRAP["events"])
    assert resolve_gameweek(events, override=7) == 7


def test_resolve_gameweek_raises_without_is_next_or_override():
    events = pd.DataFrame(BOOTSTRAP["events"]).assign(is_next=False)
    with pytest.raises(ValueError, match="is_next"):
        resolve_gameweek(events)


def test_capture_availability_appends_one_batch(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    client = _client(_bootstrap_handler)

    gameweek, n_rows = capture_availability(client, "2026-27", store_path=store)

    assert gameweek == 1
    assert n_rows == len(BOOTSTRAP["elements"])
    stored = load_observations(store)
    assert len(stored) == len(BOOTSTRAP["elements"])
    assert set(stored["season"]) == {"2026-27"}


def test_capture_availability_respects_the_gameweek_override(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    client = _client(_bootstrap_handler)

    gameweek, _ = capture_availability(client, "2026-27", gameweek=5, store_path=store)

    assert gameweek == 5


def test_capture_availability_a_repeat_run_is_a_no_op(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    capture_availability(
        _client(_bootstrap_handler), "2026-27", store_path=store, captured_at=captured_at
    )

    _, n_rows = capture_availability(
        _client(_bootstrap_handler), "2026-27", store_path=store, captured_at=captured_at
    )

    assert n_rows == 0
    assert len(load_observations(store)) == len(BOOTSTRAP["elements"])
