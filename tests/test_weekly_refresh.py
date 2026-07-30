"""Tests for scripts/weekly_refresh.py — Phase 6 orchestration sequencing, with every external
call (snapshot capture, prediction logging, odds pull, API state) faked. No network, no real
files written."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import scripts.weekly_refresh as weekly_refresh
from api.state import AppState
from backtest.prediction_log import PredictionLogEntry
from engine.data.snapshots import SnapshotManifest
from market_overlay.odds_client import OddsClientError

FAKE_SNAPSHOT = SnapshotManifest(
    season="2026-27", gameweek=1, captured_at=datetime.now(UTC), sources={}
)
FAKE_PREDICTIONS = pd.DataFrame({"player_id": [1, 2], "expected_points": [5.0, 4.0]})
FAKE_LOG_ENTRY = PredictionLogEntry(
    path=Path("logs/predictions/fake.parquet"),
    gameweek=1,
    model_version="test-version",
    logged_at=datetime.now(UTC),
)


class _FakeOddsClient:
    def __init__(self, *, should_fail: bool = False):
        self.should_fail = should_fail
        self.called = False

    def get_match_odds(self):
        self.called = True
        if self.should_fail:
            raise OddsClientError("provider unreachable")
        return [{"id": "evt1"}]


@pytest.fixture(autouse=True)
def _patch_orchestration_calls(monkeypatch):
    monkeypatch.setattr(weekly_refresh, "capture_current_gameweek", lambda *a, **k: FAKE_SNAPSHOT)
    monkeypatch.setattr(weekly_refresh, "log_predictions", lambda *a, **k: FAKE_LOG_ENTRY)
    monkeypatch.setattr(weekly_refresh, "current_model_version", lambda *a, **k: "test-version")


def _fake_app_state() -> AppState:
    return AppState(
        my_team=None,  # not exercised by this test -- run_weekly_refresh treats it opaquely
        projections={},
        team_id_by_player={},
        buy_prices={},
        fixtures=[],
        team_rates={},
        league_avg_xg_per_90=1.3,
        league_avg_xga_per_90=1.3,
        horizon_gameweeks=[1],
    )


def test_run_weekly_refresh_happy_path(monkeypatch):
    set_state_calls = []
    monkeypatch.setattr(weekly_refresh, "set_state", lambda state: set_state_calls.append(state))

    odds_client = _FakeOddsClient()
    app_state = _fake_app_state()

    result = weekly_refresh.run_weekly_refresh(
        fpl_client=object(),
        understat_client=object(),
        odds_client=odds_client,
        season="2026-27",
        understat_season_start_year=2026,
        gameweek=1,
        build_pool_projections=lambda snapshot, gameweek: FAKE_PREDICTIONS,
        build_app_state=lambda predictions: app_state,
    )

    assert result.snapshot is FAKE_SNAPSHOT
    assert result.predictions is FAKE_PREDICTIONS
    assert result.prediction_log_path == FAKE_LOG_ENTRY.path
    assert result.odds_pulled is True
    assert result.odds_error is None
    assert odds_client.called is True
    assert set_state_calls == [app_state]


def test_run_weekly_refresh_survives_odds_failure(monkeypatch):
    set_state_calls = []
    monkeypatch.setattr(weekly_refresh, "set_state", lambda state: set_state_calls.append(state))

    odds_client = _FakeOddsClient(should_fail=True)

    result = weekly_refresh.run_weekly_refresh(
        fpl_client=object(),
        understat_client=object(),
        odds_client=odds_client,
        season="2026-27",
        understat_season_start_year=2026,
        gameweek=1,
        build_pool_projections=lambda snapshot, gameweek: FAKE_PREDICTIONS,
        build_app_state=lambda predictions: _fake_app_state(),
    )

    assert result.odds_pulled is False
    assert result.odds_error == "provider unreachable"
    # Projections/logging/API-state refresh must still have happened -- odds are best-effort.
    assert result.predictions is FAKE_PREDICTIONS
    assert len(set_state_calls) == 1


def test_run_weekly_refresh_passes_snapshot_and_gameweek_to_pool_builder(monkeypatch):
    monkeypatch.setattr(weekly_refresh, "set_state", lambda state: None)
    received = {}

    def build_pool_projections(snapshot, gameweek):
        received["snapshot"] = snapshot
        received["gameweek"] = gameweek
        return FAKE_PREDICTIONS

    weekly_refresh.run_weekly_refresh(
        fpl_client=object(),
        understat_client=object(),
        odds_client=_FakeOddsClient(),
        season="2026-27",
        understat_season_start_year=2026,
        gameweek=7,
        build_pool_projections=build_pool_projections,
        build_app_state=lambda predictions: _fake_app_state(),
    )

    assert received["snapshot"] is FAKE_SNAPSHOT
    assert received["gameweek"] == 7


def test_main_refuses_to_run_without_wired_hooks():
    with pytest.raises(SystemExit):
        weekly_refresh.main(
            ["--season", "2026-27", "--understat-season-start-year", "2026", "--gameweek", "1"]
        )
