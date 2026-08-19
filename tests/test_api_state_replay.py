"""Tests for api.state's Season Replay additions: AppState.results, load_projection_cache's
sibling results.json lookup, and get_app_state's FPL_REPLAY_SEASON env override -- none of which
should change anything about the existing live 2026/27 path when unused."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

import api.state as state_module
from api.state import AppState, load_projection_cache


def _write_cache(path, season: str, gameweek: int = 1) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "season": season,
                "gameweek": gameweek,
                "horizon_gameweeks": [gameweek],
                "deadline_passed": False,
                "generated_at": "2025-08-01T00:00:00+00:00",
                "deadline_time": "2025-08-15T17:30:00+00:00",
                "model_version": "test",
                "projections": {},
                "players": {},
                "teams": {},
                "fixtures": [],
                "diagnostics": {},
            }
        )
    )


class TestResultsField:
    def test_none_when_no_sibling_results_file_exists(self, tmp_path):
        cache_path = tmp_path / "projections" / "2026-27" / "gw01.json"
        _write_cache(cache_path, "2026-27")
        results_dir = tmp_path / "replay"  # deliberately empty -- no 2026-27 dir under it

        state = load_projection_cache(cache_path, results_dir=results_dir)
        assert state.results is None

    def test_parsed_when_sibling_results_file_exists(self, tmp_path):
        cache_path = tmp_path / "projections" / "2025-26" / "gw01.json"
        _write_cache(cache_path, "2025-26")
        results_dir = tmp_path / "replay"
        results_path = results_dir / "2025-26" / "results.json"
        results_path.parent.mkdir(parents=True)
        results_path.write_text(json.dumps({"1": {"541": {"minutes": 90, "total_points": 6.0}}}))

        state = load_projection_cache(cache_path, results_dir=results_dir)
        assert state.results == {1: {541: {"minutes": 90, "total_points": 6.0}}}

    def test_defaults_to_none_when_constructed_directly(self):
        state = AppState(
            season="2026-27",
            gameweek=1,
            horizon_gameweeks=[1],
            deadline_passed=False,
            generated_at=datetime(2026, 8, 1, tzinfo=UTC),
            deadline_time=datetime(2026, 8, 15, tzinfo=UTC),
            model_version="test",
            projections={},
            players={},
            teams={},
            fixtures=[],
            diagnostics={},
        )
        assert state.results is None


class TestReplaySeasonEnvOverride:
    @pytest.fixture(autouse=True)
    def _reset(self):
        state_module.reset_state()
        yield
        state_module.reset_state()

    def test_env_var_selects_season_when_none_passed_explicitly(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "projections"
        _write_cache(cache_dir / "2025-26" / "gw01.json", "2025-26")
        _write_cache(cache_dir / "2026-27" / "gw01.json", "2026-27")
        monkeypatch.setenv("FPL_REPLAY_SEASON", "2025-26")

        state = state_module.get_app_state(cache_dir=cache_dir)
        assert state.season == "2025-26"

    def test_explicit_season_argument_still_wins_over_env_var(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / "projections"
        _write_cache(cache_dir / "2025-26" / "gw01.json", "2025-26")
        _write_cache(cache_dir / "2026-27" / "gw01.json", "2026-27")
        monkeypatch.setenv("FPL_REPLAY_SEASON", "2025-26")

        state = state_module.get_app_state(cache_dir=cache_dir, season="2026-27")
        assert state.season == "2026-27"

    def test_unset_env_var_keeps_existing_lexicographically_last_behaviour(
        self, tmp_path, monkeypatch
    ):
        cache_dir = tmp_path / "projections"
        _write_cache(cache_dir / "2025-26" / "gw01.json", "2025-26")
        _write_cache(cache_dir / "2026-27" / "gw01.json", "2026-27")
        monkeypatch.delenv("FPL_REPLAY_SEASON", raising=False)

        state = state_module.get_app_state(cache_dir=cache_dir)
        assert state.season == "2026-27"

    def test_env_var_starts_at_the_earliest_gameweek_not_the_latest(self, tmp_path, monkeypatch):
        # A replay season has every gameweek pre-built at once -- a fresh process must start the
        # user at GW1, unlike the live path's "most recently generated" semantics.
        cache_dir = tmp_path / "projections"
        _write_cache(cache_dir / "2025-26" / "gw01.json", "2025-26", gameweek=1)
        _write_cache(cache_dir / "2025-26" / "gw02.json", "2025-26", gameweek=2)
        _write_cache(cache_dir / "2025-26" / "gw38.json", "2025-26", gameweek=38)
        monkeypatch.setenv("FPL_REPLAY_SEASON", "2025-26")

        state = state_module.get_app_state(cache_dir=cache_dir)
        assert state.gameweek == 1
