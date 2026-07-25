"""Tests for immutable prediction logging tagged with model version (3.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from backtest.prediction_log import current_model_version, load_logged_predictions, log_predictions


def test_current_model_version_returns_non_empty_string():
    # This repo is a git repo (per the session's environment), so this should resolve to a real
    # tag/hash rather than falling back to "unknown" -- but either way it must be a usable string.
    version = current_model_version()
    assert isinstance(version, str)
    assert version != ""


def test_current_model_version_falls_back_to_unknown_outside_a_git_repo(tmp_path: Path):
    version = current_model_version(repo_root=tmp_path)
    assert version == "unknown"


def test_log_predictions_writes_immutable_parquet_and_round_trips(tmp_path: Path):
    predictions = pd.DataFrame({"player_id": [1, 2], "expected_points": [5.0, 3.0]})
    logged_at = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)

    entry = log_predictions(
        predictions, gameweek=3, model_version="engine-v0.3", logged_at=logged_at, log_dir=tmp_path
    )

    assert entry.path.exists()
    assert entry.gameweek == 3
    assert entry.model_version == "engine-v0.3"

    loaded = load_logged_predictions(log_dir=tmp_path)
    assert set(loaded["player_id"]) == {1, 2}
    assert (loaded["gameweek"] == 3).all()
    assert (loaded["model_version"] == "engine-v0.3").all()


def test_log_predictions_refuses_to_overwrite_existing_log(tmp_path: Path):
    predictions = pd.DataFrame({"player_id": [1], "expected_points": [5.0]})
    logged_at = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)

    log_predictions(
        predictions, gameweek=1, model_version="v1", logged_at=logged_at, log_dir=tmp_path
    )

    with pytest.raises(FileExistsError):
        log_predictions(
            predictions, gameweek=1, model_version="v1", logged_at=logged_at, log_dir=tmp_path
        )


def test_load_logged_predictions_filters_by_gameweek_and_model_version(tmp_path: Path):
    log_predictions(
        pd.DataFrame({"player_id": [1], "expected_points": [1.0]}),
        gameweek=1,
        model_version="v1",
        logged_at=datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC),
        log_dir=tmp_path,
    )
    log_predictions(
        pd.DataFrame({"player_id": [2], "expected_points": [2.0]}),
        gameweek=2,
        model_version="v2",
        logged_at=datetime(2026, 8, 2, 12, 0, 0, tzinfo=UTC),
        log_dir=tmp_path,
    )

    only_gw1 = load_logged_predictions(log_dir=tmp_path, gameweek=1)
    assert list(only_gw1["player_id"]) == [1]

    only_v2 = load_logged_predictions(log_dir=tmp_path, model_version="v2")
    assert list(only_v2["player_id"]) == [2]


def test_load_logged_predictions_empty_dir_returns_empty_frame(tmp_path: Path):
    empty_dir = tmp_path / "does_not_exist_yet"
    result = load_logged_predictions(log_dir=empty_dir)
    assert result.empty
