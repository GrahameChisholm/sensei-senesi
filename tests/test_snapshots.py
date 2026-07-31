"""Tests for engine.data.snapshots — capture, retry, fallback, and pre-deadline lookup."""

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from engine.data.snapshots import (
    SnapshotManifest,
    SourceCaptureResult,
    ValidationOutcome,
    capture_snapshot,
    get_predeadline_snapshot,
    list_snapshot_timestamps,
    load_manifest,
    load_snapshot_tables,
    prune_snapshot_history,
)

SEASON = "2025-26"
GW = 1
T0 = datetime(2026, 8, 1, 12, 0, 0, tzinfo=UTC)


def _ok_source(tables: dict[str, pd.DataFrame]):
    def fetch():
        return tables

    return fetch


def _always_fails_source():
    def fetch():
        raise RuntimeError("boom")

    return fetch


def test_capture_snapshot_writes_ok_source_and_manifest(tmp_path):
    tables = {"players": pd.DataFrame({"id": [1, 2], "name": ["a", "b"]})}
    manifest = capture_snapshot(
        SEASON, GW, sources={"fpl": _ok_source(tables)}, captured_at=T0, base_dir=tmp_path
    )
    assert manifest.sources["fpl"].status == "ok"
    assert manifest.sources["fpl"].tables == ["players"]
    assert manifest.all_ok_or_fallback

    loaded = load_manifest(tmp_path, SEASON, GW, T0)
    assert loaded.season == SEASON
    assert loaded.sources["fpl"].status == "ok"

    loaded_tables = load_snapshot_tables(tmp_path, SEASON, GW, T0, "fpl")
    pd.testing.assert_frame_equal(loaded_tables["players"], tables["players"])


def test_capture_snapshot_with_no_prior_and_failing_source_is_missing(tmp_path):
    manifest = capture_snapshot(
        SEASON, GW, sources={"fpl": _always_fails_source()}, captured_at=T0, base_dir=tmp_path
    )
    assert manifest.sources["fpl"].status == "missing"
    assert manifest.sources["fpl"].error is not None
    assert not manifest.all_ok_or_fallback


def test_capture_snapshot_falls_back_to_prior_snapshot_on_failure(tmp_path):
    tables = {"players": pd.DataFrame({"id": [1]})}
    capture_snapshot(
        SEASON, GW, sources={"fpl": _ok_source(tables)}, captured_at=T0, base_dir=tmp_path
    )

    t1 = T0 + timedelta(days=1)
    manifest = capture_snapshot(
        SEASON, GW, sources={"fpl": _always_fails_source()}, captured_at=t1, base_dir=tmp_path
    )
    assert manifest.sources["fpl"].status == "fallback"
    assert manifest.sources["fpl"].fallback_from == T0.isoformat()
    assert manifest.all_ok_or_fallback

    fallback_tables = load_snapshot_tables(tmp_path, SEASON, GW, t1, "fpl")
    pd.testing.assert_frame_equal(fallback_tables["players"], tables["players"])


def test_capture_snapshot_retries_before_falling_back(tmp_path):
    attempts = []

    def flaky_fetch():
        attempts.append(1)
        if len(attempts) < 2:
            raise RuntimeError("transient")
        return {"players": pd.DataFrame({"id": [1]})}

    manifest = capture_snapshot(
        SEASON, GW, sources={"fpl": flaky_fetch}, captured_at=T0, base_dir=tmp_path, retries=3
    )
    assert manifest.sources["fpl"].status == "ok"
    assert len(attempts) == 2


def test_capture_snapshot_treats_failed_validation_like_failed_fetch(tmp_path):
    tables = {"players": pd.DataFrame({"id": [1]})}

    def always_invalid(_tables):
        return ValidationOutcome(ok=False, reason="looks corrupted")

    manifest = capture_snapshot(
        SEASON,
        GW,
        sources={"fpl": _ok_source(tables)},
        validators={"fpl": always_invalid},
        captured_at=T0,
        base_dir=tmp_path,
        retries=2,
    )
    assert manifest.sources["fpl"].status == "missing"
    assert "looks corrupted" in manifest.sources["fpl"].error


def test_list_snapshot_timestamps_empty_when_nothing_captured(tmp_path):
    assert list_snapshot_timestamps(tmp_path, SEASON, GW) == []


def test_list_snapshot_timestamps_sorted_oldest_first(tmp_path):
    tables = {"players": pd.DataFrame({"id": [1]})}
    t1 = T0 + timedelta(days=1)
    t2 = T0 + timedelta(days=2)
    for ts in (t2, T0, t1):  # deliberately out of order
        capture_snapshot(
            SEASON, GW, sources={"fpl": _ok_source(tables)}, captured_at=ts, base_dir=tmp_path
        )

    timestamps = list_snapshot_timestamps(tmp_path, SEASON, GW)
    assert timestamps == [T0, t1, t2]


def test_get_predeadline_snapshot_picks_most_recent_before_deadline(tmp_path):
    tables = {"players": pd.DataFrame({"id": [1]})}
    t1 = T0 + timedelta(days=1)
    t2 = T0 + timedelta(days=2)
    for ts in (T0, t1, t2):
        capture_snapshot(
            SEASON, GW, sources={"fpl": _ok_source(tables)}, captured_at=ts, base_dir=tmp_path
        )

    deadline = T0 + timedelta(days=1, hours=12)
    assert get_predeadline_snapshot(tmp_path, SEASON, GW, deadline) == t1


def test_get_predeadline_snapshot_none_when_all_after_deadline(tmp_path):
    tables = {"players": pd.DataFrame({"id": [1]})}
    capture_snapshot(
        SEASON, GW, sources={"fpl": _ok_source(tables)}, captured_at=T0, base_dir=tmp_path
    )
    assert get_predeadline_snapshot(tmp_path, SEASON, GW, T0 - timedelta(days=1)) is None


def _capture_at(tmp_path, captured_at):
    tables = {"players": pd.DataFrame({"id": [1]})}
    capture_snapshot(
        SEASON, GW, sources={"fpl": _ok_source(tables)}, captured_at=captured_at, base_dir=tmp_path
    )


def test_prune_snapshot_history_keeps_only_the_latest_n(tmp_path):
    timestamps = [T0 + timedelta(days=i) for i in range(5)]
    for ts in timestamps:
        _capture_at(tmp_path, ts)

    deleted = prune_snapshot_history(tmp_path, SEASON, GW, keep_latest_n=2)

    assert deleted == timestamps[:3]
    assert list_snapshot_timestamps(tmp_path, SEASON, GW) == timestamps[3:]


def test_prune_snapshot_history_keeps_everything_when_under_the_limit(tmp_path):
    timestamps = [T0 + timedelta(days=i) for i in range(2)]
    for ts in timestamps:
        _capture_at(tmp_path, ts)

    deleted = prune_snapshot_history(tmp_path, SEASON, GW, keep_latest_n=5)

    assert deleted == []
    assert list_snapshot_timestamps(tmp_path, SEASON, GW) == timestamps


def test_prune_snapshot_history_keep_zero_deletes_everything(tmp_path):
    timestamps = [T0 + timedelta(days=i) for i in range(3)]
    for ts in timestamps:
        _capture_at(tmp_path, ts)

    deleted = prune_snapshot_history(tmp_path, SEASON, GW, keep_latest_n=0)

    assert deleted == timestamps
    assert list_snapshot_timestamps(tmp_path, SEASON, GW) == []


def test_prune_snapshot_history_rejects_negative_keep_latest_n(tmp_path):
    with pytest.raises(ValueError):
        prune_snapshot_history(tmp_path, SEASON, GW, keep_latest_n=-1)


def test_prune_snapshot_history_empty_when_nothing_captured(tmp_path):
    assert prune_snapshot_history(tmp_path, SEASON, GW, keep_latest_n=5) == []


def test_manifest_json_round_trip():
    manifest = SnapshotManifest(
        season=SEASON,
        gameweek=GW,
        captured_at=T0,
        sources={"fpl": SourceCaptureResult(status="ok", tables=["a"])},
    )
    restored = SnapshotManifest.from_json(manifest.to_json())
    assert restored.season == manifest.season
    assert restored.captured_at == manifest.captured_at
    assert restored.sources["fpl"].status == "ok"
