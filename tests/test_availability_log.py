"""Tests for the accumulating availability store (ENGINE_IMPROVEMENTS_5.md Tier 1.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.data.availability_log import (
    append_observations,
    attach_realised_minutes,
    build_availability_observations,
    load_observations,
    training_frame,
)


def _elements() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "status": ["a", "d", "i", "a"],
            "chance_of_playing_next_round": [None, 75.0, 0.0, 100.0],
            "news": ["", "Knock - 75% chance", "Hamstring injury", None],
            "selected_by_percent": [35.9, 4.2, 0.1, 12.0],
            "now_cost": [60, 80, 55, 95],
        }
    )


def test_build_observations_captures_the_signal_a_manager_would_have_seen():
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)

    observations = build_availability_observations(_elements(), gameweek=3, captured_at=captured_at)

    by_player = observations.set_index("player_id")
    # A null chance_of_playing means "no doubt reported", which FPL writes as null, not 100.
    assert by_player.loc[1, "chance_of_playing_next_round"] == 100.0
    assert by_player.loc[2, "chance_of_playing_next_round"] == 75.0
    assert by_player.loc[1, "status_score"] == 1.0
    assert by_player.loc[3, "status_score"] == 0.0
    # `has_news` is the structured "was anything reported" flag, not the free text itself.
    assert not by_player.loc[1, "has_news"]
    assert by_player.loc[2, "has_news"]
    assert not by_player.loc[4, "has_news"]  # None news is not news
    assert (observations["gameweek"] == 3).all()


def test_appending_accumulates_across_gameweeks(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    first = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    second = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)

    append_observations(
        build_availability_observations(_elements(), 3, first), 3, first, store_path=store
    )
    append_observations(
        build_availability_observations(_elements(), 4, second), 4, second, store_path=store
    )

    stored = load_observations(store)
    assert len(stored) == 8
    assert set(stored["gameweek"]) == {3, 4}


def test_appending_the_same_batch_twice_is_a_no_op(tmp_path: Path):
    # The projection build can legitimately re-run against the same snapshot; that must not
    # duplicate a gameweek's observations into the dataset the minutes model will later fit on.
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    observations = build_availability_observations(_elements(), 3, captured_at)

    append_observations(observations, 3, captured_at, store_path=store)
    second = append_observations(observations, 3, captured_at, store_path=store)

    assert second.n_rows == 0
    assert len(load_observations(store)) == 4


def test_a_later_capture_for_the_same_gameweek_is_kept_not_replaced(tmp_path: Path):
    # Team news moves through the week, and how it moved is signal. Two captures before the same
    # deadline must both survive.
    store = tmp_path / "obs.parquet"
    monday = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
    friday = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    append_observations(
        build_availability_observations(_elements(), 3, monday), 3, monday, store_path=store
    )

    improved = _elements()
    improved.loc[improved["id"] == 2, "chance_of_playing_next_round"] = 100.0
    append_observations(
        build_availability_observations(improved, 3, friday), 3, friday, store_path=store
    )

    stored = load_observations(store)
    player_2 = stored[stored["player_id"] == 2].sort_values("captured_at")
    assert list(player_2["chance_of_playing_next_round"]) == [75.0, 100.0]


def test_load_observations_before_anything_is_recorded_returns_empty(tmp_path: Path):
    assert load_observations(tmp_path / "nothing_yet.parquet").empty


def test_attach_realised_minutes_keeps_unresolved_gameweeks(tmp_path: Path):
    # A left join, so the most recent gameweek (not yet played) is not silently discarded.
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    observations = build_availability_observations(_elements(), 3, captured_at)
    realised = pd.DataFrame(
        {"player_id": [1, 2], "gameweek": [3, 3], "minutes": [90, 0], "starts": [1, 0]}
    )

    attached = attach_realised_minutes(observations, realised)

    assert len(attached) == 4
    by_player = attached.set_index("player_id")
    assert by_player.loc[1, "minutes"] == 90
    assert bool(by_player.loc[1, "started"])
    assert not bool(by_player.loc[2, "started"])
    assert pd.isna(by_player.loc[3, "minutes"])


def test_training_frame_keeps_the_last_pre_deadline_observation_per_player_gameweek(tmp_path: Path):
    """The row a manager would actually have acted on is the last one before the deadline."""
    store = tmp_path / "obs.parquet"
    monday = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
    friday = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    for stamp, chance in [(monday, 25.0), (friday, 100.0)]:
        elements = _elements()
        elements.loc[elements["id"] == 2, "chance_of_playing_next_round"] = chance
        frame = build_availability_observations(elements, 3, stamp)
        frame = attach_realised_minutes(
            frame, pd.DataFrame({"player_id": [2], "gameweek": [3], "minutes": [88], "starts": [1]})
        )
        append_observations(frame, 3, stamp, store_path=store)

    resolved = training_frame(store)

    player_2 = resolved[resolved["player_id"] == 2]
    assert len(player_2) == 1
    assert player_2["chance_of_playing_next_round"].iloc[0] == 100.0
    assert player_2["minutes"].iloc[0] == 88


def test_training_frame_is_empty_until_an_outcome_is_known(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    append_observations(
        build_availability_observations(_elements(), 3, captured_at),
        3,
        captured_at,
        store_path=store,
    )

    assert training_frame(store).empty
