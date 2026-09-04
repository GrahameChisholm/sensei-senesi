"""Tests for the accumulating availability store (ENGINE_IMPROVEMENTS_5.md Tier 1.1)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.data.availability_log import (
    LEGACY_SEASON,
    OBSERVATION_COLUMNS,
    append_observations,
    attach_realised_minutes,
    build_availability_observations,
    load_observations,
    realised_from_player_history,
    training_frame,
)
from engine.data.player_history import PlayerGameweekActual

SEASON = "2026-27"


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


def _actual(gameweek: int, minutes: int, starts: int | None = None) -> PlayerGameweekActual:
    return PlayerGameweekActual(
        gameweek=gameweek,
        minutes=minutes,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        saves=0,
        yellow_cards=0,
        red_cards=0,
        bonus=0,
        defensive_contribution=0,
        total_points=0,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        expected_goals_conceded=0.0,
        starts=starts,
    )


def test_build_observations_captures_the_signal_a_manager_would_have_seen():
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)

    observations = build_availability_observations(
        _elements(), SEASON, gameweek=3, captured_at=captured_at
    )

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
    assert (observations["season"] == SEASON).all()


def test_appending_accumulates_across_gameweeks(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    first = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    second = datetime(2026, 8, 27, 10, 0, tzinfo=UTC)

    append_observations(
        build_availability_observations(_elements(), SEASON, 3, first),
        SEASON,
        3,
        first,
        store_path=store,
    )
    append_observations(
        build_availability_observations(_elements(), SEASON, 4, second),
        SEASON,
        4,
        second,
        store_path=store,
    )

    stored = load_observations(store)
    assert len(stored) == 8
    assert set(stored["gameweek"]) == {3, 4}


def test_appending_the_same_batch_twice_is_a_no_op(tmp_path: Path):
    # The projection build can legitimately re-run against the same snapshot; that must not
    # duplicate a gameweek's observations into the dataset the minutes model will later fit on.
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    observations = build_availability_observations(_elements(), SEASON, 3, captured_at)

    append_observations(observations, SEASON, 3, captured_at, store_path=store)
    second = append_observations(observations, SEASON, 3, captured_at, store_path=store)

    assert second.n_rows == 0
    assert len(load_observations(store)) == 4


def test_a_batch_appended_twice_with_differing_microsecond_precision_is_a_no_op(tmp_path: Path):
    # `--reuse-snapshot` parses a second-precision directory name; a fresh capture uses
    # `datetime.now(UTC)`, which carries microseconds. Both name the same real capture and must
    # collapse to one batch rather than appending twice.
    store = tmp_path / "obs.parquet"
    second_precision = datetime(2026, 8, 29, 13, 3, 23, tzinfo=UTC)
    microsecond_precision = second_precision.replace(microsecond=225867)

    append_observations(
        build_availability_observations(_elements(), SEASON, 3, second_precision),
        SEASON,
        3,
        second_precision,
        store_path=store,
    )
    second = append_observations(
        build_availability_observations(_elements(), SEASON, 3, microsecond_precision),
        SEASON,
        3,
        microsecond_precision,
        store_path=store,
    )

    assert second.n_rows == 0
    assert len(load_observations(store)) == 4


def test_a_later_capture_for_the_same_gameweek_is_kept_not_replaced(tmp_path: Path):
    # Team news moves through the week, and how it moved is signal. Two captures before the same
    # deadline must both survive.
    store = tmp_path / "obs.parquet"
    monday = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
    friday = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    append_observations(
        build_availability_observations(_elements(), SEASON, 3, monday),
        SEASON,
        3,
        monday,
        store_path=store,
    )

    improved = _elements()
    improved.loc[improved["id"] == 2, "chance_of_playing_next_round"] = 100.0
    append_observations(
        build_availability_observations(improved, SEASON, 3, friday),
        SEASON,
        3,
        friday,
        store_path=store,
    )

    stored = load_observations(store)
    player_2 = stored[stored["player_id"] == 2].sort_values("captured_at")
    assert list(player_2["chance_of_playing_next_round"]) == [75.0, 100.0]


def test_load_observations_before_anything_is_recorded_returns_empty(tmp_path: Path):
    assert load_observations(tmp_path / "nothing_yet.parquet").empty


def test_load_observations_collapses_a_preexisting_duplicate_batch(tmp_path: Path):
    # Simulates a store written before the microsecond-precision bug was fixed: the same batch
    # twice, differing only in captured_at's precision.
    store = tmp_path / "obs.parquet"
    row = build_availability_observations(
        _elements(), SEASON, 3, datetime(2026, 8, 29, 13, 3, 23, tzinfo=UTC)
    )
    duplicate = row.copy()
    duplicate["captured_at"] = "2026-08-29T13:03:23.225867+00:00"
    pd.concat([row, duplicate], ignore_index=True).to_parquet(store, index=False)

    loaded = load_observations(store)

    assert len(loaded) == len(row)


def test_load_observations_backfills_legacy_season(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    legacy_columns = [c for c in OBSERVATION_COLUMNS if c != "season"]
    row = build_availability_observations(
        _elements(), SEASON, 3, datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    ).drop(columns=["season"])
    row[legacy_columns].to_parquet(store, index=False)

    loaded = load_observations(store)

    assert (loaded["season"] == LEGACY_SEASON).all()


def test_the_same_gameweek_in_two_seasons_does_not_collide(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)

    append_observations(
        build_availability_observations(_elements(), "2026-27", 3, captured_at),
        "2026-27",
        3,
        captured_at,
        store_path=store,
    )
    second = append_observations(
        build_availability_observations(_elements(), "2027-28", 3, captured_at),
        "2027-28",
        3,
        captured_at,
        store_path=store,
    )

    assert second.n_rows == 4  # not treated as a duplicate of the 2026-27 batch
    stored = load_observations(store)
    assert set(stored["season"]) == {"2026-27", "2027-28"}
    assert len(stored) == 8


def test_attach_realised_minutes_keeps_unresolved_gameweeks(tmp_path: Path):
    # A left join, so the most recent gameweek (not yet played) is not silently discarded.
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    observations = build_availability_observations(_elements(), SEASON, 3, captured_at)
    realised = pd.DataFrame(
        {
            "season": [SEASON, SEASON],
            "player_id": [1, 2],
            "gameweek": [3, 3],
            "minutes": [90, 0],
            "starts": [1, 0],
        }
    )

    attached = attach_realised_minutes(observations, realised)

    assert len(attached) == 4
    by_player = attached.set_index("player_id")
    assert by_player.loc[1, "minutes"] == 90
    assert bool(by_player.loc[1, "started"])
    assert not bool(by_player.loc[2, "started"])
    assert pd.isna(by_player.loc[3, "minutes"])


def test_realised_from_player_history_maps_actuals_to_the_join_shape():
    history = {
        1: [_actual(gameweek=2, minutes=90, starts=1), _actual(gameweek=3, minutes=0, starts=0)],
        2: [],  # a player with no played gameweeks yet is a real, empty case, not an error
    }

    realised = realised_from_player_history(history, SEASON)

    assert list(realised.columns) == ["season", "player_id", "gameweek", "minutes", "starts"]
    assert len(realised) == 2
    row = realised[(realised["player_id"] == 1) & (realised["gameweek"] == 2)].iloc[0]
    assert row["season"] == SEASON
    assert row["minutes"] == 90
    assert row["starts"] == 1


def test_training_frame_keeps_the_last_pre_deadline_observation_per_player_gameweek(tmp_path: Path):
    """The row a manager would actually have acted on is the last one before the deadline."""
    store = tmp_path / "obs.parquet"
    monday = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
    friday = datetime(2026, 8, 21, 10, 0, tzinfo=UTC)
    for stamp, chance in [(monday, 25.0), (friday, 100.0)]:
        elements = _elements()
        elements.loc[elements["id"] == 2, "chance_of_playing_next_round"] = chance
        frame = build_availability_observations(elements, SEASON, 3, stamp)
        frame = attach_realised_minutes(
            frame,
            pd.DataFrame(
                {
                    "season": [SEASON],
                    "player_id": [2],
                    "gameweek": [3],
                    "minutes": [88],
                    "starts": [1],
                }
            ),
        )
        append_observations(frame, SEASON, 3, stamp, store_path=store)

    # Pre-attached before appending, so the fallback (no `realised` argument) path is exercised.
    resolved = training_frame(store)

    player_2 = resolved[resolved["player_id"] == 2]
    assert len(player_2) == 1
    assert player_2["chance_of_playing_next_round"].iloc[0] == 100.0
    assert player_2["minutes"].iloc[0] == 88


def test_training_frame_returns_the_real_last_row_not_a_column_wise_merge(tmp_path: Path):
    """`GroupBy.last()` takes the last *non-null* value per column independently and can
    therefore assemble a row that never existed. `drop_duplicates(keep="last")` must not do that:
    the real last capture here has a null `chance_of_playing_next_round`, and the resolved row
    must keep that null rather than silently backfilling it from the earlier capture's 75.0.
    """
    store = tmp_path / "obs.parquet"
    columns = [*OBSERVATION_COLUMNS, "minutes", "started"]
    early = pd.DataFrame(
        [
            {
                "season": SEASON,
                "player_id": 1,
                "gameweek": 3,
                "captured_at": "2026-08-17T10:00:00+00:00",
                "status": "d",
                "status_score": 0.75,
                "chance_of_playing_next_round": 75.0,
                "has_news": True,
                "selected_by_percent": 10.0,
                "now_cost": 60,
                "minutes": 90,
                "started": True,
            }
        ],
        columns=columns,
    )
    late = pd.DataFrame(
        [
            {
                "season": SEASON,
                "player_id": 1,
                "gameweek": 3,
                "captured_at": "2026-08-21T10:00:00+00:00",
                "status": "a",
                "status_score": 1.0,
                "chance_of_playing_next_round": pd.NA,
                "has_news": False,
                "selected_by_percent": 10.0,
                "now_cost": 60,
                "minutes": 90,
                "started": True,
            }
        ],
        columns=columns,
    )
    pd.concat([early, late], ignore_index=True).to_parquet(store, index=False)

    resolved = training_frame(store)

    assert len(resolved) == 1
    row = resolved.iloc[0]
    assert row["status"] == "a"
    assert pd.isna(row["chance_of_playing_next_round"])


def test_training_frame_resolves_rows_from_a_signal_only_store(tmp_path: Path):
    # The store never carries outcomes itself (see the module docstring); a caller joins them on
    # at read time via `realised`.
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    append_observations(
        build_availability_observations(_elements(), SEASON, 3, captured_at),
        SEASON,
        3,
        captured_at,
        store_path=store,
    )
    history = {1: [_actual(gameweek=3, minutes=90, starts=1)]}
    realised = realised_from_player_history(history, SEASON)

    resolved = training_frame(store, realised=realised)

    assert len(resolved) == 1
    assert resolved.iloc[0]["player_id"] == 1
    assert resolved.iloc[0]["minutes"] == 90


def test_training_frame_is_empty_until_an_outcome_is_known(tmp_path: Path):
    store = tmp_path / "obs.parquet"
    captured_at = datetime(2026, 8, 20, 10, 0, tzinfo=UTC)
    append_observations(
        build_availability_observations(_elements(), SEASON, 3, captured_at),
        SEASON,
        3,
        captured_at,
        store_path=store,
    )

    assert training_frame(store).empty
