"""Accumulating store of live availability signals and what they turned out to mean
(ENGINE_IMPROVEMENTS_5.md Tier 1.1).

**Why this exists.** Availability discrimination is the single largest lever in the engine by a wide
margin. Measured on the real 2025/26 walk-forward, pooled rank correlation is 0.638, while an engine
scaled by *perfect* knowledge of who plays reaches 0.864, and perfect minutes knowledge alone
reaches 0.851. No rate-model refinement on the table is worth a fraction of that.

**Why it cannot be modelled today.** ``engine/models/minutes.py``'s ``FEATURE_COLUMNS`` includes
``chance_of_playing_next_round`` and ``status_score``, which is where FPL states its own view of
whether a player is fit. Both have **zero variance across every historical training row**, because
that data was never retrospectively available: the archive the backtest is fitted on records what
happened, not what was known beforehand. A model fitted on it therefore learns essentially no weight
for either feature, and the only current mitigation is the blunt
``minutes.KNOWN_UNAVAILABLE_P_ZERO`` floor for players already flagged as definitely out. A player
listed at 75% is treated almost identically to one at 100%.

That is not a modelling problem and no amount of feature engineering fixes it. It is a missing
dataset, and the only way to acquire it is to start recording. Hence this module: it is deliberately
plumbing rather than modelling, and its value is a function of elapsed gameweeks rather than of
effort, which is why it is worth starting before the modelling work that will consume it.

**Point-in-time discipline.** Each observation is written from a snapshot taken *before* that
gameweek's deadline, carrying only what was knowable then. Realised minutes are joined on later, at
read time, by :func:`training_frame` (via :func:`attach_realised_minutes`), never persisted back
into the store. That ordering is what makes the resulting frame safe to fit on -- the features
cannot absorb the outcome, the same guarantee ``engine/data/snapshots.py`` provides for the rest of
the engine -- and it means a label can never go stale against a gameweek that was only partially
played when an earlier read happened to run.

**Keyed by season.** FPL reassigns element ids between seasons, so the same ``player_id`` in
2026-27 GW3 and 2027-28 GW3 can be a different person. ``season`` is therefore part of every key
this module uses, not an afterthought bolted on when a second season first collides.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.data.player_history import PlayerGameweekActual
from engine.models.minutes import encode_status

logger = logging.getLogger(__name__)

DEFAULT_STORE_PATH = Path("data_store/availability/observations.parquet")

# Every row recorded before the `season` column existed is provably from this season -- the
# store's own history starts here (2026-27 GW2). Used only to backfill a legacy store on read.
LEGACY_SEASON = "2026-27"

# One row per (season, player_id, gameweek, captured_at). The key deliberately includes the capture
# time, so a re-run before the deadline after fresh team news adds an observation rather than
# replacing one: how a player's listed availability *moved* through the week is itself signal (a
# 75% that becomes 100% on Friday differs from a 75% that stays 75%), and it cannot be recovered
# later.
OBSERVATION_COLUMNS = [
    "season",
    "player_id",
    "gameweek",
    "captured_at",
    "status",
    "status_score",
    "chance_of_playing_next_round",
    "has_news",
    "selected_by_percent",
    "now_cost",
]

REALISED_COLUMNS = ["minutes", "started"]

__all__ = [
    "DEFAULT_STORE_PATH",
    "LEGACY_SEASON",
    "OBSERVATION_COLUMNS",
    "REALISED_COLUMNS",
    "AvailabilityObservationBatch",
    "build_availability_observations",
    "append_observations",
    "load_observations",
    "attach_realised_minutes",
    "realised_from_player_history",
    "training_frame",
]


@dataclass(frozen=True)
class AvailabilityObservationBatch:
    path: Path
    season: str
    gameweek: int
    captured_at: datetime
    n_rows: int


def _stamp(captured_at: datetime) -> str:
    """Second-precision UTC, matching ``engine.data.snapshots``' own directory convention
    (``strftime("%Y-%m-%dT%H%M%SZ")``, i.e. no microseconds).

    ``scripts/build_projections.py`` sets ``captured_at`` from one of two sources depending on
    whether ``--reuse-snapshot`` was passed: a snapshot directory name (second precision) or
    ``datetime.now(UTC)`` (microsecond precision). Both name the same real capture, but without
    normalising to a single precision here, the two produce different isoformat strings and the
    same batch appends twice.
    """
    return captured_at.astimezone(UTC).replace(microsecond=0).isoformat()


def _normalize_stamp(value: str) -> str:
    return _stamp(datetime.fromisoformat(value))


def build_availability_observations(
    elements: pd.DataFrame, season: str, gameweek: int, captured_at: datetime
) -> pd.DataFrame:
    """One row per live player, from the same bootstrap ``elements`` table
    :func:`~engine.data.live_adapter.build_live_availability` already reads.

    ``chance_of_playing_next_round`` is null in FPL's payload for a player carrying no injury doubt
    rather than an explicit 100, treated as 100.0 here, the same convention
    ``build_live_availability`` and ``scripts/build_projections.py``'s cache-facing ``players``
    block both use, so nothing downstream has to know which of the three produced a given row.

    ``has_news`` is a plain boolean rather than the ``news`` free text. BUILD_PLAN 2.1 records the
    decision to treat ``news`` as a display field and not parse it into a feature; whether a note
    exists at all is a different, structured signal, and it is retained because it is the one thing
    that distinguishes "fit, nothing reported" from "fit, but something was reported".
    """
    if "id" not in elements.columns:
        raise ValueError("elements must carry an 'id' column (FPL bootstrap element id)")

    chance = pd.to_numeric(
        elements.get("chance_of_playing_next_round", pd.Series(index=elements.index, dtype=float)),
        errors="coerce",
    ).fillna(100.0)
    status = elements.get("status", pd.Series("a", index=elements.index)).astype(str)
    news = elements.get("news", pd.Series("", index=elements.index)).fillna("").astype(str)

    return pd.DataFrame(
        {
            "season": str(season),
            "player_id": elements["id"].astype(int),
            "gameweek": int(gameweek),
            "captured_at": _stamp(captured_at),
            "status": status,
            "status_score": [encode_status(value) for value in status],
            "chance_of_playing_next_round": chance.astype(float),
            "has_news": news.str.strip().ne(""),
            "selected_by_percent": pd.to_numeric(
                elements.get("selected_by_percent", 0.0), errors="coerce"
            ).fillna(0.0),
            "now_cost": pd.to_numeric(elements.get("now_cost", 0.0), errors="coerce").fillna(0.0),
        },
        columns=OBSERVATION_COLUMNS,
    )


def append_observations(
    observations: pd.DataFrame,
    season: str,
    gameweek: int,
    captured_at: datetime,
    store_path: Path = DEFAULT_STORE_PATH,
) -> AvailabilityObservationBatch:
    """Append to the store, creating it on first use. Appends rather than overwrites, and is a
    no-op for a ``(season, gameweek, captured_at)`` already present, so a re-run of the projection
    build does not duplicate a batch. Unlike ``backtest.prediction_log``, this store is *not*
    immutable: it is a growing dataset, not an accuracy record, so adding to it is the normal case.
    """
    store_path.parent.mkdir(parents=True, exist_ok=True)
    existing = load_observations(store_path)
    stamp = _stamp(captured_at)
    if not existing.empty:
        already = existing[
            (existing["season"] == str(season))
            & (existing["gameweek"] == gameweek)
            & (existing["captured_at"] == stamp)
        ]
        if not already.empty:
            return AvailabilityObservationBatch(
                path=store_path, season=season, gameweek=gameweek, captured_at=captured_at, n_rows=0
            )
    combined = (
        pd.concat([existing, observations], ignore_index=True)
        if not existing.empty
        else observations
    )
    combined.to_parquet(store_path, index=False)
    return AvailabilityObservationBatch(
        path=store_path,
        season=season,
        gameweek=gameweek,
        captured_at=captured_at,
        n_rows=len(observations),
    )


def load_observations(store_path: Path = DEFAULT_STORE_PATH) -> pd.DataFrame:
    """Read the store back, or an empty frame (not an error) before anything has been recorded.

    Backfills a missing ``season`` column with :data:`LEGACY_SEASON`, then normalises every
    ``captured_at`` to second precision and drops any resulting duplicate rows on
    ``(season, player_id, gameweek, captured_at)``. A batch appended once via ``--reuse-snapshot``
    (second-precision stamp) and once via a fresh live capture (microsecond-precision stamp,
    before this normalisation existed) is otherwise byte-identical but for that stamp, and would
    silently double-count every downstream reader. Collapsing here, on read, means every existing
    consumer is repaired at once without rewriting the parquet on disk.
    """
    if not store_path.exists():
        return pd.DataFrame(columns=OBSERVATION_COLUMNS)
    frame = pd.read_parquet(store_path)
    if "season" not in frame.columns:
        frame = frame.assign(season=LEGACY_SEASON)
    if "captured_at" in frame.columns:
        frame = frame.assign(captured_at=frame["captured_at"].map(_normalize_stamp))
    before = len(frame)
    frame = frame.drop_duplicates(subset=["season", "player_id", "gameweek", "captured_at"])
    if len(frame) < before:
        logger.info(
            "load_observations: collapsed %d duplicate availability observation(s) from %s",
            before - len(frame),
            store_path,
        )
    return frame.reset_index(drop=True)


def attach_realised_minutes(observations: pd.DataFrame, realised: pd.DataFrame) -> pd.DataFrame:
    """Left-join what actually happened onto the recorded signals.

    ``realised`` needs ``season``, ``player_id``, ``gameweek``, ``minutes``, and optionally
    ``starts`` -- see :func:`realised_from_player_history` for the usual way to build it. A left
    join, so an observation for a gameweek not yet played survives with null outcomes rather than
    vanishing: the store's job is to accumulate, and dropping unresolved rows would quietly discard
    the most recent gameweek every time this is called.
    """
    if observations.empty:
        return observations.assign(minutes=pd.NA, started=pd.NA)
    columns = ["season", "player_id", "gameweek", "minutes"]
    if "starts" in realised.columns:
        columns.append("starts")
    merged = observations.merge(
        realised[columns], on=["season", "player_id", "gameweek"], how="left"
    )
    merged["started"] = (
        merged["starts"].astype("Float64") > 0 if "starts" in merged.columns else pd.NA
    )
    return merged.drop(columns=["starts"], errors="ignore")


def realised_from_player_history(
    history: Mapping[int, Sequence[PlayerGameweekActual]], season: str
) -> pd.DataFrame:
    """``season``/``player_id``/``gameweek``/``minutes``/``starts`` for every played gameweek in
    ``history`` -- the shape :func:`attach_realised_minutes` expects.

    Built from the already-parsed :class:`~engine.data.player_history.PlayerGameweekActual`
    objects the Player Stats page and ``scripts/build_projections.py`` already load via
    :func:`~engine.data.player_history.load_live_player_history`, rather than from raw cache JSON,
    so this module gains no knowledge of the projection cache's shape and no new HTTP calls are
    needed to label a batch. ``season`` is a parameter because that dataclass carries no season of
    its own.
    """
    rows = [
        {
            "season": str(season),
            "player_id": player_id,
            "gameweek": actual.gameweek,
            "minutes": actual.minutes,
            "starts": actual.starts,
        }
        for player_id, actuals in history.items()
        for actual in actuals
    ]
    return pd.DataFrame(rows, columns=["season", "player_id", "gameweek", "minutes", "starts"])


def training_frame(
    store_path: Path = DEFAULT_STORE_PATH, realised: pd.DataFrame | None = None
) -> pd.DataFrame:
    """The store reduced to one row per (season, player_id, gameweek) with a resolved outcome,
    ready to fit.

    ``realised`` -- typically :func:`realised_from_player_history` -- is joined here, at read
    time, via :func:`attach_realised_minutes`. Keeping the join out of the persisted store is what
    lets the store itself stay purely point-in-time (see the module docstring): a label computed
    this way can never go stale against a gameweek that was only partially played when an earlier
    call happened to run, and the parquet on disk never needs a migration when the labelling logic
    changes. Omit ``realised`` to fall back to any outcome columns already present on the frame
    (a caller that pre-attached before appending, as some tests do).

    Keeps the **last** observation before the deadline for each ``(season, player_id, gameweek)``,
    the one a manager would actually have acted on -- via ``drop_duplicates(..., keep="last")``
    rather than ``groupby(...).last()``, since the latter takes the last *non-null* value **per
    column independently** and can therefore assemble a row that never existed (e.g. a null
    ``chance_of_playing_next_round`` in the final capture pulling that one field from an earlier
    capture while every other field comes from the last). Sorting ``captured_at``
    lexicographically is only chronologically correct because :func:`load_observations` has
    already normalised every stamp to the same (second) precision. Rows with no realised minutes
    are dropped here (and only here, not in :func:`attach_realised_minutes`), since they cannot
    contribute to a fit.
    """
    frame = load_observations(store_path)
    if realised is not None:
        frame = attach_realised_minutes(frame, realised)
    if frame.empty or "minutes" not in frame.columns:
        return pd.DataFrame()
    resolved = frame.dropna(subset=["minutes"])
    if resolved.empty:
        return pd.DataFrame()
    return resolved.sort_values("captured_at").drop_duplicates(
        subset=["season", "player_id", "gameweek"], keep="last"
    )
