"""Point-in-time snapshots (1.2) — the anti-leakage mechanism, and the most important design
decision in the whole data layer.

Captures the full state of every data source *as it stood at that moment* and freezes it as an
immutable parquet snapshot keyed by ``(season, gameweek, captured_at)``. Backtesting later
replays these snapshots. **Never** backtest against current data that has since been revised
(prices move, injuries resolve, xG gets recalculated) — that silently leaks the future into the
past and flatters the model.

**Cadence: daily, not just once per deadline.** Capture daily, with retry-on-failure for each
day's attempt; "the pre-deadline snapshot" (what backtesting replays, what a live decision reads)
is simply the most recent capture strictly before the deadline cutoff
(:func:`get_predeadline_snapshot`). If every retry for a source is exhausted, fall back to the
most recent prior snapshot's data for that source rather than losing the gameweek — see
:func:`capture_snapshot`. A failed *validation* (1.4) is treated identically to a failed fetch:
both trigger the same fallback path.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pandas as pd

DEFAULT_BASE_DIR = Path("data_store/snapshots")

SourceFetchFn = Callable[[], dict[str, pd.DataFrame]]
SourceValidateFn = Callable[[dict[str, pd.DataFrame]], "ValidationOutcome"]

SourceStatus = Literal["ok", "fallback", "missing"]


@dataclass(frozen=True)
class ValidationOutcome:
    """Result of a validation.py sanity check — kept minimal here to avoid a circular import;
    validation.py constructs this directly."""

    ok: bool
    reason: str | None = None


@dataclass
class SourceCaptureResult:
    status: SourceStatus
    tables: list[str] = field(default_factory=list)
    fallback_from: str | None = None
    error: str | None = None


@dataclass
class SnapshotManifest:
    season: str
    gameweek: int
    captured_at: datetime
    sources: dict[str, SourceCaptureResult]

    @property
    def all_ok_or_fallback(self) -> bool:
        return all(s.status != "missing" for s in self.sources.values())

    def to_json(self) -> str:
        return json.dumps(
            {
                "season": self.season,
                "gameweek": self.gameweek,
                "captured_at": self.captured_at.isoformat(),
                "sources": {
                    name: {
                        "status": result.status,
                        "tables": result.tables,
                        "fallback_from": result.fallback_from,
                        "error": result.error,
                    }
                    for name, result in self.sources.items()
                },
            },
            indent=2,
        )

    @classmethod
    def from_json(cls, raw: str) -> SnapshotManifest:
        data = json.loads(raw)
        return cls(
            season=data["season"],
            gameweek=data["gameweek"],
            captured_at=datetime.fromisoformat(data["captured_at"]),
            sources={
                name: SourceCaptureResult(
                    status=s["status"],
                    tables=s["tables"],
                    fallback_from=s["fallback_from"],
                    error=s["error"],
                )
                for name, s in data["sources"].items()
            },
        )


def _gameweek_dir(base_dir: Path, season: str, gameweek: int) -> Path:
    return base_dir / season / f"gw{gameweek}"


def _snapshot_dir(base_dir: Path, season: str, gameweek: int, captured_at: datetime) -> Path:
    return _gameweek_dir(base_dir, season, gameweek) / captured_at.strftime("%Y-%m-%dT%H%M%SZ")


def list_snapshot_timestamps(base_dir: Path, season: str, gameweek: int) -> list[datetime]:
    """All captured_at timestamps that exist on disk for this (season, gameweek), oldest first."""
    gw_dir = _gameweek_dir(base_dir, season, gameweek)
    if not gw_dir.exists():
        return []
    timestamps = []
    for child in gw_dir.iterdir():
        if not child.is_dir():
            continue
        try:
            timestamps.append(datetime.strptime(child.name, "%Y-%m-%dT%H%M%SZ").replace(tzinfo=UTC))
        except ValueError:
            continue
    return sorted(timestamps)


def _load_source_tables(snapshot_dir: Path, source_name: str) -> dict[str, pd.DataFrame]:
    source_dir = snapshot_dir / source_name
    return {
        parquet_file.stem: pd.read_parquet(parquet_file)
        for parquet_file in sorted(source_dir.glob("*.parquet"))
    }


def _write_source_tables(
    snapshot_dir: Path, source_name: str, tables: dict[str, pd.DataFrame]
) -> None:
    source_dir = snapshot_dir / source_name
    source_dir.mkdir(parents=True, exist_ok=True)
    for table_name, df in tables.items():
        df.to_parquet(source_dir / f"{table_name}.parquet")


def _fetch_with_retries(
    fetch: SourceFetchFn, validate: SourceValidateFn | None, retries: int
) -> tuple[dict[str, pd.DataFrame] | None, str | None]:
    last_error: str | None = None
    for _attempt in range(max(retries, 1)):
        try:
            tables = fetch()
        except Exception as exc:  # noqa: BLE001 - any fetch failure triggers retry/fallback
            last_error = f"fetch failed: {exc}"
            continue
        if validate is not None:
            outcome = validate(tables)
            if not outcome.ok:
                last_error = f"validation failed: {outcome.reason}"
                continue
        return tables, None
    return None, last_error


def capture_snapshot(
    season: str,
    gameweek: int,
    sources: dict[str, SourceFetchFn],
    validators: dict[str, SourceValidateFn] | None = None,
    captured_at: datetime | None = None,
    base_dir: Path = DEFAULT_BASE_DIR,
    retries: int = 3,
) -> SnapshotManifest:
    """Capture one snapshot for every named source.

    For each source: fetch, then validate (if a validator is supplied). A failed fetch and a
    failed validation are handled identically — both retry up to ``retries`` times, and if every
    attempt fails, fall back to the most recent prior snapshot's data for that source (1.4's
    "a failed sanity check is treated the same as a failed fetch"). If no prior snapshot exists to
    fall back to, that source is recorded as ``"missing"`` in the manifest rather than raising —
    callers decide whether a partial snapshot is acceptable.
    """
    captured_at = captured_at or datetime.now(UTC)
    validators = validators or {}
    snap_dir = _snapshot_dir(base_dir, season, gameweek, captured_at)

    prior_timestamps = list_snapshot_timestamps(base_dir, season, gameweek)
    prior_timestamp = prior_timestamps[-1] if prior_timestamps else None

    results: dict[str, SourceCaptureResult] = {}

    for source_name, fetch in sources.items():
        tables, error = _fetch_with_retries(fetch, validators.get(source_name), retries)

        if tables is not None:
            _write_source_tables(snap_dir, source_name, tables)
            results[source_name] = SourceCaptureResult(
                status="ok", tables=sorted(tables), error=None
            )
            continue

        if prior_timestamp is not None:
            prior_dir = _snapshot_dir(base_dir, season, gameweek, prior_timestamp)
            fallback_tables = _load_source_tables(prior_dir, source_name)
            if fallback_tables:
                _write_source_tables(snap_dir, source_name, fallback_tables)
                results[source_name] = SourceCaptureResult(
                    status="fallback",
                    tables=sorted(fallback_tables),
                    fallback_from=prior_timestamp.isoformat(),
                    error=error,
                )
                continue

        results[source_name] = SourceCaptureResult(status="missing", tables=[], error=error)

    manifest = SnapshotManifest(
        season=season, gameweek=gameweek, captured_at=captured_at, sources=results
    )
    snap_dir.mkdir(parents=True, exist_ok=True)
    (snap_dir / "manifest.json").write_text(manifest.to_json())
    return manifest


def load_manifest(
    base_dir: Path, season: str, gameweek: int, captured_at: datetime
) -> SnapshotManifest:
    snap_dir = _snapshot_dir(base_dir, season, gameweek, captured_at)
    return SnapshotManifest.from_json((snap_dir / "manifest.json").read_text())


def load_snapshot_tables(
    base_dir: Path, season: str, gameweek: int, captured_at: datetime, source_name: str
) -> dict[str, pd.DataFrame]:
    snap_dir = _snapshot_dir(base_dir, season, gameweek, captured_at)
    return _load_source_tables(snap_dir, source_name)


def get_predeadline_snapshot(
    base_dir: Path, season: str, gameweek: int, deadline: datetime
) -> datetime | None:
    """The most recent captured_at strictly before ``deadline`` — what backtesting replays and
    what a live decision reads. ``None`` if nothing was captured before the deadline.
    """
    candidates = [
        ts for ts in list_snapshot_timestamps(base_dir, season, gameweek) if ts < deadline
    ]
    return candidates[-1] if candidates else None
