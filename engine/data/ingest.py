"""Ties the FPL client, Understat client, validation, and snapshotting together into the single
on-demand operation Phase 1's Definition of Done asks for: "produce a clean snapshot for the
current gameweek."

This is deliberately thin — every real decision (retry/fallback policy, what "clean" means) lives
in snapshots.py / validation.py. This module just wires the concrete sources.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy.orm import Session

from engine.data import storage, validation
from engine.data.fpl_client import FPLClient, bootstrap_to_dataframes, fixtures_to_dataframe
from engine.data.snapshots import (
    DEFAULT_BASE_DIR,
    SnapshotManifest,
    capture_snapshot,
    list_snapshot_timestamps,
    load_snapshot_tables,
)
from engine.data.understat_client import UnderstatClient, league_data_to_dataframes


def _record_freshness(engine, manifest: SnapshotManifest, now: datetime) -> None:
    with Session(engine) as session:
        for source_name, result in manifest.sources.items():
            row = session.get(storage.DataFreshness, source_name)
            if row is None:
                row = storage.DataFreshness(
                    source=source_name,
                    last_successful_pull_at=now,
                    last_attempt_at=now,
                    last_attempt_ok=result.status == "ok",
                )
                session.add(row)
            else:
                row.last_attempt_at = now
                row.last_attempt_ok = result.status == "ok"
                if result.status == "ok":
                    row.last_successful_pull_at = now
        session.commit()


def capture_current_gameweek(
    fpl_client: FPLClient,
    understat_client: UnderstatClient,
    season: str,
    understat_season_start_year: int,
    gameweek: int,
    base_dir: Path = DEFAULT_BASE_DIR,
    captured_at: datetime | None = None,
    db_path: str = storage.DEFAULT_DB_PATH,
    retries: int = 3,
) -> SnapshotManifest:
    """Fetch, validate, and freeze today's snapshot for one gameweek.

    ``season`` is the snapshot's own directory key (e.g. ``"2025-26"``);
    ``understat_season_start_year`` is the year Understat's ``getLeagueData`` expects (e.g.
    ``2025``) — kept separate because the two sources use different season-labelling conventions.
    """
    captured_at = captured_at or datetime.now(UTC)

    prior_timestamps = list_snapshot_timestamps(base_dir, season, gameweek)
    previous_row_counts: dict[str, dict[str, int]] = {}
    if prior_timestamps:
        latest_prior = prior_timestamps[-1]
        for source_name in ("fpl", "understat"):
            tables = load_snapshot_tables(base_dir, season, gameweek, latest_prior, source_name)
            if tables:
                previous_row_counts[source_name] = validation.row_counts(tables)

    def fetch_fpl():
        bootstrap = fpl_client.get_bootstrap_static()
        tables = bootstrap_to_dataframes(bootstrap)
        tables["fixtures"] = fixtures_to_dataframe(fpl_client.get_fixtures())
        return tables

    def fetch_understat():
        league_data = understat_client.get_league_data(understat_season_start_year)
        return league_data_to_dataframes(league_data)

    validators = {
        "fpl": validation.make_validator(
            validation.validate_fpl_tables, previous_row_counts.get("fpl")
        ),
        "understat": validation.make_validator(
            validation.validate_understat_tables, previous_row_counts.get("understat")
        ),
    }

    manifest = capture_snapshot(
        season=season,
        gameweek=gameweek,
        sources={"fpl": fetch_fpl, "understat": fetch_understat},
        validators=validators,
        captured_at=captured_at,
        base_dir=base_dir,
        retries=retries,
    )

    engine = storage.init_db(db_path)
    _record_freshness(engine, manifest, captured_at)

    for source_name, result in manifest.sources.items():
        if result.status != "ok":
            validation.alert(
                f"{source_name} snapshot for {season} gw{gameweek} at {captured_at.isoformat()} "
                f"was {result.status}: {result.error}"
            )

    return manifest
