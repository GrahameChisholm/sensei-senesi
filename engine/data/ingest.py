"""Ties the FPL client, Understat client, validation, and snapshotting together into the single
on-demand operation Phase 1's Definition of Done asks for: "produce a clean snapshot for the
current gameweek."

This is deliberately thin — every real decision (retry/fallback policy, what "clean" means) lives
in snapshots.py / validation.py. This module just wires the concrete sources.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from engine.data import storage, validation
from engine.data.crosswalk import (
    MANUAL_OVERLAY_UNDERSTAT_TO_FPL,
    build_crosswalk,
    fpl_id_by_name_from_elements,
    fpl_web_name_by_id_from_elements,
    understat_players_from_league_data,
)
from engine.data.fpl_client import FPLClient, bootstrap_to_dataframes, fixtures_to_dataframe
from engine.data.snapshots import (
    DEFAULT_BASE_DIR,
    SnapshotManifest,
    capture_snapshot,
    list_snapshot_timestamps,
    load_snapshot_tables,
)
from engine.data.understat_client import (
    UnderstatClient,
    league_data_to_dataframes,
    player_data_to_dataframe,
)

logger = logging.getLogger(__name__)


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
        for source_name in (
            "fpl",
            "understat",
            "fpl_element_summaries",
            "understat_player_histories",
        ):
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

    def fetch_fpl_element_summaries():
        """Per-player, per-gameweek history for every current element (A1) — the live-path
        equivalent of the vaastav ``merged_gw.csv`` archive
        ``backtest.run_season.engineer_features`` builds from. FPL has no bulk endpoint (one
        request per player, ~500-700 requests); a self-contained bootstrap pull here (rather than
        reusing the ``fetch_fpl`` closure's result) keeps this source independently retryable, at
        the cost of one extra lightweight bootstrap-static call.
        """
        elements = bootstrap_to_dataframes(fpl_client.get_bootstrap_static())["elements"]
        summaries = fpl_client.iter_element_summaries(elements["id"].tolist())
        history_rows = [row for summary in summaries.values() for row in summary.get("history", [])]
        return {"histories": pd.DataFrame(history_rows)}

    def fetch_understat_player_histories():
        """Per-player, per-match Understat history for every player the live crosswalk can match
        (A1) — the live-path equivalent of ``backtest.run_season.fetch_understat_player_histories``
        replaying Understat data for the vaastav-sourced backtest. Builds its own crosswalk from a
        fresh bootstrap + league-data pull (independently retryable, same tradeoff as the FPL
        element-summary source above) using the flat, season-agnostic
        ``MANUAL_OVERLAY_UNDERSTAT_TO_FPL`` (see crosswalk.py's own docstring for why live
        ingestion — always exactly one season in play — doesn't need the backtest driver's
        season-keyed overlay). Non-strict: a genuinely new/unmatchable player this gameweek should
        not fail the whole snapshot; ``histories`` is simply thinner for that player until the
        crosswalk (or its manual overlay) is updated to cover them.
        """
        elements = bootstrap_to_dataframes(fpl_client.get_bootstrap_static())["elements"]
        league_data = understat_client.get_league_data(understat_season_start_year)
        crosswalk = build_crosswalk(
            understat_players_from_league_data(league_data),
            fpl_id_by_name_from_elements(elements),
            overlay=MANUAL_OVERLAY_UNDERSTAT_TO_FPL,
            strict=False,
            fpl_id_by_web_name=fpl_web_name_by_id_from_elements(elements),
        )
        history_frames = []
        for entry in crosswalk:
            player_data = understat_client.get_player_data(entry.understat_id)
            frame = player_data_to_dataframe(player_data)
            frame["fpl_id"] = entry.fpl_id
            history_frames.append(frame)
        histories = (
            pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame()
        )
        return {"histories": histories}

    validators = {
        "fpl": validation.make_validator(
            validation.validate_fpl_tables, previous_row_counts.get("fpl")
        ),
        "understat": validation.make_validator(
            validation.validate_understat_tables, previous_row_counts.get("understat")
        ),
        "fpl_element_summaries": validation.make_validator(
            validation.validate_fpl_element_summaries,
            previous_row_counts.get("fpl_element_summaries"),
        ),
        "understat_player_histories": validation.make_validator(
            validation.validate_understat_player_histories,
            previous_row_counts.get("understat_player_histories"),
        ),
    }

    manifest = capture_snapshot(
        season=season,
        gameweek=gameweek,
        sources={
            "fpl": fetch_fpl,
            "understat": fetch_understat,
            "fpl_element_summaries": fetch_fpl_element_summaries,
            "understat_player_histories": fetch_understat_player_histories,
        },
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
