"""Capture-only entry point for the availability observation store
(ENGINE_IMPROVEMENTS_5.md Tier 1.1, see ``engine/data/availability_log.py``'s own docstring for
why the store exists).

``scripts/build_projections.py`` already appends an observation batch on every run, but a full
build fetches Understat, takes a snapshot, refits every component, and simulates -- several
minutes of work to capture what is really just one FPL API call. That cost means the intra-week
movement the store's own docstring calls unrecoverable (how a player's listed availability moved
through the week, not just where it landed) is in practice captured only by accident, whenever a
build happens to run.

This script exists to make a capture cheap enough to run daily: fetch ``bootstrap-static``, append
one observation batch, and stop. Deliberately does **not** call
:func:`~engine.data.ingest.capture_current_gameweek` -- that also pulls Understat, per-player
element summaries, and player histories, none of which this store needs. It is a decision, not an
oversight, that this path takes no snapshot of its own: the observation store is itself the
point-in-time record.

::

    uv run python scripts/capture_availability.py --season 2026-27 [--gameweek 4]
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from engine.data.availability_log import (
    DEFAULT_STORE_PATH,
    append_observations,
    build_availability_observations,
)
from engine.data.fpl_client import FPLClient, bootstrap_to_dataframes

__all__ = ["resolve_gameweek", "capture_availability"]


def resolve_gameweek(events: pd.DataFrame, override: int | None = None) -> int:
    """The gameweek to tag this capture with: ``override`` if given, otherwise whichever event
    FPL's own bootstrap flags ``is_next``.

    ``is_next`` is not just convenient, it is the semantically correct choice --
    ``chance_of_playing_next_round`` is by definition about the very next round, so tagging a
    capture by the event FPL itself calls "next" keeps the observation's meaning aligned with the
    field it is recording. Raises if no event is flagged ``is_next`` and no override was given,
    rather than guessing, since a mis-tagged observation cannot be told apart from a correct one
    later.
    """
    if override is not None:
        return override
    next_events = events[events["is_next"].astype(bool)]
    if next_events.empty:
        raise ValueError(
            "no event flagged is_next in FPL's bootstrap data, and no --gameweek override given"
        )
    return int(next_events.iloc[0]["id"])


def capture_availability(
    fpl_client: FPLClient,
    season: str,
    gameweek: int | None = None,
    store_path: Path = DEFAULT_STORE_PATH,
    captured_at: datetime | None = None,
) -> tuple[int, int]:
    """Fetch ``bootstrap-static`` and append one observation batch. Returns
    ``(gameweek, n_rows)`` for the caller to report. ``captured_at`` defaults to
    ``datetime.now(UTC)``; overridable so a caller (a test, or a future backfill) can pin it."""
    bootstrap = fpl_client.get_bootstrap_static()
    tables = bootstrap_to_dataframes(bootstrap)
    resolved_gameweek = resolve_gameweek(tables["events"], gameweek)
    captured_at = captured_at or datetime.now(UTC)

    observations = build_availability_observations(
        tables["elements"], season, resolved_gameweek, captured_at
    )
    batch = append_observations(
        observations, season, resolved_gameweek, captured_at, store_path=store_path
    )
    return resolved_gameweek, batch.n_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Append one availability observation batch, without a full projection build."
    )
    parser.add_argument("--season", required=True, help='e.g. "2026-27"')
    parser.add_argument(
        "--gameweek",
        type=int,
        default=None,
        help="Override the auto-detected (is_next) gameweek",
    )
    args = parser.parse_args(argv)

    with FPLClient() as fpl_client:
        gameweek, n_rows = capture_availability(fpl_client, args.season, args.gameweek)

    print(f"Captured {n_rows} availability observation(s) for {args.season} GW{gameweek}")


if __name__ == "__main__":
    main()
