"""Data validation & freshness (1.4).

Sanity checks on every pull (row counts, null rates, obvious anomalies), plus a "last updated"
record per source so a future dashboard can show data freshness.

**Failure mode: a failed sanity check is treated the same as a failed fetch.** A capture that
fetches successfully but looks wrong (a sudden null-rate spike, a missing team, a row count that
collapsed overnight) is a worse failure mode than being one day stale, since a silently-corrupted
snapshot would feed wrong inputs into every downstream component for a week. Validators here
return a :class:`~engine.data.snapshots.ValidationOutcome` that ``snapshots.capture_snapshot``
treats identically to a fetch exception — reject, fall back to the last known-good snapshot,
alert (see :func:`alert`).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import pandas as pd

from engine.data.snapshots import ValidationOutcome

logger = logging.getLogger("engine.data.validation")

PREMIER_LEAGUE_TEAM_COUNT = 20

# Lenient floors, not exact expectations — deliberately below any real matchday count so a
# genuinely small (but not broken) pull doesn't trip the alarm; tuned to catch collapse, not
# to assert precision.
MIN_FPL_ELEMENTS = 400
MIN_UNDERSTAT_PLAYERS = 300

MAX_ROW_COLLAPSE_FRACTION = 0.5  # a >50% drop vs. the last good pull is a collapse, not noise


def _fail(reason: str) -> ValidationOutcome:
    logger.warning("validation failed: %s", reason)
    return ValidationOutcome(ok=False, reason=reason)


def _ok() -> ValidationOutcome:
    return ValidationOutcome(ok=True)


def check_row_count(df: pd.DataFrame, table_name: str, minimum: int) -> ValidationOutcome:
    if len(df) < minimum:
        return _fail(f"'{table_name}' has {len(df)} rows, expected at least {minimum}")
    return _ok()


def check_required_columns(
    df: pd.DataFrame, table_name: str, required: list[str]
) -> ValidationOutcome:
    missing = [c for c in required if c not in df.columns]
    if missing:
        return _fail(f"'{table_name}' is missing required columns: {missing}")
    return _ok()


def check_null_rate(
    df: pd.DataFrame, table_name: str, columns: list[str], max_null_rate: float = 0.05
) -> ValidationOutcome:
    for column in columns:
        if column not in df.columns or len(df) == 0:
            continue
        null_rate = df[column].isna().mean()
        if null_rate > max_null_rate:
            return _fail(
                f"'{table_name}.{column}' null rate {null_rate:.1%} exceeds {max_null_rate:.1%}"
            )
    return _ok()


def check_row_count_collapse(
    table_name: str,
    current_count: int,
    previous_count: int,
    max_drop_fraction: float = MAX_ROW_COLLAPSE_FRACTION,
) -> ValidationOutcome:
    if previous_count == 0:
        return _ok()
    drop_fraction = 1 - (current_count / previous_count)
    if drop_fraction > max_drop_fraction:
        return _fail(
            f"'{table_name}' row count collapsed from {previous_count} to {current_count} "
            f"({drop_fraction:.0%} drop)"
        )
    return _ok()


def _combine(*outcomes: ValidationOutcome) -> ValidationOutcome:
    failures = [o.reason for o in outcomes if not o.ok]
    if failures:
        return ValidationOutcome(ok=False, reason="; ".join(failures))
    return ValidationOutcome(ok=True)


def validate_fpl_tables(
    tables: dict[str, pd.DataFrame],
    previous_row_counts: dict[str, int] | None = None,
) -> ValidationOutcome:
    """Sanity-check a ``bootstrap_to_dataframes`` payload before it's allowed into a snapshot."""
    previous_row_counts = previous_row_counts or {}
    elements = tables["elements"]
    teams = tables["teams"]

    outcomes = [
        check_row_count(elements, "elements", MIN_FPL_ELEMENTS),
        check_required_columns(
            elements,
            "elements",
            [
                "id",
                "web_name",
                "team",
                "element_type",
                "status",
                "now_cost",
                "minutes",
                "defensive_contribution",
                "clearances_blocks_interceptions",
                "tackles",
                "recoveries",
            ],
        ),
        check_null_rate(elements, "elements", ["id", "team", "element_type", "status"]),
        check_row_count(teams, "teams", PREMIER_LEAGUE_TEAM_COUNT),
    ]
    if "elements" in previous_row_counts:
        outcomes.append(
            check_row_count_collapse("elements", len(elements), previous_row_counts["elements"])
        )
    return _combine(*outcomes)


def validate_understat_tables(
    tables: dict[str, pd.DataFrame],
    previous_row_counts: dict[str, int] | None = None,
) -> ValidationOutcome:
    """Sanity-check a ``league_data_to_dataframes`` payload before it's allowed into a snapshot."""
    previous_row_counts = previous_row_counts or {}
    players = tables["players"]

    outcomes = [
        check_row_count(players, "players", MIN_UNDERSTAT_PLAYERS),
        check_required_columns(
            players, "players", ["id", "player_name", "team_title", "xG", "xA", "npxG", "time"]
        ),
        check_null_rate(players, "players", ["id", "player_name"]),
    ]
    if "players" in previous_row_counts:
        outcomes.append(
            check_row_count_collapse("players", len(players), previous_row_counts["players"])
        )
    return _combine(*outcomes)


# Lenient floors for the two A1 live-history sources, matching MIN_FPL_ELEMENTS/
# MIN_UNDERSTAT_PLAYERS' own "catch collapse, not assert precision" philosophy. Both are naturally
# much larger than an elements/players table (one history row per player per gameweek so far this
# season, not one row per player) -- floors are correspondingly higher, but still well below a
# real season's true row count so a genuinely small early-season pull doesn't trip the alarm.
MIN_FPL_ELEMENT_SUMMARY_ROWS = 400
MIN_UNDERSTAT_PLAYER_HISTORY_ROWS = 300


def validate_fpl_element_summaries(
    tables: dict[str, pd.DataFrame],
    previous_row_counts: dict[str, int] | None = None,
) -> ValidationOutcome:
    """Sanity-check the A1 live per-gameweek player-history pull
    (``engine.data.ingest.capture_current_gameweek``'s ``fpl_element_summaries`` source) before
    it's allowed into a snapshot."""
    previous_row_counts = previous_row_counts or {}
    histories = tables["histories"]

    outcomes = [
        check_row_count(histories, "histories", MIN_FPL_ELEMENT_SUMMARY_ROWS),
        check_required_columns(
            histories, "histories", ["element", "round", "minutes", "total_points"]
        ),
        check_null_rate(histories, "histories", ["element", "round"]),
    ]
    if "histories" in previous_row_counts:
        outcomes.append(
            check_row_count_collapse("histories", len(histories), previous_row_counts["histories"])
        )
    return _combine(*outcomes)


def validate_understat_player_histories(
    tables: dict[str, pd.DataFrame],
    previous_row_counts: dict[str, int] | None = None,
) -> ValidationOutcome:
    """Sanity-check the A1 live per-match Understat player-history pull
    (``engine.data.ingest.capture_current_gameweek``'s ``understat_player_histories`` source)
    before it's allowed into a snapshot. Deliberately does not require a minimum row count to grow
    monotonically the way the FPL element-summary check does -- a thin crosswalk match share (a
    genuinely new signing not yet in the manual overlay) legitimately shrinks this table without
    the pull itself having failed, so a collapse check would produce false alarms this source
    doesn't need; the row-count floor and required-columns checks still catch a truly broken pull.
    """
    previous_row_counts = previous_row_counts or {}
    histories = tables["histories"]

    return _combine(
        check_row_count(histories, "histories", MIN_UNDERSTAT_PLAYER_HISTORY_ROWS),
        check_required_columns(histories, "histories", ["fpl_id", "date", "xG", "xA"]),
        check_null_rate(histories, "histories", ["fpl_id", "date"]),
    )


def make_validator(
    validate: Callable[[dict[str, pd.DataFrame], dict[str, int] | None], ValidationOutcome],
    previous_row_counts: dict[str, int] | None,
) -> Callable[[dict[str, pd.DataFrame]], ValidationOutcome]:
    """Bind a previous snapshot's row counts into a plain ``SourceValidateFn`` (one argument),
    matching the signature ``snapshots.capture_snapshot`` expects."""

    def _validator(tables: dict[str, pd.DataFrame]) -> ValidationOutcome:
        return validate(tables, previous_row_counts)

    return _validator


def row_counts(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {name: len(df) for name, df in tables.items()}


def alert(message: str) -> None:
    """Surface a validation/fallback failure. Phase 1 has no notification channel wired up yet
    (Slack/email/etc.), so this logs at ERROR level — callers in later phases can attach a
    ``logging.Handler`` to ``engine.data.validation`` to route this wherever they like, without
    this module needing to know about it.
    """
    logger.error("ALERT: %s", message)
