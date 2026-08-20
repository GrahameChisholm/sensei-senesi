"""SQLite storage schema (1.3) — structured relational data: players, teams, fixtures, gameweek
results, and the ground-truth results table every backtest scores against.

Point-in-time *snapshots* (the anti-leakage mechanism) are immutable parquet files, not rows in
this database — see snapshots.py. This module holds the plain relational bookkeeping: current
player/team reference data, the fixture list, and — critically — the ``GameweekResult`` table
recording what *actually* happened, which is append-only and never revised once a gameweek is
final.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

DEFAULT_DB_PATH = "data_store/fpl.sqlite"

# SQLite has no native tz-aware timestamp storage: a tz-aware datetime written here round-trips
# as a naive one on read (value unchanged, offset dropped). Convention: every datetime passed
# into this module is UTC, tz-aware or not — callers comparing a read-back value against one
# they constructed should not assume tzinfo survives the round trip.


def _utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id: Mapped[int] = mapped_column(primary_key=True)  # FPL team id
    name: Mapped[str] = mapped_column(String, nullable=False)
    short_name: Mapped[str] = mapped_column(String, nullable=False)


class Player(Base):
    __tablename__ = "players"

    id: Mapped[int] = mapped_column(primary_key=True)  # FPL element id
    first_name: Mapped[str] = mapped_column(String, nullable=False)
    second_name: Mapped[str] = mapped_column(String, nullable=False)
    web_name: Mapped[str] = mapped_column(String, nullable=False)
    team_id: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    position: Mapped[str] = mapped_column(String, nullable=False)  # GK/DEF/MID/FWD
    understat_id: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Fixture(Base):
    __tablename__ = "fixtures"

    id: Mapped[int] = mapped_column(primary_key=True)  # FPL fixture id
    event: Mapped[int | None] = mapped_column(Integer, nullable=True)  # gameweek number
    kickoff_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    team_h: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    team_a: Mapped[int] = mapped_column(ForeignKey("teams.id"), nullable=False)
    team_h_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    team_a_score: Mapped[int | None] = mapped_column(Integer, nullable=True)
    finished: Mapped[bool] = mapped_column(default=False)


class GameweekResult(Base):
    """Ground truth: what a player actually scored in a gameweek. Append-only — a backtest's
    scoring reference, never touched once FPL marks the gameweek final (``event.data_checked``).
    """

    __tablename__ = "gameweek_results"
    __table_args__ = (
        UniqueConstraint("player_id", "event", name="uq_gameweek_result_player_event"),
    )

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), nullable=False)
    event: Mapped[int] = mapped_column(Integer, nullable=False)  # gameweek number
    fixture_id: Mapped[int] = mapped_column(ForeignKey("fixtures.id"), nullable=False)
    minutes: Mapped[int] = mapped_column(Integer, nullable=False)
    total_points: Mapped[int] = mapped_column(Integer, nullable=False)
    goals_scored: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    assists: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    clean_sheets: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    goals_conceded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    defensive_contribution: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    saves: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bonus: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    bps: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    yellow_cards: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    red_cards: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    penalties_missed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)


class DataFreshness(Base):
    """Last-updated record per source (1.4) — lets a future dashboard show data freshness, and
    lets validation reason about how stale the last *good* pull was.
    """

    __tablename__ = "data_freshness"

    source: Mapped[str] = mapped_column(String, primary_key=True)  # e.g. "fpl", "understat"
    last_successful_pull_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_attempt_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    last_attempt_ok: Mapped[bool] = mapped_column(default=True)


class AppSettings(Base):
    """User-configurable app settings (BUILD_PLAN 5.2's Settings screen: "FPL team ID, mini-league
    ID, planning horizon default") — a singleton row (``id`` is always 1), separate from
    :class:`~api.state.AppState` since settings persist across restarts and weekly refreshes while
    ``AppState`` is rebuilt from scratch every refresh. ``mini_league_ids`` is stored as a
    comma-separated string (SQLite has no native array column) — see ``api.settings`` for the
    parsed dataclass this table backs.
    """

    __tablename__ = "app_settings"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    fpl_team_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mini_league_ids: Mapped[str] = mapped_column(String, default="")
    planning_horizon_gameweeks: Mapped[int] = mapped_column(Integer, default=5)


class SavedSquad(Base):
    """The team-selection page's persisted squad (D17/G6): a single row (``id`` is always 1 —
    this is a single-user local tool) holding both the real, confirmed
    ``features.squad_draft.CommittedSquad`` and any unconfirmed ``PendingDraft`` (so an in-progress
    edit survives a refresh or closed tab, per D17), each serialised to JSON. A single JSON column
    per concept rather than a normalised schema: both objects are always read and written whole,
    and their shape is still evolving alongside the rest of the team-selection page.
    """

    __tablename__ = "saved_squads"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    season: Mapped[str] = mapped_column(String, nullable=False)
    committed_gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    committed_state_json: Mapped[str | None] = mapped_column(String, nullable=True)
    chip_usage_json: Mapped[str] = mapped_column(String, nullable=False)
    active_chip: Mapped[str | None] = mapped_column(String, nullable=True)
    active_chip_gameweek: Mapped[int | None] = mapped_column(Integer, nullable=True)
    free_hit_snapshot_json: Mapped[str | None] = mapped_column(String, nullable=True)
    free_hit_snapshot_gameweek: Mapped[int | None] = mapped_column(Integer, nullable=True)
    pending_draft_json: Mapped[str | None] = mapped_column(String, nullable=True)
    # The hits already charged for `committed_gameweek` specifically
    # (features.squad_draft.CommittedSquad.gameweek_hit_cost).
    gameweek_hit_cost: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class SavedBuildPicks(Base):
    """The team-selection page's in-progress initial-build squad (0 to 15 picks, D6/D23), stored
    separately from :class:`SavedSquad` since it isn't a real ``MyTeamState`` and shouldn't block
    that table's schema. A single row (``id`` is always 1), since this is a single-user local
    tool.
    """

    __tablename__ = "saved_build_picks"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    season: Mapped[str] = mapped_column(String, nullable=False)
    picks_json: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


def get_engine(db_path: str = DEFAULT_DB_PATH) -> Engine:
    return create_engine(f"sqlite:///{db_path}")


def init_db(db_path: str = DEFAULT_DB_PATH) -> Engine:
    """Create the schema if it doesn't exist yet. Safe to call every process start."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    return engine
