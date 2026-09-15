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
    """The team-selection page's account-level row: a single row (``id`` is always 1 — this is a
    single-user local tool) holding whatever doesn't vary week to week. ``budget_ceiling`` is the
    personal budget ceiling checked on every add/remove/optimize call, computed live at import time
    from the imported squad's current player prices plus the entry's bank, floored at the classic
    £100m. ``season_transfers_made`` is FPL's own ``last_deadline_total_transfers`` for the
    imported entry -- how many transfers that real manager has made this season, refreshed on every
    import.

    ``squad_json``/``starting_xi_json``/``bench_order_json``/``captain_id``/``vice_captain_id``
    are legacy, from before per-gameweek squads (:class:`SavedSquadGameweek`) existed. They are no
    longer written, and are read only as the one-time seed a season with no
    ``SavedSquadGameweek`` rows yet forks its decision gameweek from.
    """

    __tablename__ = "saved_squads"

    id: Mapped[int] = mapped_column(primary_key=True, default=1)
    season: Mapped[str] = mapped_column(String, nullable=False)
    squad_json: Mapped[str] = mapped_column(String, nullable=False)
    starting_xi_json: Mapped[str] = mapped_column(String, nullable=False)
    bench_order_json: Mapped[str] = mapped_column(String, nullable=False)
    captain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vice_captain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    mini_league_ids: Mapped[str] = mapped_column(String, default="")
    budget_ceiling: Mapped[int] = mapped_column(Integer, nullable=False)
    season_transfers_made: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


class SavedSquadGameweek(Base):
    """One horizon gameweek's own squad snapshot (the week-on-week simulator) -- squad membership,
    starting XI/bench order, and captain/vice, forked from whichever earlier gameweek it was first
    edited under. Account-level fields that don't vary week to week (``budget_ceiling``,
    ``mini_league_ids``) stay on :class:`SavedSquad` rather than being duplicated here.

    A gameweek with no row here yet is not "empty" -- callers resolve it by walking backward to
    the nearest earlier gameweek that has one (see ``api.state.get_squad_state``), falling back to
    :class:`SavedSquad`'s legacy squad columns for a season that predates this table.
    """

    __tablename__ = "saved_squad_gameweeks"
    __table_args__ = (UniqueConstraint("season", "gameweek", name="uq_saved_squad_gw"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    season: Mapped[str] = mapped_column(String, nullable=False)
    gameweek: Mapped[int] = mapped_column(Integer, nullable=False)
    squad_json: Mapped[str] = mapped_column(String, nullable=False)
    starting_xi_json: Mapped[str] = mapped_column(String, nullable=False)
    bench_order_json: Mapped[str] = mapped_column(String, nullable=False)
    captain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    vice_captain_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow
    )


def ensure_saved_squads_schema(engine: Engine) -> None:
    """``Base.metadata.create_all`` only creates tables that don't exist yet -- it never alters an
    existing table's columns, so a ``saved_squads`` table created before ``season_transfers_made``
    existed needs an explicit ``ALTER TABLE`` here, guarded by a column-existence check so it's a
    cheap no-op on every later call. There's no formal migration framework in this repo (a local,
    single-user SQLite file), so this is the lightweight equivalent for the one column that has
    needed it so far."""
    with engine.connect() as connection:
        columns = {row[1] for row in connection.exec_driver_sql("PRAGMA table_info(saved_squads)")}
        if columns and "season_transfers_made" not in columns:
            connection.exec_driver_sql(
                "ALTER TABLE saved_squads ADD COLUMN season_transfers_made "
                "INTEGER NOT NULL DEFAULT 0"
            )
            connection.commit()


def get_engine(db_path: str = DEFAULT_DB_PATH) -> Engine:
    return create_engine(f"sqlite:///{db_path}")


def init_db(db_path: str = DEFAULT_DB_PATH) -> Engine:
    """Create the schema if it doesn't exist yet. Safe to call every process start."""
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)
    ensure_saved_squads_schema(engine)
    return engine
