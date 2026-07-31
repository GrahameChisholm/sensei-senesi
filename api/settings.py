"""Persistent app settings (BUILD_PLAN 5.2 Settings screen: FPL team ID, mini-league ID,
planning-horizon default) — backed by :class:`engine.data.storage.AppSettings`, a singleton row.

Separate from :class:`~api.state.AppState`: settings are user-entered and persist across restarts
and weekly refreshes, while ``AppState`` is rebuilt from scratch every refresh and never survives
a restart on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import Engine
from sqlalchemy.orm import Session

from engine.data import storage

DEFAULT_PLANNING_HORIZON_GAMEWEEKS = 5
_SETTINGS_ROW_ID = 1


@dataclass(frozen=True)
class AppSettingsData:
    fpl_team_id: int | None
    mini_league_ids: tuple[int, ...]
    planning_horizon_gameweeks: int


def _parse_mini_league_ids(raw: str) -> tuple[int, ...]:
    return tuple(int(part) for part in raw.split(",") if part)


def get_settings(engine: Engine) -> AppSettingsData:
    """Read the singleton settings row, or the defaults (no team configured, no leagues, the
    standard 5-gameweek horizon) if nothing has been saved yet."""
    with Session(engine) as session:
        row = session.get(storage.AppSettings, _SETTINGS_ROW_ID)
        if row is None:
            return AppSettingsData(
                fpl_team_id=None,
                mini_league_ids=(),
                planning_horizon_gameweeks=DEFAULT_PLANNING_HORIZON_GAMEWEEKS,
            )
        return AppSettingsData(
            fpl_team_id=row.fpl_team_id,
            mini_league_ids=_parse_mini_league_ids(row.mini_league_ids),
            planning_horizon_gameweeks=row.planning_horizon_gameweeks,
        )


def save_settings(engine: Engine, settings: AppSettingsData) -> None:
    """Upsert the singleton settings row."""
    with Session(engine) as session:
        row = session.get(storage.AppSettings, _SETTINGS_ROW_ID)
        if row is None:
            row = storage.AppSettings(id=_SETTINGS_ROW_ID)
            session.add(row)
        row.fpl_team_id = settings.fpl_team_id
        row.mini_league_ids = ",".join(str(i) for i in settings.mini_league_ids)
        row.planning_horizon_gameweeks = settings.planning_horizon_gameweeks
        session.commit()


def get_db_engine() -> Engine:
    """FastAPI dependency — the process-wide SQLite engine, matching
    ``engine.data.ingest``'s own ``storage.init_db()`` call. Tests override this via
    ``app.dependency_overrides[get_db_engine]`` to inject an isolated in-memory/tmp-path engine.
    """
    return storage.init_db()
