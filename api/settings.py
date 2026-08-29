"""JSON-free (de)serialization for the app-wide settings singleton (MINI_LEAGUE_PLAN M14) --
``engine.data.storage.AppSettings`` already names this module as the parsed dataclass its
comma-separated ``mini_league_ids`` column backs. A single row (``id=1``, mirroring
``api.persistence``'s own single-row squad convention), read and written whole, since there is
exactly one FPL team ID and one set of tracked leagues for this single-user local tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from engine.data.storage import AppSettings

__all__ = ["AppSettingsData", "load_app_settings", "save_app_settings"]


@dataclass(frozen=True)
class AppSettingsData:
    fpl_team_id: int | None = None
    mini_league_ids: tuple[int, ...] = field(default=())
    planning_horizon_gameweeks: int = 5


def save_app_settings(session: Session, settings: AppSettingsData) -> None:
    row = session.get(AppSettings, 1)
    if row is None:
        row = AppSettings(id=1)
        session.add(row)

    row.fpl_team_id = settings.fpl_team_id
    row.mini_league_ids = ",".join(str(league_id) for league_id in settings.mini_league_ids)
    row.planning_horizon_gameweeks = settings.planning_horizon_gameweeks
    session.commit()


def load_app_settings(
    session: Session, legacy_mini_league_ids: tuple[int, ...] = ()
) -> AppSettingsData:
    """Returns the saved settings, or a fresh default if none have ever been saved.

    ``legacy_mini_league_ids`` is MINI_LEAGUE_PLAN M14's one-time carry-over: before this module
    existed, a manager's league IDs (if they'd entered any) lived on
    ``api.squad_state.SquadState.mini_league_ids`` -- itself now vestigial, kept only for its
    existing round trip rather than requiring a SQLite migration to drop. On the very first load
    (no row exists yet) with a non-empty legacy value, that value is carried across and persisted
    immediately, so this carry-over only ever happens once -- every subsequent load finds a real
    row and ignores ``legacy_mini_league_ids`` entirely, even if the caller keeps passing it.
    """
    row = session.get(AppSettings, 1)
    if row is None:
        settings = AppSettingsData(mini_league_ids=legacy_mini_league_ids)
        if legacy_mini_league_ids:
            save_app_settings(session, settings)
        return settings

    return AppSettingsData(
        fpl_team_id=row.fpl_team_id,
        mini_league_ids=tuple(int(mid) for mid in row.mini_league_ids.split(",") if mid),
        planning_horizon_gameweeks=row.planning_horizon_gameweeks,
    )
