"""In-memory application state the API's endpoints read from: the projection cache
``scripts/build_projections.py`` writes, plus the squad's one live sandbox state
(``api.squad_state.SquadState``), persisted via ``api.persistence``.

Endpoints never fetch or compute projections themselves, and never touch squad legality directly —
every mutation delegates to ``features.squad_rules``/``features.squad_optimizer``, matching this
page's "no FPL rule logic in the API" layering rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from api.persistence import load_squad_state, save_squad_state
from api.settings import AppSettingsData, load_app_settings, save_app_settings
from api.squad_state import SquadState
from engine.aggregate import ComponentBreakdown
from engine.data.player_history import PlayerGameweekActual
from engine.data.storage import DEFAULT_DB_PATH, Base, get_engine
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerGameweekProjection,
    PlayerHorizonProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.simulate import PlayerSimulationSummary
from features.fixtures import TeamRates

__all__ = [
    "DEFAULT_PROJECTION_CACHE_DIR",
    "AppState",
    "load_projection_cache",
    "get_app_state",
    "set_app_state",
    "get_squad_state",
    "set_squad_state",
    "get_app_settings",
    "set_app_settings",
    "reset_state",
]

DEFAULT_PROJECTION_CACHE_DIR = Path("data_store/projections")


def _breakdown_from_dict(data: dict) -> ComponentBreakdown:
    return ComponentBreakdown(**data)


def _minutes_from_dict(data: dict) -> MinutesDistribution:
    return MinutesDistribution(**data)


def _simulation_from_dict(data: dict | None, player_id: int) -> PlayerSimulationSummary | None:
    if data is None:
        return None
    # MINI_LEAGUE_PLAN M9: `std` is absent from any cache built before it existed. Fall back to the
    # normal-distribution approximation from the persisted floor (P10)/ceiling (P90) -- that range
    # spans 2 * 1.2816 standard deviations under normality. Strictly worse than a persisted std
    # (FPL points are right-skewed, so this understates the tail), but keeps an old cache usable.
    std = data.get("std")
    if std is None:
        std = (data["ceiling"] - data["floor"]) / 2.5631
    return PlayerSimulationSummary(
        player_id=player_id,
        mean=data["mean"],
        median=data["median"],
        floor=data["floor"],
        ceiling=data["ceiling"],
        prob_big_haul=data["prob_big_haul"],
        raw_points=np.array([]),
        std=std,
    )


def _gameweek_projection_from_dict(
    player_id: int, position: str, data: dict
) -> PlayerGameweekProjection:
    return project_player_gameweek(
        player_id,
        position,
        data["gameweek"],
        _minutes_from_dict(data["minutes"]),
        _breakdown_from_dict(data["breakdown"]),
        _simulation_from_dict(data.get("simulation"), player_id),
    )


def _horizon_projection_from_dict(player_id: int, data: dict) -> PlayerHorizonProjection:
    position = data["position"]
    gameweeks = {
        gw_data["gameweek"]: _gameweek_projection_from_dict(player_id, position, gw_data)
        for gw_data in data["gameweeks"]
    }
    return project_player_horizon(player_id, position, gameweeks)


def _player_actual_from_dict(data: dict) -> PlayerGameweekActual:
    return PlayerGameweekActual(**data)


def _team_rates_from_dict(data: dict) -> TeamRates:
    return TeamRates(
        home_xg_per_90=data["home_xg_per_90"],
        away_xg_per_90=data["away_xg_per_90"],
        home_xga_per_90=data["home_xga_per_90"],
        away_xga_per_90=data["away_xga_per_90"],
    )


@dataclass
class AppState:
    """One loaded projection cache (§6.1 of the team-page plan) — everything the API needs that
    isn't squad-specific."""

    season: str
    gameweek: int
    horizon_gameweeks: list[int]
    deadline_passed: bool
    generated_at: datetime
    deadline_time: datetime
    model_version: str
    projections: dict[int, PlayerHorizonProjection]
    players: dict[int, dict]
    teams: dict[int, dict]
    fixtures: list[dict]
    diagnostics: dict
    # Player Stats page (D1/D4/G1) -- this season's actual per-gameweek performance, live only.
    # Empty for any cache built before this field existed (see load_projection_cache's .get()).
    player_history: dict[int, list[PlayerGameweekActual]] = field(default_factory=dict)
    # Fixture-swing plan Phase 1 -- every team's current xG/xGA rate, live for the first time.
    # Empty for any cache built before this field existed, or a team the live pull has no rate
    # for yet (true GW1); every downstream caller of features.fixtures already handles a team
    # missing from the rates mapping it's given.
    team_rates: dict[int, TeamRates] = field(default_factory=dict)
    team_id_by_player: dict[int, int] = field(init=False)
    buy_prices: dict[int, int] = field(init=False)
    position_by_player: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.team_id_by_player = {pid: p["team_id"] for pid, p in self.players.items()}
        self.buy_prices = {pid: p["price"] for pid, p in self.players.items()}
        self.position_by_player = {pid: p["position"] for pid, p in self.players.items()}

    def expected_points(self, gameweek: int | None = None) -> dict[int, float]:
        """One EV number per projected player for ``gameweek`` (defaults to the current
        gameweek) — exactly what ``features.formation.select_starting_xi``/
        ``features.squad_rules.optimise_xi`` need."""
        target = gameweek if gameweek is not None else self.gameweek
        return {
            pid: horizon.gameweeks[target].expected_points
            for pid, horizon in self.projections.items()
            if target in horizon.gameweeks
        }


def load_projection_cache(path: Path) -> AppState:
    raw = json.loads(path.read_text())
    projections = {
        int(player_id): _horizon_projection_from_dict(int(player_id), horizon_data)
        for player_id, horizon_data in raw["projections"].items()
    }
    players = {int(player_id): data for player_id, data in raw["players"].items()}
    teams = {int(team_id): data for team_id, data in raw["teams"].items()}
    player_history = {
        int(player_id): [_player_actual_from_dict(row) for row in rows]
        for player_id, rows in raw.get("player_history", {}).items()
    }
    team_rates = {
        int(team_id): _team_rates_from_dict(data)
        for team_id, data in raw.get("team_rates", {}).items()
    }
    return AppState(
        season=raw["season"],
        gameweek=raw["gameweek"],
        horizon_gameweeks=raw["horizon_gameweeks"],
        deadline_passed=raw["deadline_passed"],
        generated_at=datetime.fromisoformat(raw["generated_at"]),
        deadline_time=datetime.fromisoformat(raw["deadline_time"]),
        model_version=raw["model_version"],
        projections=projections,
        players=players,
        teams=teams,
        fixtures=raw["fixtures"],
        diagnostics=raw["diagnostics"],
        player_history=player_history,
        team_rates=team_rates,
    )


def _latest_cache_path(cache_dir: Path, season: str) -> Path:
    season_dir = cache_dir / season
    candidates = sorted(season_dir.glob("gw*.json"))
    if not candidates:
        raise FileNotFoundError(f"no projection cache found under {season_dir}")
    return candidates[-1]


_app_state: AppState | None = None
_squad_state: SquadState | None = None
_app_settings: AppSettingsData | None = None
_db_path: str = DEFAULT_DB_PATH


def get_app_state(
    cache_dir: Path = DEFAULT_PROJECTION_CACHE_DIR, season: str | None = None
) -> AppState:
    """Loads a cache file on first access — the process-wide state every endpoint reads from.

    - ``season`` given explicitly (a caller override, or tests): the most recent cache file for
      it — matching the live path's "whichever gameweek was generated most recently" semantics.
    - Neither given: the newest season dir found, its most recent cache file — today's existing
      live-path default.
    """
    global _app_state
    if _app_state is None:
        if season is not None:
            path = _latest_cache_path(cache_dir, season)
        else:
            season_dirs = (
                sorted(d for d in cache_dir.iterdir() if d.is_dir()) if cache_dir.exists() else []
            )
            if not season_dirs:
                raise FileNotFoundError(f"no projection cache found under {cache_dir}")
            path = _latest_cache_path(cache_dir, season_dirs[-1].name)
        _app_state = load_projection_cache(path)
    return _app_state


def set_app_state(state: AppState) -> None:
    """Replace the process-wide app state — what a re-run of the batch job would call, and what
    tests use to inject a fixture state instead of touching disk."""
    global _app_state
    _app_state = state


def _get_session() -> Session:
    engine = get_engine(_db_path)
    Base.metadata.create_all(engine)
    return Session(engine)


def get_squad_state() -> SquadState:
    """Loads the saved squad on first access, or starts a fresh empty one if none has ever been
    saved for the current season."""
    global _squad_state
    if _squad_state is None:
        app_state = get_app_state()
        session = _get_session()
        loaded = load_squad_state(session, app_state.season)
        _squad_state = loaded if loaded is not None else SquadState()
    return _squad_state


def set_squad_state(state: SquadState) -> None:
    """Persist a new squad state and update the process-wide singleton — every successful
    mutation calls this."""
    global _squad_state
    app_state = get_app_state()
    session = _get_session()
    save_squad_state(session, app_state.season, state)
    _squad_state = state


def get_app_settings() -> AppSettingsData:
    """Loads the saved app-wide settings (MINI_LEAGUE_PLAN M14) on first access. The very first
    load carries over any league IDs already recorded on the squad's own (now-vestigial)
    ``mini_league_ids`` field -- see :func:`~api.settings.load_app_settings`'s own docstring for
    why that's a one-time thing rather than a standing fallback."""
    global _app_settings
    if _app_settings is None:
        session = _get_session()
        legacy_mini_league_ids = get_squad_state().mini_league_ids
        _app_settings = load_app_settings(session, legacy_mini_league_ids)
    return _app_settings


def set_app_settings(settings: AppSettingsData) -> None:
    """Persist new app-wide settings and update the process-wide singleton — every successful
    mutation calls this, mirroring :func:`set_squad_state`."""
    global _app_settings
    session = _get_session()
    save_app_settings(session, settings)
    _app_settings = settings


def reset_state(db_path: str = DEFAULT_DB_PATH) -> None:
    """Test-only: clear every process-wide singleton so the next access reloads from scratch."""
    global _app_state, _squad_state, _app_settings, _db_path
    _app_state = None
    _squad_state = None
    _app_settings = None
    _db_path = db_path
