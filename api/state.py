"""In-memory application state the API's endpoints read from (mirrors the deleted ``api/state.py``'s
own "populated once, read by every request" shape, extended for the team-selection page): the
projection cache ``scripts/build_projections.py`` writes, plus the squad's committed/pending state
(``features.squad_draft``), persisted via ``api.persistence``.

Endpoints never fetch or compute projections themselves, and never touch squad legality directly —
every mutation delegates to ``features.squad_rules``/``features.squad_draft``, matching this page's
"no FPL rule logic in the API" layering rule.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from api.persistence import (
    load_season_log,
    load_squad_state,
    save_season_log,
    save_squad_state,
)
from engine.aggregate import ComponentBreakdown
from engine.data.storage import DEFAULT_DB_PATH, Base, get_engine
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerGameweekProjection,
    PlayerHorizonProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.simulate import PlayerSimulationSummary
from features.squad_draft import CommittedSquad, PendingDraft, advance_gameweek
from features.team_state import SquadPlayer

__all__ = [
    "DEFAULT_PROJECTION_CACHE_DIR",
    "AppState",
    "load_projection_cache",
    "get_app_state",
    "set_app_state",
    "get_squad_state",
    "set_squad_state",
    "confirm_and_save",
    "get_build_picks",
    "set_build_picks",
    "get_season_log",
    "set_season_log",
    "reset_state",
]

DEFAULT_PROJECTION_CACHE_DIR = Path("data_store/projections")
# Season Replay only (scripts.build_replay_projections' own DEFAULT_RESULTS_DIR): real recorded
# {gameweek: {player_id: {minutes, total_points}}} ground truth, one file per season, that
# POST /squad/advance scores a committed squad against. The live 2026/27 cache has no sibling file
# here, so AppState.results stays None for it -- see load_projection_cache.
DEFAULT_RESULTS_DIR = Path("data_store/replay")
# Set (e.g. `FPL_REPLAY_SEASON=2025-26 uvicorn api.main:app`) to point a fresh process at a
# historical replay season instead of get_app_state()'s normal "lexicographically last season dir"
# default -- the live 2026/27 path is completely unaffected when this is unset.
_REPLAY_SEASON_ENV_VAR = "FPL_REPLAY_SEASON"


def _breakdown_from_dict(data: dict) -> ComponentBreakdown:
    return ComponentBreakdown(**data)


def _minutes_from_dict(data: dict) -> MinutesDistribution:
    return MinutesDistribution(**data)


def _simulation_from_dict(data: dict | None, player_id: int) -> PlayerSimulationSummary | None:
    if data is None:
        return None
    return PlayerSimulationSummary(
        player_id=player_id,
        mean=data["mean"],
        median=data["median"],
        floor=data["floor"],
        ceiling=data["ceiling"],
        prob_big_haul=data["prob_big_haul"],
        raw_points=np.array([]),
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
    # Season Replay only -- None for the live 2026/27 cache (no ground-truth results exist for a
    # season that hasn't been played yet). See DEFAULT_RESULTS_DIR.
    results: dict[int, dict[int, dict]] | None = None
    team_id_by_player: dict[int, int] = field(init=False)
    buy_prices: dict[int, int] = field(init=False)
    position_by_player: dict[int, str] = field(init=False)

    def __post_init__(self) -> None:
        self.team_id_by_player = {pid: p["team_id"] for pid, p in self.players.items()}
        self.buy_prices = {pid: p["price"] for pid, p in self.players.items()}
        self.position_by_player = {pid: p["position"] for pid, p in self.players.items()}

    def expected_points(self, gameweek: int | None = None) -> dict[int, float]:
        """One EV number per projected player for ``gameweek`` (defaults to the current
        gameweek) — exactly what ``simulator.formation.select_starting_xi``/
        ``features.squad_rules.optimise_xi`` need."""
        target = gameweek if gameweek is not None else self.gameweek
        return {
            pid: horizon.gameweeks[target].expected_points
            for pid, horizon in self.projections.items()
            if target in horizon.gameweeks
        }


def _load_results_for_season(season: str, results_dir: Path = DEFAULT_RESULTS_DIR) -> dict | None:
    """Season Replay's ground truth, if this season has one -- ``None`` for the live 2026/27 cache,
    which has no ``data_store/replay/2026-27/results.json`` (that season hasn't been played yet)."""
    path = results_dir / season / "results.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    return {
        int(gameweek): {int(player_id): result for player_id, result in gw_results.items()}
        for gameweek, gw_results in raw.items()
    }


def load_projection_cache(path: Path, results_dir: Path = DEFAULT_RESULTS_DIR) -> AppState:
    raw = json.loads(path.read_text())
    projections = {
        int(player_id): _horizon_projection_from_dict(int(player_id), horizon_data)
        for player_id, horizon_data in raw["projections"].items()
    }
    players = {int(player_id): data for player_id, data in raw["players"].items()}
    teams = {int(team_id): data for team_id, data in raw["teams"].items()}
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
        results=_load_results_for_season(raw["season"], results_dir),
    )


def _latest_cache_path(cache_dir: Path, season: str) -> Path:
    season_dir = cache_dir / season
    candidates = sorted(season_dir.glob("gw*.json"))
    if not candidates:
        raise FileNotFoundError(f"no projection cache found under {season_dir}")
    return candidates[-1]


def _earliest_cache_path(cache_dir: Path, season: str) -> Path:
    """Season Replay's own cold-start pick: unlike the live path (where "latest generated" means
    "this real season's actual current gameweek"), a replay season has every gameweek's cache
    pre-built at once (``scripts.build_replay_projections``), so a fresh process should always
    start the user at gameweek 1 -- not jump straight to the season's final gameweek."""
    season_dir = cache_dir / season
    candidates = sorted(season_dir.glob("gw*.json"))
    if not candidates:
        raise FileNotFoundError(f"no projection cache found under {season_dir}")
    return candidates[0]


_app_state: AppState | None = None
_committed: CommittedSquad | None = None
_pending: PendingDraft | None = None
_build_picks: list[SquadPlayer] | None = None
_season_log: list[dict] | None = None
_db_path: str = DEFAULT_DB_PATH


def get_app_state(
    cache_dir: Path = DEFAULT_PROJECTION_CACHE_DIR, season: str | None = None
) -> AppState:
    """Loads a cache file on first access — the process-wide state every endpoint reads from.

    - ``season`` given explicitly (a caller override, or tests): the most recent cache file for
      it — matching the live path's "whichever gameweek was generated most recently" semantics.
    - Nothing given, but ``FPL_REPLAY_SEASON`` is set (Season Replay's opt-in switch): the
      *earliest* cache file for that season instead — a replay season has every gameweek
      pre-built at once, so a fresh process should start the user at gameweek 1, not jump to the
      season's last gameweek the way "most recent" would.
    - Neither given: the newest season dir found, its most recent cache file — today's existing
      live-path default, completely unaffected when ``FPL_REPLAY_SEASON`` is unset.
    """
    global _app_state
    if _app_state is None:
        replay_season = os.environ.get(_REPLAY_SEASON_ENV_VAR)
        if season is not None:
            path = _latest_cache_path(cache_dir, season)
        elif replay_season is not None:
            path = _earliest_cache_path(cache_dir, replay_season)
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


def get_squad_state() -> tuple[CommittedSquad, PendingDraft | None]:
    """Loads the saved squad on first access, running :func:`~features.squad_draft.advance_gameweek`
    if the cache has moved on to a later gameweek since the squad was last saved — this is where
    Free Hit reversion (D15) and stale-draft discarding (D24) actually fire."""
    global _committed, _pending
    if _committed is None:
        app_state = get_app_state()
        session = _get_session()
        loaded = load_squad_state(session, app_state.season)
        if loaded is None:
            _committed = CommittedSquad(team_state=None, committed_gameweek=app_state.gameweek)
            _pending = None
        else:
            committed, pending = loaded
            if committed.committed_gameweek != app_state.gameweek:
                committed, pending = advance_gameweek(committed, pending, app_state.gameweek)
            _committed, _pending = committed, pending
    return _committed, _pending


def set_squad_state(committed: CommittedSquad, pending: PendingDraft | None) -> None:
    """Persist a new squad/draft state and update the process-wide singletons — every successful
    draft mutation and confirm/discard calls this."""
    global _committed, _pending
    app_state = get_app_state()
    session = _get_session()
    save_squad_state(session, app_state.season, committed, pending)
    _committed, _pending = committed, pending


def confirm_and_save(committed: CommittedSquad, pending: PendingDraft | None = None) -> None:
    """Convenience alias for the common "confirm, then persist the result" sequence."""
    set_squad_state(committed, pending)


def get_build_picks() -> list[SquadPlayer]:
    """The in-progress initial-build squad (D6/D23) — 0 to 15 picks, not yet a real
    ``MyTeamState`` (which requires exactly 15/11/4 at every step, so it can't represent this
    state at all). Deliberately **not** persisted across a restart, unlike an ongoing edit draft
    (D17) — a one-time bootstrapping step is a smaller loss to redo than an in-progress edit."""
    global _build_picks
    if _build_picks is None:
        _build_picks = []
    return _build_picks


def set_build_picks(picks: list[SquadPlayer]) -> None:
    global _build_picks
    _build_picks = picks


def get_season_log() -> list[dict]:
    """Season Replay's running gameweek-by-gameweek score history (``[]`` for a fresh replay, or
    always for the live 2026/27 squad, which never calls :func:`set_season_log`)."""
    global _season_log
    if _season_log is None:
        session = _get_session()
        _season_log = load_season_log(session)
    return _season_log


def set_season_log(log: list[dict]) -> None:
    """Persist the season log and update the process-wide singleton — called by
    ``POST /squad/advance`` after every gameweek it scores."""
    global _season_log
    session = _get_session()
    save_season_log(session, log)
    _season_log = log


def reset_state(db_path: str = DEFAULT_DB_PATH) -> None:
    """Test-only: clear every process-wide singleton so the next access reloads from scratch."""
    global _app_state, _committed, _pending, _build_picks, _season_log, _db_path
    _app_state = None
    _committed = None
    _pending = None
    _build_picks = None
    _season_log = None
    _db_path = db_path
