"""In-memory application state the API's endpoints read from: the projection cache
``scripts/build_projections.py`` writes, plus the squad's per-gameweek plan state
(``api.squad_state.SquadState``, one snapshot per horizon gameweek -- the week-on-week simulator),
persisted via ``api.persistence``.

Endpoints never fetch or compute projections themselves, and never touch squad legality directly —
every mutation delegates to ``features.squad_rules``/``features.squad_optimizer``, matching this
page's "no FPL rule logic in the API" layering rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from sqlalchemy.orm import Session

from api.persistence import (
    delete_squad_gameweeks_after,
    latest_squad_gameweek_at_or_before,
    load_squad_gameweek,
    load_squad_state,
    save_squad_account_fields,
    save_squad_gameweek,
)
from api.settings import AppSettingsData, load_app_settings, save_app_settings
from api.squad_state import SquadState
from engine.aggregate import ComponentBreakdown
from engine.data.player_history import PlayerGameweekActual
from engine.data.storage import DEFAULT_DB_PATH, Base, ensure_saved_squads_schema, get_engine
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerGameweekProjection,
    PlayerHorizonProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.simulate import PlayerSimulationSummary
from features.fixtures import TeamRates
from features.squad_rules import INITIAL_BUDGET, count_transfers

__all__ = [
    "DEFAULT_PROJECTION_CACHE_DIR",
    "AppState",
    "load_projection_cache",
    "get_app_state",
    "set_app_state",
    "get_squad_state",
    "set_squad_state",
    "reset_squad_state",
    "get_squad_transfers_made",
    "get_app_settings",
    "set_app_settings",
    "reset_state",
]

DEFAULT_PROJECTION_CACHE_DIR = Path("data_store/projections")

# FPL sets a gameweek's deadline 90 minutes before its first kickoff. Deriving a deadline that
# way is the fallback for any cache built before `deadline_times` was recorded, since fixture
# kickoff times have always been in the cache.
DEADLINE_BEFORE_FIRST_KICKOFF = timedelta(minutes=90)


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
    # The gameweek this cache was *built* for, i.e. whatever `--gameweek` the operator passed
    # to scripts/build_projections.py. It is a fixed property of the file and never advances;
    # `decision_gameweek` below is what a manager can still act on.
    gameweek: int
    horizon_gameweeks: list[int]
    # Frozen at build time (`captured_at >= deadline_time`) and therefore stale the moment the
    # deadline passes. Kept so the cache round-trips intact, but nothing serves it: the API
    # answers with `is_deadline_passed()`, computed against the clock now.
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
    # Every horizon gameweek's real FPL deadline, which is what lets the API work out at read
    # time which gameweek is still open. Empty for any cache built before this field existed
    # (see load_projection_cache's .get()), where `deadline_for` falls back instead.
    deadline_times: dict[int, datetime] = field(default_factory=dict)
    team_id_by_player: dict[int, int] = field(init=False)
    buy_prices: dict[int, int] = field(init=False)
    position_by_player: dict[int, str] = field(init=False)
    first_kickoff_by_gameweek: dict[int, datetime] = field(init=False)

    def __post_init__(self) -> None:
        self.team_id_by_player = {pid: p["team_id"] for pid, p in self.players.items()}
        self.buy_prices = {pid: p["price"] for pid, p in self.players.items()}
        self.position_by_player = {pid: p["position"] for pid, p in self.players.items()}
        self.first_kickoff_by_gameweek = {}
        for row in self.fixtures:
            kickoff = row.get("kickoff_time")
            if not kickoff:
                continue
            gameweek = row["gameweek"]
            parsed = datetime.fromisoformat(kickoff)
            earliest = self.first_kickoff_by_gameweek.get(gameweek)
            if earliest is None or parsed < earliest:
                self.first_kickoff_by_gameweek[gameweek] = parsed

    def deadline_for(self, gameweek: int) -> datetime | None:
        """That gameweek's FPL deadline, or None when this cache holds nothing to derive one from.

        Prefers the deadlines the cache recorded, falls back to the single ``deadline_time`` for
        the cache's own gameweek, and finally to 90 minutes before the gameweek's first kickoff.
        Returning None rather than guessing is deliberate: it keeps ``decision_gameweek`` from
        advancing past a gameweek on evidence it does not actually have.
        """
        recorded = self.deadline_times.get(gameweek)
        if recorded is not None:
            return recorded
        if gameweek == self.gameweek:
            return self.deadline_time
        first_kickoff = self.first_kickoff_by_gameweek.get(gameweek)
        if first_kickoff is None:
            return None
        return first_kickoff - DEADLINE_BEFORE_FIRST_KICKOFF

    @property
    def decision_gameweek(self) -> int:
        """The earliest horizon gameweek a manager can still change anything for.

        Once a gameweek's deadline passes, every mutation this app offers (transfers, captaincy,
        bench order, chips) has stopped affecting it, so the decision gameweek advances at the
        deadline rather than when that gameweek's matches finish. This is a safety net for the
        window between a deadline and the next ``build_projections`` run, not a substitute for it:
        the later gameweeks in a cache were fit on older data than a fresh build would use, which
        is why the API reports ``gameweek`` alongside it.

        Clamps to the last horizon gameweek when every deadline in the horizon has gone, which
        means the cache is stale rather than that anyone can act on that gameweek.
        """
        if not self.horizon_gameweeks:
            return self.gameweek
        now = datetime.now(UTC)
        for gameweek in self.horizon_gameweeks:
            deadline = self.deadline_for(gameweek)
            if deadline is None or now < deadline:
                return gameweek
        return self.horizon_gameweeks[-1]

    @property
    def remaining_horizon_gameweeks(self) -> list[int]:
        """``horizon_gameweeks`` from ``decision_gameweek`` onward: the gameweeks still worth
        planning over, with any that have already locked dropped."""
        target = self.decision_gameweek
        return [gameweek for gameweek in self.horizon_gameweeks if gameweek >= target] or [target]

    def is_deadline_passed(self) -> bool:
        """Whether ``decision_gameweek``'s own deadline has gone, against the clock now rather
        than the cache's frozen ``deadline_passed``. Ordinarily False, since the decision gameweek
        is by definition the first one still open, so a True here means every gameweek in the
        horizon has locked and the cache needs rebuilding."""
        deadline = self.deadline_for(self.decision_gameweek)
        return deadline is not None and datetime.now(UTC) >= deadline

    def expected_points(self, gameweek: int | None = None) -> dict[int, float]:
        """One EV number per projected player for ``gameweek``, exactly what
        ``features.formation.select_starting_xi``/``features.squad_rules.optimise_xi`` need.
        Defaults to the decision gameweek, since optimising an XI for a gameweek that has already
        locked changes nothing."""
        target = gameweek if gameweek is not None else self.decision_gameweek
        return {
            pid: horizon.gameweeks[target].expected_points
            for pid, horizon in self.projections.items()
            if target in horizon.gameweeks
        }

    def refresh_prices(self, now_cost_by_player: dict[int, int]) -> None:
        """Overwrite every already-known player's price from a fresh live pull -- e.g. the
        bootstrap-static snapshot a squad import already fetches -- so every price-dependent read
        (the Players/Player Stats/Differentials pages, the transfer suggester, the squad
        optimiser) reflects today's market rather than whatever this process's projection cache
        happened to be built with, without needing a full ``build_projections`` rerun.

        Silently skips any id this cache doesn't already know about; adding a brand new player is
        not this method's job, only refreshing prices for ones the cache already has a full row
        for. Updates ``buy_prices`` in lockstep with ``players``, since the former is a one-time
        copy taken in ``__post_init__``, not a live view over the latter.
        """
        for player_id, price in now_cost_by_player.items():
            player = self.players.get(player_id)
            if player is None:
                continue
            player["price"] = price
            self.buy_prices[player_id] = price


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
    deadline_times = {
        int(gameweek): datetime.fromisoformat(value)
        for gameweek, value in raw.get("deadline_times", {}).items()
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
        deadline_times=deadline_times,
    )


def _latest_cache_path(cache_dir: Path, season: str) -> Path:
    season_dir = cache_dir / season
    candidates = sorted(season_dir.glob("gw*.json"))
    if not candidates:
        raise FileNotFoundError(f"no projection cache found under {season_dir}")
    return candidates[-1]


_app_state: AppState | None = None
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
    ensure_saved_squads_schema(engine)
    return Session(engine)


def _resolve_gameweek_squad(session: Session, season: str, gameweek: int) -> SquadState | None:
    """This exact gameweek's own saved squad, or -- if it has never been edited -- the nearest
    earlier gameweek's, walked backward via
    :func:`~api.persistence.latest_squad_gameweek_at_or_before`. Returns ``None`` only when no
    gameweek this early has ever been saved for the season, which is the caller's cue to fall back
    to the legacy pre-migration seed. A pure read: resolving a gameweek never writes anything,
    which is what makes an unedited later gameweek "live follow" whatever an earlier one currently
    resolves to."""
    own = load_squad_gameweek(session, season, gameweek)
    if own is not None:
        return own
    predecessor_gameweek = latest_squad_gameweek_at_or_before(session, season, gameweek - 1)
    if predecessor_gameweek is None:
        return None
    return load_squad_gameweek(session, season, predecessor_gameweek)


def get_squad_state(gameweek: int | None = None) -> SquadState:
    """The squad in effect for ``gameweek`` (default: the decision gameweek) -- its own saved
    snapshot if it has one, else whichever earlier gameweek it currently resolves to, else the
    legacy pre-per-gameweek squad, else a fresh empty squad. Account-level fields
    (``budget_ceiling``/``mini_league_ids``/``season_transfers_made``), which don't vary week to
    week, are always merged in from the legacy row regardless of which gameweek this resolves to."""
    app_state = get_app_state()
    target = gameweek if gameweek is not None else app_state.decision_gameweek
    session = _get_session()

    account = load_squad_state(session, app_state.season)
    mini_league_ids = account.mini_league_ids if account is not None else ()
    budget_ceiling = account.budget_ceiling if account is not None else INITIAL_BUDGET
    season_transfers_made = account.season_transfers_made if account is not None else 0

    resolved = _resolve_gameweek_squad(session, app_state.season, target)
    if resolved is None:
        resolved = account
    if resolved is None:
        return SquadState(
            mini_league_ids=mini_league_ids,
            budget_ceiling=budget_ceiling,
            season_transfers_made=season_transfers_made,
        )
    return replace(
        resolved,
        mini_league_ids=mini_league_ids,
        budget_ceiling=budget_ceiling,
        season_transfers_made=season_transfers_made,
    )


def set_squad_state(state: SquadState, gameweek: int | None = None) -> None:
    """Persist ``state`` as ``gameweek``'s own squad snapshot (default: the decision gameweek) --
    every successful mutation calls this. Also persists the account-level fields
    (``budget_ceiling``/``mini_league_ids``/``season_transfers_made``) to the legacy row, since
    those apply across every gameweek rather than to this one alone -- but never the legacy row's
    own squad columns, which stay frozen as the one-time migration seed (see
    ``api.persistence.save_squad_account_fields``)."""
    app_state = get_app_state()
    target = gameweek if gameweek is not None else app_state.decision_gameweek
    session = _get_session()
    save_squad_gameweek(session, app_state.season, target, state)
    save_squad_account_fields(
        session,
        app_state.season,
        state.budget_ceiling,
        state.mini_league_ids,
        state.season_transfers_made,
    )


def reset_squad_state(state: SquadState, gameweek: int | None = None) -> None:
    """Like :func:`set_squad_state`, but for resetting/resyncing the *real* squad (clear/import):
    also drops every later gameweek's saved snapshot, since each was a fork of a baseline that no
    longer exists. Those gameweeks go back to live-following ``gameweek`` until edited again."""
    app_state = get_app_state()
    target = gameweek if gameweek is not None else app_state.decision_gameweek
    session = _get_session()
    delete_squad_gameweeks_after(session, app_state.season, target)
    set_squad_state(state, target)


def get_squad_transfers_made(gameweek: int | None = None) -> int:
    """How many player swaps ``gameweek``'s squad made versus whichever earlier gameweek it was
    forked from (or the legacy pre-migration squad, for the earliest gameweek). 0 for a gameweek
    that is still live-following an earlier one, since it hasn't diverged from it yet."""
    app_state = get_app_state()
    target = gameweek if gameweek is not None else app_state.decision_gameweek
    session = _get_session()

    account = load_squad_state(session, app_state.season)
    current = _resolve_gameweek_squad(session, app_state.season, target) or account
    predecessor = _resolve_gameweek_squad(session, app_state.season, target - 1) or account
    if current is None or predecessor is None:
        return 0
    return count_transfers(predecessor.squad, current.squad)


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
    """Test-only: clear every process-wide singleton so the next access reloads from scratch.
    Squad state itself is never cached process-wide (see :func:`get_squad_state`), so there is no
    squad singleton to clear here -- only ``_db_path`` needs updating for it to read from
    ``db_path`` next call."""
    global _app_state, _app_settings, _db_path
    _app_state = None
    _app_settings = None
    _db_path = db_path
