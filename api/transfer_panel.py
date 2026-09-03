"""Transfer banner assembly (TRANSFER_BANNER) for the Team page's ``GET /squad/transfers``.

Resolves everything :func:`~features.transfer_planner.plan_transfers` needs and hands it over in
one call: the planning gameweeks, each candidate's expected points summed across them, the single
gameweek the league math is measured at, and the mini-league snapshot behind the field's effective
ownership. Exactly the role :mod:`api.mini_league_panel` plays for the Mini League page, and none
of the actual decision logic lives here.

**The league is optional and never fatal.** A manager with no FPL team ID saved, no league
configured, or a league whose live fetch fails still gets a working banner: the planner falls back
to ranking on expected points alone when it is handed no rivals (see its ``_plan_key``). This
mirrors :func:`api.main._resolve_differentials_ownership`'s own never-fail contract rather than
the Mini League page's "no league, no page" one, because the Team page has always worked without
a league and adding a banner to it must not change that.

**Solve results are cached in process.** The suggestion is a pure function of the squad, budget,
transfer count, planning gameweeks, chip, and league snapshot, so re-solving on every render of a
banner that sits under a pitch the manager is actively editing would be wasted work. The cache is
keyed on exactly those inputs, so any squad edit is a natural miss and the banner updates itself
without needing to be told to.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from api.mini_league_panel import (
    DEFAULT_SEASON_LENGTH_GAMEWEEKS,
    get_cached_league_snapshot,
)
from api.settings import AppSettingsData
from api.state import AppState
from engine.data.fpl_client import FPLClient, FPLClientError
from engine.data.league_state_builder import LeagueEntry, LeagueSnapshot
from engine.projections import PlayerGameweekProjection
from features.mini_league import PlayerOwnership, compute_league_ownership
from features.squad_optimizer import PlayerCandidate
from features.team_state import MyTeamState
from features.transfer_planner import DEFAULT_PLAN_COUNT, TransferSuggestion, plan_transfers

__all__ = [
    "MAX_TRANSFERS",
    "LeagueContext",
    "build_transfer_suggestion",
    "resolve_league_context",
    "reset_suggestion_cache",
]

# The largest transfer count the banner offers. Past three the integer program's search space and
# the honesty of a three-gameweek projection both degrade faster than the suggestion improves, and
# a manager planning a four plus move rebuild is really asking for Auto Build, which already
# exists on this page and rebuilds the whole 15.
MAX_TRANSFERS = 3


@dataclass(frozen=True)
class LeagueContext:
    """The mini-league half of the banner's inputs, or the empty version of it when no league is
    available. ``league_id`` is ``None`` precisely when the planner is going to fall back to
    ranking on expected points, which is what the banner tells the manager."""

    league_id: int | None
    league_name: str
    picks_gameweek: int | None
    rivals: tuple[LeagueEntry, ...]
    ownership_by_player: dict[int, PlayerOwnership]
    my_total_points: int


_EMPTY_LEAGUE = LeagueContext(
    league_id=None,
    league_name="",
    picks_gameweek=None,
    rivals=(),
    ownership_by_player={},
    my_total_points=0,
)


def _league_context(snapshot: LeagueSnapshot, my_entry_id: int) -> LeagueContext:
    my_entry = next((entry for entry in snapshot.entries if entry.entry_id == my_entry_id), None)
    rivals = tuple(entry for entry in snapshot.entries if entry.entry_id != my_entry_id)
    if my_entry is None or not rivals:
        return _EMPTY_LEAGUE
    return LeagueContext(
        league_id=snapshot.league_id,
        league_name=snapshot.league_name,
        picks_gameweek=snapshot.picks_gameweek,
        rivals=rivals,
        ownership_by_player=compute_league_ownership(snapshot, exclude_entry_id=my_entry_id),
        my_total_points=my_entry.total_points,
    )


def resolve_league_context(
    settings: AppSettingsData, league_id: int | None = None
) -> LeagueContext:
    """The requested league, or the first tracked one, resolved against the shared TTL cache.
    Returns the empty context rather than raising for every reason a league might not be usable:
    none configured, no FPL team ID saved, the fetch failing, the manager not being a member, or
    a league of one with no field to measure against."""
    resolved_league_id = (
        league_id
        if league_id is not None
        else (settings.mini_league_ids[0] if settings.mini_league_ids else None)
    )
    if resolved_league_id is None or settings.fpl_team_id is None:
        return _EMPTY_LEAGUE

    try:
        with FPLClient() as client:
            snapshot = get_cached_league_snapshot(client, resolved_league_id)
    except FPLClientError:
        return _EMPTY_LEAGUE

    return _league_context(snapshot, settings.fpl_team_id)


def _resolve_projections(app_state: AppState, gameweek: int) -> dict[int, PlayerGameweekProjection]:
    return {
        player_id: horizon.gameweeks[gameweek]
        for player_id, horizon in app_state.projections.items()
        if gameweek in horizon.gameweeks
    }


def _candidates(app_state: AppState, gameweeks: Sequence[int]) -> list[PlayerCandidate]:
    """Every priced, projected player valued at their total expected points across ``gameweeks``,
    which is what the integer program maximizes. A player projected for only some of those
    gameweeks is valued on the ones he has, matching
    :func:`~features.squad_points.projected_points`'s own treatment of a missing gameweek as a
    contribution of nothing rather than a reason to drop the player entirely.

    Deliberately the whole pool, not a pruned shortlist: CBC solves this program in well under a
    tenth of a second at the full FPL player base, so narrowing the pool would trade away real
    suggestions (a cheap enabler, an unfashionable mid-price upgrade) for a speed-up nothing
    needs.
    """
    return [
        PlayerCandidate(
            player_id=player_id,
            position=app_state.position_by_player[player_id],
            team_id=app_state.team_id_by_player[player_id],
            price=app_state.buy_prices[player_id],
            expected_points=sum(
                horizon.gameweeks[gameweek].expected_points
                for gameweek in gameweeks
                if gameweek in horizon.gameweeks
            ),
        )
        for player_id, horizon in app_state.projections.items()
        if player_id in app_state.buy_prices and player_id in app_state.position_by_player
    ]


@dataclass
class _CacheEntry:
    key: tuple
    suggestion: TransferSuggestion
    league: LeagueContext


_suggestion_cache: dict[tuple, _CacheEntry] = {}
_CACHE_LIMIT = 8


def reset_suggestion_cache() -> None:
    """Test-only: clear the cache so the next call always re-solves."""
    _suggestion_cache.clear()


def build_transfer_suggestion(
    app_state: AppState,
    team_state: MyTeamState,
    settings: AppSettingsData,
    budget: int,
    max_transfers: int = 1,
    gameweeks: Sequence[int] | None = None,
    chip: str | None = None,
    league_id: int | None = None,
    n_plans: int = DEFAULT_PLAN_COUNT,
    season_length_gameweeks: int = DEFAULT_SEASON_LENGTH_GAMEWEEKS,
) -> tuple[TransferSuggestion, LeagueContext]:
    """Assemble and solve the banner's suggestion, or return the cached one for identical inputs.

    ``budget`` is the squad's own persisted spend ceiling (the classic 100m, or the higher
    figure recorded when a real squad already worth more than that was imported), so a suggested
    plan is affordable under exactly the rule every manual add on this page is already checked
    against.

    ``gameweeks`` defaults to the decision gameweek alone. The league math is always measured at
    the decision gameweek even when the planning horizon is longer, since rival picks are only
    known for one gameweek at a time (a rival's squad two gameweeks out is unknowable, not merely
    unfetched) and inventing a horizon for them would be the one place this feature guessed.
    """
    if not 1 <= max_transfers <= MAX_TRANSFERS:
        raise ValueError(f"transfers must be between 1 and {MAX_TRANSFERS}, got {max_transfers}")

    league_gameweek = app_state.decision_gameweek
    planning_gameweeks = tuple(gameweeks) if gameweeks else (league_gameweek,)
    league = resolve_league_context(settings, league_id)

    key = (
        frozenset(team_state.player_ids),
        team_state.captain_id,
        max_transfers,
        planning_gameweeks,
        budget,
        chip,
        league.league_id,
        league.picks_gameweek,
        n_plans,
    )
    cached = _suggestion_cache.get(key)
    if cached is not None:
        return cached.suggestion, cached.league

    suggestion = plan_transfers(
        team_state,
        _candidates(app_state, planning_gameweeks),
        app_state.projections,
        planning_gameweeks,
        _resolve_projections(app_state, league_gameweek),
        league.ownership_by_player,
        league.rivals,
        league_gameweek=league_gameweek,
        my_total_points=league.my_total_points,
        gameweeks_remaining=max(season_length_gameweeks - league_gameweek, 0),
        budget=budget,
        max_transfers=max_transfers,
        chip=chip,
        n_plans=n_plans,
    )

    if len(_suggestion_cache) >= _CACHE_LIMIT:
        _suggestion_cache.pop(next(iter(_suggestion_cache)))
    _suggestion_cache[key] = _CacheEntry(key=key, suggestion=suggestion, league=league)
    return suggestion, league


def marginal_gains(suggestion: TransferSuggestion) -> list[float]:
    """What each additional transfer buys, in expected points, from
    ``suggestion.best_by_transfer_count``. Index 0 is the first transfer's own gain, index 1 what
    the second adds on top of it, and so on. Empty when no plan was found at any count."""
    gains: list[float] = []
    previous = 0.0
    for plan in suggestion.best_by_transfer_count:
        gains.append(plan.expected_points_delta - previous)
        previous = plan.expected_points_delta
    return gains


def ownership_of(player_id: int, league: LeagueContext) -> float | None:
    """A player's effective ownership across the field, or ``None`` when no league is in play.

    A player absent from the ownership map is ``0.0``, not ``None``: nobody in the league owning
    him is the strongest possible statement of his differential value, and the exact one the
    banner most wants to make. Only the genuine absence of a league gives ``None``, so the two
    cases stay distinguishable downstream instead of both rendering as no answer.
    """
    if league.league_id is None:
        return None
    ownership = league.ownership_by_player.get(player_id)
    return 0.0 if ownership is None else ownership.eo_multiplier
