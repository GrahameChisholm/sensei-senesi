"""Mini League page row assembly (MINI_LEAGUE_PLAN Phase 5) -- turns an already-fetched
:class:`~engine.data.league_state_builder.LeagueSnapshot` plus already-loaded ``AppState`` into the
one bulk response ``GET /mini-league/{league_id}`` serves (M17). All the actual league math is
``features.mini_league``'s job; this module's own job is resolving this gameweek's projections
once and handing every function the same resolved inputs, exactly the role
``api.differentials_panel``/``api.panel`` play for their own pages.

Also owns the in-process TTL cache over :func:`~engine.data.league_state_builder.build_league_
snapshot` (M15): rival picks physically cannot change between FPL deadlines (MINI_LEAGUE_PLAN M1),
so a live re-fetch on every page load is wasted work, not freshness. The plan's original framing
keyed this cache on ``(league_id, picks_gameweek)`` -- impossible in practice, since
``picks_gameweek`` is only known *after* a fetch resolves it, which is exactly the chicken-and-egg
problem a pre-fetch cache key can't solve. Keyed on ``(league_id, limit)`` with a plain time-based
TTL instead, which delivers the same practical behaviour the plan wanted (don't re-hit FPL on
every reload; a manual ``refresh`` still busts it).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from api.state import AppState
from engine.data.fpl_client import FPLClient
from engine.data.league_state_builder import (
    DEFAULT_RIVAL_LIMIT,
    LeagueSnapshot,
    build_league_snapshot,
)
from engine.projections import PlayerGameweekProjection
from features.mini_league import (
    CaptainOption,
    HeadToHead,
    PlayerExposure,
    RivalChipState,
    RivalPosture,
    compute_chip_states,
    compute_coverage,
    compute_exposures,
    compute_head_to_head,
    compute_league_ownership,
    compute_posture,
    league_template_xi,
    rank_captain_options,
)
from features.team_state import MyTeamState

__all__ = [
    "DEFAULT_SEASON_LENGTH_GAMEWEEKS",
    "DEFAULT_TEMPLATE_XI_SIZE",
    "CACHE_TTL_SECONDS",
    "RivalRow",
    "MiniLeaguePanel",
    "build_mini_league_panel",
    "get_cached_league_snapshot",
    "reset_snapshot_cache",
]

# A real Premier League season -- used only to derive "gameweeks remaining" for
# features.mini_league.compute_posture; mirrors web/src/pages/FixturesPage.tsx's own
# MAX_GAMEWEEK constant, the closest existing precedent for hardcoding this number.
DEFAULT_SEASON_LENGTH_GAMEWEEKS = 38

DEFAULT_TEMPLATE_XI_SIZE = 11

CACHE_TTL_SECONDS = 600.0


@dataclass(frozen=True)
class RivalRow:
    """One rival's full row for the Mini League page's standings panel -- identity/points from the
    snapshot, plus every per-rival number ``features.mini_league`` computes."""

    entry_id: int
    manager_name: str
    team_name: str
    rank: int
    total_points: int
    gameweek_points: int
    chip_state: RivalChipState
    posture: RivalPosture
    head_to_head: HeadToHead


@dataclass(frozen=True)
class MiniLeaguePanel:
    """The whole bulk response (M17) -- one round trip per league, since a 15-30 differential-row
    head-to-head per rival is small even for a 50-rival league, and it means switching the
    selected rival in the UI is instant with no further round trip."""

    league_id: int
    league_name: str
    picks_gameweek: int
    gameweek: int
    my_rank: int
    my_total_points: int
    coverage: float
    template_xi: tuple[int, ...]
    exposures: tuple[PlayerExposure, ...]
    captain_options: tuple[CaptainOption, ...]
    rivals: tuple[RivalRow, ...]


def _resolve_projections(app_state: AppState, gameweek: int) -> dict[int, PlayerGameweekProjection]:
    return {
        player_id: horizon.gameweeks[gameweek]
        for player_id, horizon in app_state.projections.items()
        if gameweek in horizon.gameweeks
    }


def build_mini_league_panel(
    app_state: AppState,
    team_state: MyTeamState,
    snapshot: LeagueSnapshot,
    my_entry_id: int,
    chip: str | None = None,
    season_length_gameweeks: int = DEFAULT_SEASON_LENGTH_GAMEWEEKS,
) -> MiniLeaguePanel:
    """Assemble the full panel for ``app_state.gameweek`` -- always the app's own current decision
    gameweek, never ``snapshot.picks_gameweek`` (M1's separate, possibly-lagging "what picks data
    do we actually have" field). Raises ``ValueError`` if ``my_entry_id`` isn't actually present in
    ``snapshot`` -- not a member of this league, or beyond the fetched rival limit -- since there
    is then no sensible "the field, excluding me" to compute.
    """
    my_entry = next((entry for entry in snapshot.entries if entry.entry_id == my_entry_id), None)
    if my_entry is None:
        raise ValueError(
            f"FPL entry {my_entry_id} was not found in league {snapshot.league_id} "
            "(not a member of this league, or beyond the fetched rival limit)"
        )

    gameweek = app_state.gameweek
    projections = _resolve_projections(app_state, gameweek)
    ownership = compute_league_ownership(snapshot, exclude_entry_id=my_entry_id)

    exposure_player_ids = set(team_state.player_ids) | set(ownership)
    exposures = tuple(
        compute_exposures(exposure_player_ids, team_state, ownership, projections, chip=chip)
    )
    captain_options = tuple(rank_captain_options(team_state.starting_xi, ownership, projections))
    coverage = compute_coverage(team_state, ownership, chip=chip)
    template_xi = league_template_xi(ownership, n=DEFAULT_TEMPLATE_XI_SIZE)

    gameweeks_remaining = max(season_length_gameweeks - gameweek, 0)
    chip_state_by_entry = {state.entry_id: state for state in compute_chip_states(snapshot.entries)}

    rivals: list[RivalRow] = []
    for entry in snapshot.entries:
        if entry.entry_id == my_entry_id:
            continue
        head_to_head = compute_head_to_head(team_state, entry, projections, chip=chip)
        posture = compute_posture(my_entry.total_points, entry, head_to_head, gameweeks_remaining)
        rivals.append(
            RivalRow(
                entry_id=entry.entry_id,
                manager_name=entry.manager_name,
                team_name=entry.team_name,
                rank=entry.rank,
                total_points=entry.total_points,
                gameweek_points=entry.gameweek_points,
                chip_state=chip_state_by_entry[entry.entry_id],
                posture=posture,
                head_to_head=head_to_head,
            )
        )

    return MiniLeaguePanel(
        league_id=snapshot.league_id,
        league_name=snapshot.league_name,
        picks_gameweek=snapshot.picks_gameweek,
        gameweek=gameweek,
        my_rank=my_entry.rank,
        my_total_points=my_entry.total_points,
        coverage=coverage,
        template_xi=template_xi,
        exposures=exposures,
        captain_options=captain_options,
        rivals=tuple(rivals),
    )


@dataclass
class _CacheEntry:
    fetched_at: float
    snapshot: LeagueSnapshot


_snapshot_cache: dict[tuple[int, int], _CacheEntry] = {}


def get_cached_league_snapshot(
    client: FPLClient, league_id: int, limit: int = DEFAULT_RIVAL_LIMIT, refresh: bool = False
) -> LeagueSnapshot:
    """A fresh :func:`~engine.data.league_state_builder.build_league_snapshot` call, or the cached
    one from within the last :data:`CACHE_TTL_SECONDS` if ``refresh`` is ``False`` (M15).
    ``refresh`` bypasses the cache for this call and overwrites the cached entry, for the moments
    right after a deadline when a manager wants this gameweek's data before the TTL would
    naturally expire.
    """
    key = (league_id, limit)
    if not refresh:
        cached = _snapshot_cache.get(key)
        if cached is not None and (time.monotonic() - cached.fetched_at) < CACHE_TTL_SECONDS:
            return cached.snapshot

    snapshot = build_league_snapshot(client, league_id, limit=limit)
    _snapshot_cache[key] = _CacheEntry(fetched_at=time.monotonic(), snapshot=snapshot)
    return snapshot


def reset_snapshot_cache() -> None:
    """Test-only: clear the cache so the next call always fetches."""
    _snapshot_cache.clear()
