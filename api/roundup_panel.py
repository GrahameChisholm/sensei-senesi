"""Roundup page row assembly (ROUNDUP_PLAN) -- turns an already-fetched
:class:`~engine.data.league_state_builder.LeagueSnapshot` (fetched for one specific, already
``data_checked`` gameweek), that gameweek's live per-player points, and live bootstrap player/team
data into the one bulk response ``GET /roundup/{league_id}`` serves. All the actual recap math is
``features.roundup``'s job; this module's own job is resolving the template XI (via
``features.mini_league``, since the template feeds :func:`~features.roundup.template_overlap`)
and standings-by-gameweek once, then handing every function the same resolved inputs -- the same
role ``api.mini_league_panel`` plays for the Mini League page.

Unlike the Mini League page, the Roundup page has no "me": every manager is treated equally (no
self-highlight, per the design), so nothing here needs a saved ``fpl_team_id`` or a complete
squad. It also does not depend on the projections cache at all -- everything is read live from
FPL, via :class:`~engine.data.fpl_client.FPLClient`, so this page works correctly even before the
first ``build_projections`` run of a new gameweek.

Caching (ROUNDUP_PLAN): a *fully assembled* :class:`RoundupPanel` is cached indefinitely per
``(league_id, gameweek)`` -- unlike the Mini League page's 600s TTL, a ``data_checked`` gameweek's
recap cannot change (FPL has already finalised bonus points), so there is nothing to expire. A
``refresh`` parameter bypasses and overwrites, for the rare case FPL retroactively corrects a
result.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.data.fpl_client import FPLClient
from engine.data.league_state_builder import LeagueSnapshot, build_league_snapshot
from features.mini_league import compute_league_ownership, league_template_xi
from features.roundup import (
    BenchRegretRow,
    CaptainReturn,
    ChipSummary,
    DifferentialHaulRow,
    ManagerGameweekRow,
    Mover,
    PlayerRoundupRow,
    TemplateOverlapRow,
    bench_regret,
    biggest_differential_haul,
    biggest_movers,
    captain_scoreboard,
    chips_played,
    highest_scoring_owned_player,
    standings_by_gameweek,
    template_overlap,
    top_and_bottom,
)

__all__ = [
    "RoundupPanel",
    "build_roundup_panel",
    "resolve_latest_complete_gameweek",
    "get_cached_roundup_panel",
    "reset_roundup_cache",
]


def resolve_latest_complete_gameweek(events: list[dict]) -> int:
    """The highest gameweek FPL has fully confirmed (``data_checked``) -- the gameweek a recap
    defaults to when the caller doesn't name one. Mirrors
    ``scripts.build_projections.resolve_build_gameweek``'s own "earliest not yet data_checked"
    logic, mirrored rather than reused since that function answers the opposite question (what to
    build next, not what to look back on). Raises if nothing is ``data_checked`` yet (true GW1,
    before any gameweek has finished) -- there is no gameweek to recap.
    """
    checked = [event["id"] for event in events if event.get("data_checked")]
    if not checked:
        raise ValueError(
            "no gameweek is data_checked yet -- there is nothing to recap until the first "
            "gameweek of the season finishes"
        )
    return max(checked)


@dataclass(frozen=True)
class RoundupPanel:
    league_id: int
    league_name: str
    gameweek: int
    standings_by_gameweek: dict[int, tuple[int, ...]]
    top_managers: tuple[ManagerGameweekRow, ...]
    bottom_managers: tuple[ManagerGameweekRow, ...]
    top_player: PlayerRoundupRow | None
    captain_returns: tuple[CaptainReturn, ...]
    bench_regret: tuple[BenchRegretRow, ...]
    movers: tuple[Mover, ...]
    template_xi: tuple[int, ...]
    template_overlap: tuple[TemplateOverlapRow, ...]
    differential_hauls: tuple[DifferentialHaulRow, ...]
    chips: tuple[ChipSummary, ...]


def build_roundup_panel(
    snapshot: LeagueSnapshot,
    live_points: dict[int, int],
    position_by_player: dict[int, str],
    gameweek: int,
) -> RoundupPanel:
    """Assemble the full Roundup panel for ``gameweek`` -- ``snapshot`` must already be fetched at
    that exact gameweek (:func:`~engine.data.league_state_builder.build_league_snapshot`'s
    ``gameweek`` argument), and ``live_points`` from that gameweek's
    :meth:`~engine.data.fpl_client.FPLClient.get_event_live`.
    """
    entries = snapshot.entries
    standings = standings_by_gameweek(entries, up_to_gameweek=gameweek)

    ownership = compute_league_ownership(snapshot, exclude_entry_id=None)
    template_xi = league_template_xi(ownership, position_by_player) if ownership else ()

    top_managers, bottom_managers = top_and_bottom(entries, gameweek)

    return RoundupPanel(
        league_id=snapshot.league_id,
        league_name=snapshot.league_name,
        gameweek=gameweek,
        standings_by_gameweek=standings,
        top_managers=top_managers,
        bottom_managers=bottom_managers,
        top_player=highest_scoring_owned_player(entries, live_points),
        captain_returns=captain_scoreboard(entries, live_points),
        bench_regret=bench_regret(entries, live_points),
        movers=biggest_movers(entries, standings, gameweek),
        template_xi=template_xi,
        template_overlap=template_overlap(entries, template_xi),
        differential_hauls=biggest_differential_haul(entries, live_points),
        chips=chips_played(entries, live_points, standings, gameweek),
    )


_roundup_cache: dict[tuple[int, int], RoundupPanel] = {}


def get_cached_roundup_panel(
    client: FPLClient,
    league_id: int,
    gameweek: int,
    live_points: dict[int, int],
    position_by_player: dict[int, str],
    refresh: bool = False,
) -> RoundupPanel:
    """A fresh :func:`build_roundup_panel` call, or the cached one for this exact
    ``(league_id, gameweek)`` -- never expired, since a ``data_checked`` gameweek's recap cannot
    change once FPL has finalised it. ``refresh`` bypasses and overwrites, for the rare case FPL
    retroactively corrects a result.
    """
    key = (league_id, gameweek)
    if not refresh and key in _roundup_cache:
        return _roundup_cache[key]

    snapshot = build_league_snapshot(client, league_id, gameweek=gameweek)
    panel = build_roundup_panel(snapshot, live_points, position_by_player, gameweek)
    _roundup_cache[key] = panel
    return panel


def reset_roundup_cache() -> None:
    """Test-only: clear the cache so the next call always rebuilds."""
    _roundup_cache.clear()
