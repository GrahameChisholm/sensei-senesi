"""Build a mini-league snapshot from FPL classic-league (entry) API responses (MINI_LEAGUE_PLAN
Phase 2) -- the league-scoped counterpart to :mod:`~engine.data.team_state_builder`'s single-entry
squad assembly.

Unlike :func:`~engine.data.team_state_builder.build_my_team_state`, which shapes JSON the API layer
has already fetched for one entry, this module does its own fetch orchestration (matching
:func:`~engine.data.player_history.load_live_player_history`'s precedent, not
``team_state_builder``'s): a league snapshot needs one standings call per page plus one picks call
and one history call per rival, and that fan-out belongs in one tested module rather than inlined
in ``api/``.

**Picks-gameweek resolution (MINI_LEAGUE_PLAN M1).** Rival picks are not public until a gameweek's
deadline passes -- the picks endpoint 404s for a gameweek that hasn't been played yet. Deadlines are
league-wide FPL calendar state, not a per-manager fact, so this is resolved once, the same way
``api.main.import_squad`` already resolves it for a single entry: probe an entry's ``get_entry``
response for ``current_event`` (FPL's own "the most recent gameweek that has a picks record"
field) and use that gameweek for every rival's picks call. The resolved value is carried on
:class:`LeagueSnapshot` as ``picks_gameweek`` precisely so a caller can tell it apart from the app's
own current gameweek and render the staleness this implies, rather than silently presenting one
gameweek's data as if it were another's.

The probe tries the fetched entries **in order** (standings rank) and falls through to the next
one on an :class:`~engine.data.fpl_client.FPLClientError` -- a single manager's account being
deleted, banned, or otherwise inaccessible (a real, observed FPL 404, unrelated to the league or
the calendar) must not take down the whole league's fetch just because that manager happened to be
first in the standings. Only if every fetched entry fails does the error propagate.

**Multiplier is the single source of truth for what a pick was worth (M3).** FPL has already
applied every chip effect (captain, triple captain, bench boost) to each pick's ``multiplier`` --
this module reads it as-is rather than reconstructing chip effects from ``active_chip`` and bench
position, so it stays correct through a change to the chip rules. Every one of a rival's 15 picks
is kept, including benched players at ``multiplier == 0``, since a player's mere presence in the 15
(M6's raw ownership) and their multiplier (effective ownership) are two different signals
downstream.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from engine.data.fpl_client import FPLClient, FPLClientError

__all__ = [
    "DEFAULT_RIVAL_LIMIT",
    "ChipUsage",
    "LeagueEntry",
    "LeagueSnapshot",
    "build_league_snapshot",
]

# MINI_LEAGUE_PLAN M16: one page is 50 entries; a caller fetches whole pages up to this cap rather
# than an arbitrary partial page, so the entry count reported is always a real page boundary.
DEFAULT_RIVAL_LIMIT = 50


@dataclass(frozen=True)
class ChipUsage:
    """One chip played by one entry -- ``name`` is passed through verbatim from FPL rather than
    mapped onto a fixed enum, so a chip this codebase doesn't yet recognise (a rules change) is
    still carried and displayable instead of silently dropped."""

    name: str
    gameweek: int


@dataclass(frozen=True)
class LeagueEntry:
    """One manager's row in the league, as of :class:`LeagueSnapshot`'s ``picks_gameweek``.

    ``picks`` maps every one of the entry's 15 player_ids to that player's ``multiplier`` for
    ``picks_gameweek`` (0 for a benched player with no Bench Boost active, 1 started, 2 captain, 3
    triple captain) -- see the module docstring's M3 note for why this is read as-is rather than
    reconstructed.
    """

    entry_id: int
    manager_name: str
    team_name: str
    rank: int
    total_points: int
    gameweek_points: int
    picks: dict[int, int]
    chips: tuple[ChipUsage, ...]


@dataclass(frozen=True)
class LeagueSnapshot:
    """A full mini-league as of one fetch -- every entry (including the caller's own; the caller
    knows its own entry_id from settings and excludes it where that matters, e.g.
    ``features.mini_league``'s effective-ownership denominator, rather than this module guessing at
    it). ``picks_gameweek`` is ``0`` only in the degenerate case of a league with no entries at all,
    since there is then no probe entry to resolve it from."""

    league_id: int
    league_name: str
    picks_gameweek: int
    entries: tuple[LeagueEntry, ...]


def _fetch_standings_results(
    client: FPLClient, league_id: int, limit: int
) -> tuple[str, list[dict]]:
    """Pages through ``FPLClient.get_league_standings`` (M16) until ``limit`` entries are
    collected or the league runs out of pages, whichever comes first."""
    results: list[dict] = []
    league_name = ""
    page = 1
    while len(results) < limit:
        payload = client.get_league_standings(league_id, page=page)
        if page == 1:
            league_name = payload["league"]["name"]
        standings = payload["standings"]
        results.extend(standings["results"])
        if not standings.get("has_next"):
            break
        page += 1
    return league_name, results[:limit]


def _resolve_picks_gameweek(client: FPLClient, entry_ids: Sequence[int]) -> int:
    """Try each of ``entry_ids`` in order, falling through to the next on an
    :class:`FPLClientError` (a single inaccessible manager, not a league-wide problem -- see the
    module docstring's M1 note). ``entry_ids`` is guaranteed non-empty by the caller
    (:func:`build_league_snapshot` only calls this once it has at least one standings result), so
    if every entry fails, the last error is the one raised.
    """
    last_error: FPLClientError | None = None
    for entry_id in entry_ids:
        try:
            return client.get_entry(entry_id)["current_event"]
        except FPLClientError as exc:
            last_error = exc
    assert last_error is not None
    raise last_error


def _build_entry(client: FPLClient, result: dict, picks_gameweek: int) -> LeagueEntry | None:
    """Returns ``None`` if this specific entry's picks or chip history can't be fetched (the same
    "one manager's account is inaccessible" failure the picks-gameweek probe falls through on,
    just discovered later, once we're already committed to this ``picks_gameweek``) -- one rival
    being unreachable is real, if unhelpful, information, not a reason to fail every other rival's
    fetch too."""
    entry_id = result["entry"]
    try:
        picks_payload = client.get_entry_picks(entry_id, picks_gameweek)
        history = client.get_entry_history(entry_id)
    except FPLClientError:
        return None
    picks = {pick["element"]: pick["multiplier"] for pick in picks_payload["picks"]}

    chips = tuple(
        ChipUsage(name=chip["name"], gameweek=chip["event"]) for chip in history.get("chips", [])
    )

    return LeagueEntry(
        entry_id=entry_id,
        manager_name=result.get("player_name", ""),
        team_name=result.get("entry_name", ""),
        rank=result["rank"],
        total_points=result["total"],
        gameweek_points=result.get("event_total", 0),
        picks=picks,
        chips=chips,
    )


def build_league_snapshot(
    client: FPLClient, league_id: int, limit: int = DEFAULT_RIVAL_LIMIT
) -> LeagueSnapshot:
    """Fetch and assemble one classic mini-league's full snapshot: standings (paginated up to
    ``limit``), each entry's picks at the resolved ``picks_gameweek`` (M1), and each entry's chip
    history (M11). An unknown or private *league* ID surfaces as whatever
    :class:`~engine.data.fpl_client.FPLClientError` the underlying standings request raises -- this
    module does not translate it, matching
    :func:`~engine.data.team_state_builder.build_my_team_state`'s own "engine-layer errors stay
    engine-layer" convention; the caller (``api/``) is responsible for turning that into
    caller-facing 400.

    A single *entry* within an otherwise-valid league being unreachable (a deleted, banned, or
    otherwise inaccessible manager account -- an observed real-world FPL 404, not a hypothetical)
    is a different situation: it says something about that one manager, not about the league ID
    the caller asked for, so it is not raised. Such an entry is simply absent from ``entries``
    rather than aborting the whole fetch (:func:`_resolve_picks_gameweek` falls through to another
    entry for the probe; :func:`_build_entry` drops the entry itself if its picks/history can't be
    fetched even after ``picks_gameweek`` was resolved from someone else).
    """
    league_name, results = _fetch_standings_results(client, league_id, limit)
    if not results:
        return LeagueSnapshot(
            league_id=league_id, league_name=league_name, picks_gameweek=0, entries=()
        )

    picks_gameweek = _resolve_picks_gameweek(client, [result["entry"] for result in results])
    entries = tuple(
        entry
        for entry in (_build_entry(client, result, picks_gameweek) for result in results)
        if entry is not None
    )
    return LeagueSnapshot(
        league_id=league_id,
        league_name=league_name,
        picks_gameweek=picks_gameweek,
        entries=entries,
    )
