"""Gameweek recap math for the Roundup page (ROUNDUP_PLAN) -- every number the shareable
GW recap poster needs, computed purely over an already-fetched
:class:`~engine.data.league_state_builder.LeagueSnapshot` (fetched for one specific gameweek, see
:func:`~engine.data.league_state_builder.build_league_snapshot`'s ``gameweek`` argument) plus that
gameweek's live per-player actuals (``FPLClient.get_event_live``, ``player_id -> total_points``).
No I/O, no HTTP, matching every other module in this package.

Backward-looking, not forward-looking. This is the retrospective counterpart to
``features.mini_league``, which answers "what should I do this week" from projections; this
module answers "what actually happened last week" from live results. The two modules are kept
separate rather than merged since they read fundamentally different inputs (projections vs.
actuals) even though they share the same ``LeagueSnapshot``/``LeagueEntry`` shapes.

**Two distinct "did this count" concepts, deliberately never collapsed into one:**

1. **Pick position** (``LeagueEntry.pick_positions``, 1-11 started / 12-15 bench). Used only by
   :func:`template_overlap`, via ``features.mini_league.league_template_xi``'s own
   ``starter_count`` maximand -- a manager's *chosen* XI, unaffected by a Bench Boost gameweek.
2. **Multiplier** (``LeagueEntry.picks``, 0 benched / 1 started / 2 captain / 3 triple captain).
   Used by every points-scoring number below (:func:`bench_regret`, :func:`captain_scoreboard`,
   :func:`chips_played`) -- FPL's own automatic-substitution-aware truth for what actually
   counted, matching the M3 convention already documented on
   :class:`~engine.data.league_state_builder.LeagueEntry`.

**Standings-sourced fields are never read for a specific gameweek's numbers.**
``LeagueEntry.rank``/``total_points``/``gameweek_points`` come from FPL's live standings
endpoint, which always reflects the *current* state, not necessarily the gameweek this recap is
about (stepping back to view an earlier gameweek's roundup while a later one is underway would
silently show today's rank instead). Every gameweek-specific number here is read from
``LeagueEntry.history`` instead, which carries one immutable row per gameweek that was actually
played. An entry missing a history row for the requested gameweek (a rival who joined the league
late) is simply absent from that function's result, never a fabricated zero.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.data.league_state_builder import LeagueEntry

__all__ = [
    "ManagerGameweekRow",
    "PlayerRoundupRow",
    "CaptainReturn",
    "BenchRegretRow",
    "Mover",
    "TemplateOverlapRow",
    "DifferentialHaulRow",
    "ChipSummary",
    "standings_by_gameweek",
    "top_and_bottom",
    "highest_scoring_owned_player",
    "captain_scoreboard",
    "bench_regret",
    "biggest_movers",
    "template_overlap",
    "biggest_differential_haul",
    "chips_played",
]


def _points_at_gameweek(entry: LeagueEntry, gameweek: int) -> int | None:
    return next((row.points for row in entry.history if row.event == gameweek), None)


def _total_points_at_gameweek(entry: LeagueEntry, gameweek: int) -> int | None:
    return next((row.total_points for row in entry.history if row.event == gameweek), None)


def standings_by_gameweek(
    entries: Sequence[LeagueEntry], up_to_gameweek: int
) -> dict[int, tuple[int, ...]]:
    """For every gameweek from 1 to ``up_to_gameweek``, every entry with a history row for that
    gameweek, ranked by cumulative points as of it (ties broken by ``entry_id`` for a
    deterministic result). FPL has no "league rank as of gameweek N" endpoint, so this is how
    both the bump chart and :func:`biggest_movers` get one: reconstructed entirely from
    ``LeagueEntry.history``, never from the live (current-only) standings fields.
    """
    result: dict[int, tuple[int, ...]] = {}
    for gameweek in range(1, up_to_gameweek + 1):
        rows = [
            (entry.entry_id, total)
            for entry in entries
            for total in [_total_points_at_gameweek(entry, gameweek)]
            if total is not None
        ]
        rows.sort(key=lambda row: (-row[1], row[0]))
        result[gameweek] = tuple(entry_id for entry_id, _ in rows)
    return result


@dataclass(frozen=True)
class ManagerGameweekRow:
    entry_id: int
    manager_name: str
    team_name: str
    gameweek_points: int


def top_and_bottom(
    entries: Sequence[LeagueEntry], gameweek: int, n: int = 3
) -> tuple[tuple[ManagerGameweekRow, ...], tuple[ManagerGameweekRow, ...]]:
    """The top and bottom ``n`` managers by this gameweek's points, threshold-based rather than
    index-based -- if the ``n``-th and ``(n + 1)``-th place tie on points, both are included,
    since dropping one arbitrarily would misrepresent an actual tie as a clean cutoff.
    """
    rows = [
        ManagerGameweekRow(entry.entry_id, entry.manager_name, entry.team_name, points)
        for entry in entries
        for points in [_points_at_gameweek(entry, gameweek)]
        if points is not None
    ]
    if not rows:
        return (), ()

    by_points_desc = sorted(rows, key=lambda row: (-row.gameweek_points, row.entry_id))
    by_points_asc = sorted(rows, key=lambda row: (row.gameweek_points, row.entry_id))

    top_threshold = by_points_desc[min(n, len(by_points_desc)) - 1].gameweek_points
    bottom_threshold = by_points_asc[min(n, len(by_points_asc)) - 1].gameweek_points

    top = tuple(row for row in by_points_desc if row.gameweek_points >= top_threshold)
    bottom = tuple(row for row in by_points_asc if row.gameweek_points <= bottom_threshold)
    return top, bottom


@dataclass(frozen=True)
class PlayerRoundupRow:
    """One player's actual result this gameweek, plus who in the league had him -- ``owner``
    means anywhere in the 15, ``starter`` means pick position 1-11, ``captain`` means multiplier
    at least 2 (captain or triple captain)."""

    player_id: int
    live_points: int
    owner_entry_ids: tuple[int, ...]
    starter_entry_ids: tuple[int, ...]
    captain_entry_ids: tuple[int, ...]


def highest_scoring_owned_player(
    entries: Sequence[LeagueEntry], live_points: Mapping[int, int]
) -> PlayerRoundupRow | None:
    """The single highest-scoring player owned by at least one entry in this league (ties broken
    by ``player_id`` for a deterministic result). ``None`` if no entry owns any player at all (an
    empty league)."""
    owned_ids = {player_id for entry in entries for player_id in entry.picks}
    if not owned_ids:
        return None

    best_id = max(owned_ids, key=lambda pid: (live_points.get(pid, 0), -pid))
    return PlayerRoundupRow(
        player_id=best_id,
        live_points=live_points.get(best_id, 0),
        owner_entry_ids=tuple(entry.entry_id for entry in entries if best_id in entry.picks),
        starter_entry_ids=tuple(
            entry.entry_id for entry in entries if entry.pick_positions.get(best_id, 12) <= 11
        ),
        captain_entry_ids=tuple(
            entry.entry_id for entry in entries if entry.picks.get(best_id, 0) >= 2
        ),
    )


@dataclass(frozen=True)
class CaptainReturn:
    """One entry's captaincy call and what it actually returned. ``captain_player_id`` and
    ``multiplier`` are ``None``/``0`` in the (real-world-observed but rare) case where no pick has
    multiplier >= 2 at all."""

    entry_id: int
    manager_name: str
    captain_player_id: int | None
    multiplier: int
    points: int


def captain_scoreboard(
    entries: Sequence[LeagueEntry], live_points: Mapping[int, int]
) -> tuple[CaptainReturn, ...]:
    """Every entry's captaincy return this gameweek. The caller picks the best/worst call as the
    max/min of the returned tuple's ``points`` -- this function does not rank, matching this
    package's "the frontend/caller sorts" convention."""
    rows = []
    for entry in entries:
        captain_id, multiplier = next(
            ((pid, m) for pid, m in entry.picks.items() if m >= 2), (None, 0)
        )
        points = 0 if captain_id is None else live_points.get(captain_id, 0) * multiplier
        rows.append(
            CaptainReturn(
                entry_id=entry.entry_id,
                manager_name=entry.manager_name,
                captain_player_id=captain_id,
                multiplier=multiplier,
                points=points,
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class BenchRegretRow:
    entry_id: int
    manager_name: str
    points_left_on_bench: int
    contributing_player_ids: tuple[int, ...]


def bench_regret(
    entries: Sequence[LeagueEntry], live_points: Mapping[int, int]
) -> tuple[BenchRegretRow, ...]:
    """Points each entry left on the bench: every pick with multiplier 0 (M3 -- already correct
    through automatic substitutions and any chip effect, so a Bench Boost gameweek naturally
    reports zero here for that entry). ``contributing_player_ids`` lists only the benched players
    who actually scored, sorted by descending points, since a benched player on 0 isn't "regret."
    """
    rows = []
    for entry in entries:
        benched = [pid for pid, multiplier in entry.picks.items() if multiplier == 0]
        scored = sorted(
            (pid for pid in benched if live_points.get(pid, 0) > 0),
            key=lambda pid: -live_points.get(pid, 0),
        )
        rows.append(
            BenchRegretRow(
                entry_id=entry.entry_id,
                manager_name=entry.manager_name,
                points_left_on_bench=sum(live_points.get(pid, 0) for pid in benched),
                contributing_player_ids=tuple(scored),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class Mover:
    entry_id: int
    manager_name: str
    rank_before: int
    rank_after: int
    delta: int  # positive = climbed (rank number got smaller)


def biggest_movers(
    entries: Sequence[LeagueEntry], standings: Mapping[int, tuple[int, ...]], gameweek: int
) -> tuple[Mover, ...]:
    """Every entry's league-rank change from ``gameweek - 1`` to ``gameweek``, sorted by
    descending ``delta`` (biggest climb first, biggest fall last). Empty at gameweek 1, or if
    either gameweek is missing from ``standings`` (typically :func:`standings_by_gameweek`'s
    output) -- there is no prior rank to compare against, matching this codebase's "no baseline,
    no rows" precedent (e.g. ``features.fixture_swing``'s own preseason handling).
    """
    if gameweek <= 1:
        return ()
    before = standings.get(gameweek - 1)
    after = standings.get(gameweek)
    if not before or not after:
        return ()

    rank_before = {entry_id: i + 1 for i, entry_id in enumerate(before)}
    rank_after = {entry_id: i + 1 for i, entry_id in enumerate(after)}
    manager_name = {entry.entry_id: entry.manager_name for entry in entries}

    movers = []
    for entry_id in rank_after:
        if entry_id not in rank_before:
            continue
        movers.append(
            Mover(
                entry_id=entry_id,
                manager_name=manager_name.get(entry_id, ""),
                rank_before=rank_before[entry_id],
                rank_after=rank_after[entry_id],
                delta=rank_before[entry_id] - rank_after[entry_id],
            )
        )
    movers.sort(key=lambda mover: (-mover.delta, mover.entry_id))
    return tuple(movers)


@dataclass(frozen=True)
class TemplateOverlapRow:
    entry_id: int
    manager_name: str
    overlap_count: int
    template_size: int


def template_overlap(
    entries: Sequence[LeagueEntry], template_xi_ids: Sequence[int]
) -> tuple[TemplateOverlapRow, ...]:
    """How many of the league's template XI (``features.mini_league.league_template_xi``) each
    entry actually started (pick position 1-11) this gameweek. The highest is "Sheep of the
    Week" -- most template."""
    template_set = set(template_xi_ids)
    rows = []
    for entry in entries:
        starters = {pid for pid, position in entry.pick_positions.items() if position <= 11}
        rows.append(
            TemplateOverlapRow(
                entry_id=entry.entry_id,
                manager_name=entry.manager_name,
                overlap_count=len(starters & template_set),
                template_size=len(template_set),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class DifferentialHaulRow:
    entry_id: int
    manager_name: str
    differential_player_ids: tuple[int, ...]
    total_points: int


def biggest_differential_haul(
    entries: Sequence[LeagueEntry], live_points: Mapping[int, int]
) -> tuple[DifferentialHaulRow, ...]:
    """How many actual points each entry got from true differentials: players in their 15 that no
    *other* entry in the league owns at all. The highest is "Hipster of the Week" -- not the
    inverse of :func:`template_overlap` (fewest template-XI starters), a different and more
    specific question: how much did going against the grain actually pay this week."""
    owner_count: dict[int, int] = {}
    for entry in entries:
        for player_id in entry.picks:
            owner_count[player_id] = owner_count.get(player_id, 0) + 1

    rows = []
    for entry in entries:
        differentials = tuple(sorted(pid for pid in entry.picks if owner_count.get(pid, 0) == 1))
        rows.append(
            DifferentialHaulRow(
                entry_id=entry.entry_id,
                manager_name=entry.manager_name,
                differential_player_ids=differentials,
                total_points=sum(live_points.get(pid, 0) for pid in differentials),
            )
        )
    return tuple(rows)


@dataclass(frozen=True)
class ChipSummary:
    """One chip played this gameweek. ``points_effect`` is the chip's own points contribution:
    the bench points it added for Bench Boost, one extra multiplier's worth of the captain's
    points for Triple Captain, or ``None`` for Wildcard/Free Hit (and any chip name this module
    doesn't recognise), which have no direct points effect of their own to report."""

    entry_id: int
    manager_name: str
    chip_name: str
    resulting_rank: int | None
    points_effect: int | None


def chips_played(
    entries: Sequence[LeagueEntry],
    live_points: Mapping[int, int],
    standings: Mapping[int, tuple[int, ...]],
    gameweek: int,
) -> tuple[ChipSummary, ...]:
    """Every chip played in ``gameweek``, with a chip-specific points effect. ``resulting_rank``
    comes from ``standings`` (:func:`standings_by_gameweek`'s output), never a live standings
    field, for the same reason :func:`biggest_movers` does -- it must reflect this gameweek, not
    today."""
    rank_after = {entry_id: i + 1 for i, entry_id in enumerate(standings.get(gameweek, ()))}

    rows = []
    for entry in entries:
        for chip in entry.chips:
            if chip.gameweek != gameweek:
                continue
            if chip.name == "bboost":
                points_effect = sum(
                    live_points.get(pid, 0)
                    for pid, position in entry.pick_positions.items()
                    if position >= 12
                )
            elif chip.name == "3xc":
                captain_id = next((pid for pid, m in entry.picks.items() if m == 3), None)
                points_effect = None if captain_id is None else live_points.get(captain_id, 0)
            else:
                points_effect = None
            rows.append(
                ChipSummary(
                    entry_id=entry.entry_id,
                    manager_name=entry.manager_name,
                    chip_name=chip.name,
                    resulting_rank=rank_after.get(entry.entry_id),
                    points_effect=points_effect,
                )
            )
    return tuple(rows)
