"""Starting XI / bench-order selection and FPL's real autosub rule (``planning/
SEASON_SIMULATOR.md`` component 3's "Formation/starting XI" and "Autosubs" bullets) — nothing in
``features/`` picks a valid formation today (``features.chips.evaluate_wildcard`` explicitly
disclaims a full optimal-XI search as out of v1 scope, and ``features.team_state.MyTeamState``
only validates squad/XI *counts*, never position-legality), so this is genuinely new logic, not a
reuse of anything that already existed.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from engine.scoring import DEF, FWD, GK, MID
from features.team_state import SquadPlayer

__all__ = ["VALID_FORMATIONS", "select_starting_xi", "apply_autosubs"]

# Every legal FPL formation: (DEF, MID, FWD) summing to 10 outfield starters, with exactly 1 GK
# always separate — FPL's own constraints are 3-5 DEF, 2-5 MID, 1-3 FWD.
VALID_FORMATIONS: tuple[tuple[int, int, int], ...] = tuple(
    (d, m, f) for d in range(3, 6) for m in range(2, 6) for f in range(1, 4) if d + m + f == 10
)


def select_starting_xi(
    squad: Sequence[SquadPlayer], expected_points: Mapping[int, float]
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Highest-total-EV valid formation: exactly 1 GK (the squad's higher-EV one), and whichever
    (DEF, MID, FWD) split in :data:`VALID_FORMATIONS` the top-N-by-EV picks per position sum
    highest for. Bench order: the reserve GK is always last (real FPL — it can only come on for
    the starting GK), the three outfield reserves ordered by descending EV (real FPL's own
    priority order is manager-set; highest-EV-first is the only sane default with no explicit
    manager preference to reproduce).
    """
    by_position: dict[str, list[SquadPlayer]] = {GK: [], DEF: [], MID: [], FWD: []}
    for player in squad:
        by_position[player.position].append(player)
    for players in by_position.values():
        players.sort(key=lambda p: expected_points.get(p.player_id, 0.0), reverse=True)

    if not by_position[GK]:
        raise ValueError("squad has no goalkeeper")

    best_score = float("-inf")
    best_combo: tuple[int, ...] | None = None
    for d, m, f in VALID_FORMATIONS:
        if len(by_position[DEF]) < d or len(by_position[MID]) < m or len(by_position[FWD]) < f:
            continue
        starters = (
            [by_position[GK][0]]
            + by_position[DEF][:d]
            + by_position[MID][:m]
            + by_position[FWD][:f]
        )
        score = sum(expected_points.get(p.player_id, 0.0) for p in starters)
        if score > best_score:
            best_score = score
            best_combo = tuple(p.player_id for p in starters)

    if best_combo is None:
        raise ValueError("no valid formation fits this squad's position counts")

    starting_ids = set(best_combo)
    reserve_gk = [p for p in by_position[GK] if p.player_id not in starting_ids]
    outfield_bench = [
        p
        for position in (DEF, MID, FWD)
        for p in by_position[position]
        if p.player_id not in starting_ids
    ]
    outfield_bench.sort(key=lambda p: expected_points.get(p.player_id, 0.0), reverse=True)
    bench_order = tuple(p.player_id for p in outfield_bench) + tuple(
        p.player_id for p in reserve_gk
    )
    return best_combo, bench_order


def apply_autosubs(
    starting_xi: tuple[int, ...],
    bench_order: tuple[int, ...],
    squad: Sequence[SquadPlayer],
    minutes_played: Mapping[int, int],
) -> tuple[int, ...]:
    """FPL's real autosub rule: a bench player in ``bench_order`` comes on for a starter who
    played 0 minutes, in bench order, skipped if it would leave the resulting XI short of the
    legal 3-DEF/1-FWD floor. The reserve goalkeeper only ever considers replacing the starting
    goalkeeper (bench order already places it last, so by the time it's reached the only
    zero-minute "starter" left standing is normally the keeper; the position check below guards
    that explicitly rather than relying on ordering alone).
    """
    position_by_id = {p.player_id: p.position for p in squad}
    final_xi = list(starting_xi)

    def count(position: str, ids: Sequence[int]) -> int:
        return sum(1 for pid in ids if position_by_id[pid] == position)

    for bench_id in bench_order:
        zero_minute_starters = [pid for pid in final_xi if minutes_played.get(pid, 0) == 0]
        if not zero_minute_starters:
            break
        bench_position = position_by_id[bench_id]
        candidate_out = next(
            (
                pid
                for pid in zero_minute_starters
                if (position_by_id[pid] == GK) == (bench_position == GK)
            ),
            None,
        )
        if candidate_out is None:
            continue
        trial_xi = [bench_id if pid == candidate_out else pid for pid in final_xi]
        if count(DEF, trial_xi) < 3 or count(FWD, trial_xi) < 1:
            continue
        final_xi = trial_xi

    return tuple(final_xi)
