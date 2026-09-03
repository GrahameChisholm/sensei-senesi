"""Best-possible-squad solver: given a pool of candidate players and whichever players are
already picked, find the legal 15-man squad that maximizes projected points, via an integer
linear program (PuLP + the bundled CBC solver).

Filling every slot from empty and filling only a few just-vacated slots are the same problem: the
already-picked players are simply constrained to stay in the result (``locked_player_ids``), so a
full rebuild is just the trivial case of zero locked players.

``objective="full_squad"`` maximizes all 15 players' points, so the bench holds real squad depth
(autosubs, price rises, future gameweeks) rather than minimum-cost fodder — the default for
building or rebuilding a squad, and also correct when Bench Boost is active and bench points
count directly. ``objective="starting_xi"`` maximizes only the starting XI's points, spending the
entire remaining budget there and leaving the bench however cheap-but-legal falls out; only ever
right when the bench is truly worth nothing, which real usage never actually is. Either way, the
starting XI itself always comes out of :func:`~features.formation.select_starting_xi`'s own
highest-EV-of-the-15 pick afterward, never baked into the objective.

Captain/vice are chosen post-hoc (the two highest-EV starting-XI players), not baked into the ILP
objective: doubling one player's score is a nonlinear max-selection that would need extra binaries
for a gain that's always just the gap between the top two scorers, never enough to change which 15
players are optimal.

``current_squad_ids``/``max_transfers`` turn the same program into the transfer problem
(TRANSFER_BANNER): "keep at least ``15 - max_transfers`` of the players I already own." A full
rebuild is then just ``max_transfers = 15``, and a one-transfer suggestion is the same solve with
a tighter bound, so ``features.transfer_planner`` needs no second solver of its own. This is a
weaker constraint than ``locked_player_ids``, which names *which* players must stay; here the
solver chooses which ones leave, and that choice is the whole answer.

``excluded_squads`` carries the no-good cuts that let a caller enumerate the top N distinct
squads rather than just the optimum: re-solve with each already-found 15 forbidden, and CBC
returns the next best. Cheaper and far more predictable than asking CBC for a solution pool, and
it keeps every returned squad genuinely distinct rather than differing only in solver bookkeeping.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pulp

from features.formation import VALID_FORMATIONS, select_starting_xi
from features.squad_rules import (
    INITIAL_BUDGET,
    MAX_PER_CLUB,
    POSITION_QUOTA,
    SQUAD_SIZE,
    RuleViolation,
    SquadRuleError,
)
from features.team_state import SquadPlayer

__all__ = [
    "PlayerCandidate",
    "OptimizedSquad",
    "SquadOptimizerError",
    "optimise_squad",
]

_OBJECTIVES = ("starting_xi", "full_squad")
# Small enough (price differences are tenths of a million, at most a few hundred) to never
# override a real points difference, but large enough to reliably survive CBC's default relative
# MIP gap tolerance -- 1e-6 turned out too small in practice: CBC would stop at a solution tied on
# points but not tie-break-optimal, since the tie-break's effect fell within its optimality gap.
_BENCH_TIE_BREAK_WEIGHT = 1e-3


@dataclass(frozen=True)
class PlayerCandidate:
    player_id: int
    position: str
    team_id: int
    price: int
    expected_points: float


@dataclass(frozen=True)
class OptimizedSquad:
    squad: tuple[SquadPlayer, ...]
    starting_xi: tuple[int, ...]
    bench_order: tuple[int, ...]
    captain_id: int
    vice_captain_id: int


class SquadOptimizerError(ValueError):
    """Raised when no legal squad can be found — the remaining budget/quota/club-limit is too
    tight for any combination of unlocked candidates to fill the remaining slots."""


def optimise_squad(
    candidates: Sequence[PlayerCandidate],
    locked_player_ids: frozenset[int] = frozenset(),
    objective: str = "starting_xi",
    captain_multiplier: float = 2.0,
    budget: int = INITIAL_BUDGET,
    squad_size: int = SQUAD_SIZE,
    position_quota: Mapping[str, int] = POSITION_QUOTA,
    max_per_club: int = MAX_PER_CLUB,
    valid_formations: Sequence[tuple[int, int, int]] = VALID_FORMATIONS,
    current_squad_ids: frozenset[int] | None = None,
    max_transfers: int | None = None,
    excluded_squads: Sequence[frozenset[int]] = (),
) -> OptimizedSquad:
    """Find the legal squad of ``squad_size`` maximizing projected points, keeping every player in
    ``locked_player_ids`` in the result. ``captain_multiplier`` is accepted for signature symmetry
    with the points-preview call but never changes which players are picked (see module
    docstring).

    ``max_transfers``, given together with ``current_squad_ids``, caps how many of those players
    the result may drop. A current squad player with no candidate row at all (no projection this
    gameweek, so nothing to value him at) counts as already gone and spends one of those
    transfers, rather than raising: a squad can legitimately contain a player the projection cache
    has nothing for, and refusing to suggest anything at all in that case would be the worse
    answer.

    ``excluded_squads`` forbids each named set of exactly the players it contains from being the
    result again, so repeated calls walk down the ranking instead of returning the same optimum.
    """
    if objective not in _OBJECTIVES:
        raise ValueError(f"objective must be one of {_OBJECTIVES}, got {objective!r}")
    if max_transfers is not None:
        if current_squad_ids is None:
            raise ValueError("max_transfers requires current_squad_ids")
        if max_transfers < 0:
            raise ValueError(f"max_transfers must be non-negative, got {max_transfers}")

    by_id = {c.player_id: c for c in candidates}
    if len(by_id) != len(candidates):
        raise ValueError("candidates contains duplicate player_ids")
    missing_locked = locked_player_ids - set(by_id)
    if missing_locked:
        raise ValueError(f"locked_player_ids not found in candidates: {sorted(missing_locked)}")

    _preflight_locked_selection(by_id, locked_player_ids, position_quota, max_per_club, budget)

    problem = pulp.LpProblem("squad_optimizer", pulp.LpMaximize)

    squad_vars = {
        c.player_id: pulp.LpVariable(f"squad_{c.player_id}", cat="Binary") for c in candidates
    }
    xi_vars = {c.player_id: pulp.LpVariable(f"xi_{c.player_id}", cat="Binary") for c in candidates}
    formation_vars = {
        (d, m, f): pulp.LpVariable(f"formation_{d}_{m}_{f}", cat="Binary")
        for d, m, f in valid_formations
    }

    for player_id in squad_vars:
        problem += xi_vars[player_id] <= squad_vars[player_id]

    for player_id in locked_player_ids:
        problem += squad_vars[player_id] == 1

    problem += pulp.lpSum(squad_vars.values()) == squad_size

    by_position: dict[str, list[int]] = {}
    for c in candidates:
        by_position.setdefault(c.position, []).append(c.player_id)
    for position, quota in position_quota.items():
        problem += pulp.lpSum(squad_vars[pid] for pid in by_position.get(position, [])) == quota

    by_club: dict[int, list[int]] = {}
    for c in candidates:
        by_club.setdefault(c.team_id, []).append(c.player_id)
    for player_ids in by_club.values():
        problem += pulp.lpSum(squad_vars[pid] for pid in player_ids) <= max_per_club

    problem += pulp.lpSum(by_id[pid].price * squad_vars[pid] for pid in squad_vars) <= budget

    if max_transfers is not None and current_squad_ids is not None:
        # Only the current players actually present as candidates can be *kept*; any that are
        # missing have already left, so they come out of the same transfer allowance.
        keepable = [pid for pid in current_squad_ids if pid in squad_vars]
        minimum_kept = len(current_squad_ids) - max_transfers
        if minimum_kept > len(keepable):
            raise SquadOptimizerError(
                f"{len(current_squad_ids) - len(keepable)} current squad player(s) have no "
                f"candidate row, which already costs more than the {max_transfers} transfer(s) "
                "allowed"
            )
        problem += pulp.lpSum(squad_vars[pid] for pid in keepable) >= minimum_kept

    for excluded in excluded_squads:
        present = [pid for pid in excluded if pid in squad_vars]
        if present:
            problem += pulp.lpSum(squad_vars[pid] for pid in present) <= len(present) - 1

    problem += pulp.lpSum(formation_vars.values()) == 1
    problem += pulp.lpSum(xi_vars.values()) == 11
    gk_ids = by_position.get("GK", [])
    problem += pulp.lpSum(xi_vars[pid] for pid in gk_ids) == 1
    for position, index in (("DEF", 0), ("MID", 1), ("FWD", 2)):
        position_ids = by_position.get(position, [])
        problem += pulp.lpSum(xi_vars[pid] for pid in position_ids) == pulp.lpSum(
            formation[index] * var for formation, var in formation_vars.items()
        )

    points_objective = pulp.lpSum(
        by_id[pid].expected_points
        * (squad_vars[pid] if objective == "full_squad" else xi_vars[pid])
        for pid in squad_vars
    )
    bench_tie_break = pulp.lpSum(
        by_id[pid].price * (squad_vars[pid] - xi_vars[pid]) for pid in squad_vars
    )
    problem += points_objective - _BENCH_TIE_BREAK_WEIGHT * bench_tie_break

    problem.solve(pulp.PULP_CBC_CMD(msg=False))
    if problem.status != pulp.LpStatusOptimal:
        raise SquadOptimizerError(
            "no legal combination of remaining players fits the budget/quota given "
            f"{len(locked_player_ids)} locked player(s)"
        )

    chosen_ids = [pid for pid, var in squad_vars.items() if var.value() > 0.5]
    squad = tuple(SquadPlayer(pid, by_id[pid].position, by_id[pid].price) for pid in chosen_ids)
    expected_points = {pid: by_id[pid].expected_points for pid in chosen_ids}
    starting_xi, bench_order = select_starting_xi(squad, expected_points)

    ranked = sorted(starting_xi, key=lambda pid: expected_points.get(pid, 0.0), reverse=True)
    captain_id, vice_captain_id = ranked[0], ranked[1]

    return OptimizedSquad(
        squad=squad,
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
    )


def _preflight_locked_selection(
    by_id: Mapping[int, PlayerCandidate],
    locked_player_ids: frozenset[int],
    position_quota: Mapping[str, int],
    max_per_club: int,
    budget: int,
) -> None:
    """Raise :class:`~features.squad_rules.SquadRuleError` if the locked players alone already
    violate quota/club-limit/budget, before ever building the ILP — this is the "the caller's
    current squad is already illegal" case, distinct from :class:`SquadOptimizerError`'s "no
    remaining combination fits.\" """
    locked_squad = [by_id[pid] for pid in locked_player_ids]

    for position, count in _position_counts(locked_squad).items():
        if count > position_quota.get(position, 0):
            raise SquadRuleError(
                RuleViolation(
                    "quota",
                    f"locked selection already has {count} {position} players, "
                    f"more than the {position_quota.get(position, 0)} allowed",
                )
            )

    club_counts: dict[int, list[int]] = {}
    for candidate in locked_squad:
        club_counts.setdefault(candidate.team_id, []).append(candidate.player_id)
    for team_id, player_ids in club_counts.items():
        if len(player_ids) > max_per_club:
            raise SquadRuleError(
                RuleViolation(
                    "club_limit",
                    f"at most {max_per_club} players allowed from one club, "
                    f"team {team_id} has {len(player_ids)}",
                    player_ids=tuple(player_ids),
                )
            )

    total_spend = sum(candidate.price for candidate in locked_squad)
    if total_spend > budget:
        raise SquadRuleError(
            RuleViolation(
                "budget",
                (
                    f"locked selection costs {total_spend / 10:.1f}m, "
                    f"over the {budget / 10:.1f}m budget"
                ),
            )
        )


def _position_counts(squad: Sequence[PlayerCandidate]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for player in squad:
        counts[player.position] = counts.get(player.position, 0) + 1
    return counts
