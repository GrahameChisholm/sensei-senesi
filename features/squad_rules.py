"""Full FPL squad legality and the mutations that touch it: the 2/5/5/3 position quota, the
max-3-per-club limit, a position-legal starting XI, and the add/remove/captaincy/bench-order/
optimise-XI operations a manager actually performs.

The squad is a permanent sandbox: nothing is ever locked in, every mutation applies immediately,
and there is no free-transfer count, hit cost, or purchase-price/sell-price distinction — a
player's price is always just their current price. ``validate_squad``/``validate_partial_squad``
check spend against a ``budget`` ceiling that is ordinarily the classic £100m
(:data:`INITIAL_BUDGET`), but can be a caller-supplied higher figure for a squad whose real value
has grown past that through price rises (see ``build_team_state``'s ``check_budget=False`` path,
used when importing a real FPL manager's squad).

Every mutation here is pure and total: it returns a new, frozen :class:`~features.team_state
.MyTeamState` (or a bare squad tuple for the partial-squad functions) or raises
:class:`SquadRuleError` (never mutates its arguments), matching every other ``features/`` module's
own discipline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from engine.scoring import DEF, FWD, GK, MID, POSITIONS
from features.formation import VALID_FORMATIONS, select_starting_xi
from features.team_state import MyTeamState, SquadPlayer

__all__ = [
    "SQUAD_SIZE",
    "POSITION_QUOTA",
    "MAX_PER_CLUB",
    "INITIAL_BUDGET",
    "RuleViolation",
    "SquadRuleError",
    "validate_squad",
    "validate_partial_squad",
    "validate_xi",
    "add_player",
    "remove_player",
    "build_team_state",
    "assemble_team_state",
    "substitute",
    "transfer",
    "set_captain",
    "set_vice_captain",
    "reorder_bench",
    "optimise_xi",
]

SQUAD_SIZE = 15
POSITION_QUOTA: Mapping[str, int] = {GK: 2, DEF: 5, MID: 5, FWD: 3}
MAX_PER_CLUB = 3
# Tenths of a million, matching features.team_state.SquadPlayer's own convention (£100.0m).
INITIAL_BUDGET = 1000


@dataclass(frozen=True)
class RuleViolation:
    """One broken rule, in a shape the API/UI can render directly without re-deriving the human
    message from a raw error code."""

    code: str
    message: str
    player_ids: tuple[int, ...] = ()


class SquadRuleError(ValueError):
    """Raised by every mutation in this module on an illegal result; carries the
    :class:`RuleViolation` that explains why, so a caller (the API layer) never has to re-parse a
    string message to decide what happened."""

    def __init__(self, violation: RuleViolation) -> None:
        super().__init__(violation.message)
        self.violation = violation


def _first_or_raise(violations: tuple[RuleViolation, ...]) -> None:
    if violations:
        raise SquadRuleError(violations[0])


def _position_counts(squad: Sequence[SquadPlayer]) -> dict[str, int]:
    counts: dict[str, int] = dict.fromkeys(POSITIONS, 0)
    for player in squad:
        counts[player.position] += 1
    return counts


def _club_counts(
    squad: Sequence[SquadPlayer], team_id_by_player: Mapping[int, int]
) -> dict[int, list[int]]:
    club_counts: dict[int, list[int]] = {}
    for player in squad:
        team_id = team_id_by_player.get(player.player_id)
        if team_id is None:
            continue
        club_counts.setdefault(team_id, []).append(player.player_id)
    return club_counts


def validate_squad(
    squad: Sequence[SquadPlayer],
    team_id_by_player: Mapping[int, int],
    *,
    budget: int = INITIAL_BUDGET,
    check_budget: bool = True,
) -> tuple[RuleViolation, ...]:
    """Every full-squad (exactly 15) legality rule: exactly :data:`POSITION_QUOTA` per position, at
    most :data:`MAX_PER_CLUB` from any one club, and, if ``check_budget``, total spend within
    ``budget``. Returns every violation found, not just the first, so a caller building a squad
    from scratch can show every problem at once rather than one at a time.

    ``check_budget=False`` is for a squad imported from a real manager's own FPL team, whose
    current value can legitimately exceed the classic £100m ceiling through price-rise profit —
    see :func:`build_team_state`.
    """
    violations: list[RuleViolation] = []

    counts = _position_counts(squad)
    for position, required in POSITION_QUOTA.items():
        if counts[position] != required:
            violations.append(
                RuleViolation(
                    code="quota",
                    message=(
                        f"squad must have exactly {required} {position} players, "
                        f"has {counts[position]}"
                    ),
                )
            )

    for team_id, player_ids in _club_counts(squad, team_id_by_player).items():
        if len(player_ids) > MAX_PER_CLUB:
            violations.append(
                RuleViolation(
                    code="club_limit",
                    message=(
                        f"at most {MAX_PER_CLUB} players allowed from one club, "
                        f"team {team_id} has {len(player_ids)}"
                    ),
                    player_ids=tuple(player_ids),
                )
            )

    if check_budget:
        total_spend = sum(player.price for player in squad)
        if total_spend > budget:
            violations.append(
                RuleViolation(
                    code="budget",
                    message=(
                        f"squad costs {total_spend / 10:.1f}m, over the {budget / 10:.1f}m budget"
                    ),
                )
            )

    return tuple(violations)


def validate_partial_squad(
    squad: Sequence[SquadPlayer],
    team_id_by_player: Mapping[int, int],
    budget: int = INITIAL_BUDGET,
) -> tuple[RuleViolation, ...]:
    """The same three rule families as :func:`validate_squad`, as ceilings rather than exact
    equality, so a squad of any size from 0 to 15 can be checked mid-build: per-position count
    ``<= POSITION_QUOTA[position]``, per-club ``<= MAX_PER_CLUB``, total spend ``<= budget``."""
    violations: list[RuleViolation] = []

    counts = _position_counts(squad)
    for position, quota in POSITION_QUOTA.items():
        if counts[position] > quota:
            violations.append(
                RuleViolation(
                    code="quota",
                    message=(
                        f"squad can have at most {quota} {position} players, "
                        f"has {counts[position]}"
                    ),
                )
            )

    for team_id, player_ids in _club_counts(squad, team_id_by_player).items():
        if len(player_ids) > MAX_PER_CLUB:
            violations.append(
                RuleViolation(
                    code="club_limit",
                    message=(
                        f"at most {MAX_PER_CLUB} players allowed from one club, "
                        f"team {team_id} has {len(player_ids)}"
                    ),
                    player_ids=tuple(player_ids),
                )
            )

    total_spend = sum(player.price for player in squad)
    if total_spend > budget:
        violations.append(
            RuleViolation(
                code="budget",
                message=f"squad costs {total_spend / 10:.1f}m, over the {budget / 10:.1f}m budget",
            )
        )

    return tuple(violations)


def validate_xi(
    starting_xi: Sequence[int], position_by_player: Mapping[int, str]
) -> tuple[RuleViolation, ...]:
    """Position-legal starting XI: exactly 11 players, exactly 1 GK, and the outfield (DEF, MID,
    FWD) split one of :data:`~features.formation.VALID_FORMATIONS` — checked as an exact
    membership test, not independent per-position ranges, since not every combination of
    individually-legal counts sums to 10 outfield players."""
    if len(starting_xi) != 11:
        return (
            RuleViolation(
                code="xi_shape",
                message=f"starting XI must have exactly 11 players, has {len(starting_xi)}",
            ),
        )

    counts: dict[str, int] = dict.fromkeys(POSITIONS, 0)
    for player_id in starting_xi:
        counts[position_by_player[player_id]] += 1

    if counts[GK] != 1:
        return (
            RuleViolation(
                code="xi_shape",
                message=f"starting XI must have exactly 1 goalkeeper, has {counts[GK]}",
            ),
        )

    formation = (counts[DEF], counts[MID], counts[FWD])
    if formation not in VALID_FORMATIONS:
        d, m, f = formation
        return (
            RuleViolation(
                code="xi_shape",
                message=(
                    f"{d} defender(s)/{m} midfielder(s)/{f} forward(s) is not a legal formation "
                    "(need 3-5 DEF, 2-5 MID, 1-3 FWD summing to 10 outfield players)"
                ),
            ),
        )
    return ()


def add_player(
    squad: Sequence[SquadPlayer],
    player: SquadPlayer,
    team_id_by_player: Mapping[int, int],
    budget: int = INITIAL_BUDGET,
) -> tuple[SquadPlayer, ...]:
    """Add one player to a 0-to-14-player sandbox squad. Raises :class:`SquadRuleError` if
    ``player`` is already in the squad, the squad is already full, or the result would violate
    :func:`validate_partial_squad`."""
    if any(p.player_id == player.player_id for p in squad):
        raise SquadRuleError(
            RuleViolation(
                "duplicate",
                f"player {player.player_id} is already in the squad",
                (player.player_id,),
            )
        )
    if len(squad) >= SQUAD_SIZE:
        raise SquadRuleError(RuleViolation("squad_full", f"squad already has {SQUAD_SIZE} players"))

    new_squad = (*squad, player)
    _first_or_raise(validate_partial_squad(new_squad, team_id_by_player, budget))
    return new_squad


def remove_player(squad: Sequence[SquadPlayer], player_id: int) -> tuple[SquadPlayer, ...]:
    """Remove one player from the sandbox squad. Raises :class:`SquadRuleError` if ``player_id``
    isn't in it."""
    if not any(p.player_id == player_id for p in squad):
        raise SquadRuleError(
            RuleViolation("unknown_player", f"player {player_id} is not in the squad", (player_id,))
        )
    return tuple(p for p in squad if p.player_id != player_id)


def build_team_state(
    squad: Sequence[SquadPlayer],
    starting_xi: Sequence[int],
    bench_order: Sequence[int],
    captain_id: int,
    vice_captain_id: int,
    team_id_by_player: Mapping[int, int],
    budget: int = INITIAL_BUDGET,
    check_budget: bool = True,
    mini_league_ids: tuple[int, ...] = (),
) -> MyTeamState:
    """Validate a caller-fully-specified squad/XI/captain/vice (exact legality, XI shape, and
    bench-shape) and promote it to :class:`~features.team_state.MyTeamState` verbatim, with no
    auto-derivation of XI/captain. Used when importing a real FPL manager's squad, which should
    keep their actual picks exactly as FPL reports them rather than re-optimizing.

    ``check_budget=False`` is for that import path specifically: a real squad's current value can
    legitimately exceed ``budget`` through price-rise profit.
    """
    position_by_player = {player.player_id: player.position for player in squad}
    _first_or_raise(
        validate_squad(squad, team_id_by_player, budget=budget, check_budget=check_budget)
    )
    _first_or_raise(validate_xi(starting_xi, position_by_player))

    squad_ids = {player.player_id for player in squad}
    bench_ids = squad_ids - set(starting_xi)
    if set(bench_order) != bench_ids:
        raise SquadRuleError(
            RuleViolation("xi_shape", "bench_order must be exactly the squad minus the starting XI")
        )

    return MyTeamState(
        squad=tuple(squad),
        starting_xi=tuple(starting_xi),
        bench_order=tuple(bench_order),
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        mini_league_ids=mini_league_ids,
    )


def assemble_team_state(
    squad: Sequence[SquadPlayer],
    expected_points: Mapping[int, float],
    team_id_by_player: Mapping[int, int],
    budget: int = INITIAL_BUDGET,
    mini_league_ids: tuple[int, ...] = (),
    preferred_captain_id: int | None = None,
    preferred_vice_captain_id: int | None = None,
) -> MyTeamState:
    """Validate a complete, legal 15 (:func:`validate_squad`, budget checked) and auto-derive the
    best XI/bench (:func:`~features.formation.select_starting_xi`). Captain/vice default to the
    top-2 XI scorers, unless ``preferred_captain_id``/``preferred_vice_captain_id`` are given and
    still eligible (in the new XI) — used when a squad that already had a captain/vice loses and
    regains a player, so an incidental swap elsewhere doesn't casually change who's captained.
    Used when a manually-built squad reaches 15 players and needs a starting lineup, and by the
    optimizer's own promotion of its result."""
    _first_or_raise(validate_squad(squad, team_id_by_player, budget=budget))

    starting_xi, bench_order = select_starting_xi(squad, expected_points)
    ranked = sorted(starting_xi, key=lambda pid: expected_points.get(pid, 0.0), reverse=True)
    captain_id = preferred_captain_id if preferred_captain_id in starting_xi else ranked[0]
    vice_captain_id = (
        preferred_vice_captain_id
        if preferred_vice_captain_id in starting_xi and preferred_vice_captain_id != captain_id
        else next(pid for pid in ranked if pid != captain_id)
    )

    return MyTeamState(
        squad=tuple(squad),
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        mini_league_ids=mini_league_ids,
    )


def substitute(
    state: MyTeamState, out_id: int, in_id: int, position_by_player: Mapping[int, str]
) -> MyTeamState:
    """Swap one starting-XI player for one bench player. A goalkeeper can only be swapped with
    another goalkeeper — this falls out of :func:`validate_xi`'s own "exactly 1 GK" rule
    automatically rather than needing a special case: swapping the only starting GK for an
    outfielder would leave the resulting XI with zero GKs, which :func:`validate_xi` already
    rejects.
    """
    if out_id not in state.starting_xi:
        raise SquadRuleError(
            RuleViolation("unknown_player", f"player {out_id} is not in the starting XI", (out_id,))
        )
    if in_id not in state.bench_order:
        raise SquadRuleError(
            RuleViolation("unknown_player", f"player {in_id} is not on the bench", (in_id,))
        )

    new_xi = tuple(in_id if pid == out_id else pid for pid in state.starting_xi)
    new_bench = tuple(out_id if pid == in_id else pid for pid in state.bench_order)

    _first_or_raise(validate_xi(new_xi, position_by_player))

    captain_id = in_id if state.captain_id == out_id else state.captain_id
    vice_captain_id = in_id if state.vice_captain_id == out_id else state.vice_captain_id

    return replace(
        state,
        starting_xi=new_xi,
        bench_order=new_bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
    )


def transfer(
    state: MyTeamState,
    out_id: int,
    in_player: SquadPlayer,
    team_id_by_player: Mapping[int, int],
    budget: int = INITIAL_BUDGET,
) -> MyTeamState:
    """Replace ``out_id`` with ``in_player`` and re-validate the result. Whichever slot ``out_id``
    occupied (starting XI or bench, captain or vice) is inherited by ``in_player`` — a squad
    invariant (``MyTeamState`` requires the captain/vice to be in the starting XI) must never be
    left dangling by a transfer. Always free — there is no hit cost or free-transfer count in this
    sandbox.
    """
    if out_id not in state.player_ids:
        raise SquadRuleError(
            RuleViolation("unknown_player", f"player {out_id} is not in the squad", (out_id,))
        )
    if in_player.player_id in state.player_ids:
        raise SquadRuleError(
            RuleViolation(
                "duplicate",
                f"player {in_player.player_id} is already in the squad",
                (in_player.player_id,),
            )
        )

    new_squad = tuple(in_player if p.player_id == out_id else p for p in state.squad)
    _first_or_raise(validate_squad(new_squad, team_id_by_player, budget=budget))

    new_xi = tuple(in_player.player_id if pid == out_id else pid for pid in state.starting_xi)
    new_bench = tuple(in_player.player_id if pid == out_id else pid for pid in state.bench_order)
    captain_id = in_player.player_id if state.captain_id == out_id else state.captain_id
    vice_captain_id = (
        in_player.player_id if state.vice_captain_id == out_id else state.vice_captain_id
    )

    return replace(
        state,
        squad=new_squad,
        starting_xi=new_xi,
        bench_order=new_bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
    )


def set_captain(state: MyTeamState, player_id: int) -> MyTeamState:
    """``player_id`` must already be in the starting XI, and must differ from the current vice
    (``MyTeamState``'s own invariant) — raised here as a typed violation rather than left to
    surface as a raw ``MyTeamState`` construction error."""
    if player_id not in state.starting_xi:
        raise SquadRuleError(
            RuleViolation(
                "unknown_player", f"player {player_id} is not in the starting XI", (player_id,)
            )
        )
    if player_id == state.vice_captain_id:
        raise SquadRuleError(
            RuleViolation(
                "duplicate", "captain and vice-captain must be different players", (player_id,)
            )
        )
    return replace(state, captain_id=player_id)


def set_vice_captain(state: MyTeamState, player_id: int) -> MyTeamState:
    if player_id not in state.starting_xi:
        raise SquadRuleError(
            RuleViolation(
                "unknown_player", f"player {player_id} is not in the starting XI", (player_id,)
            )
        )
    if player_id == state.captain_id:
        raise SquadRuleError(
            RuleViolation(
                "duplicate", "captain and vice-captain must be different players", (player_id,)
            )
        )
    return replace(state, vice_captain_id=player_id)


def reorder_bench(state: MyTeamState, bench_order: Sequence[int]) -> MyTeamState:
    """Re-order the bench (e.g. so autosubs prefer a different reserve first) — must be a
    permutation of the existing bench, never a way to add/remove a player (that's :func:`transfer`
    's job)."""
    if set(bench_order) != set(state.bench_order) or len(bench_order) != len(state.bench_order):
        raise SquadRuleError(
            RuleViolation(
                "unknown_player", "bench_order must be a permutation of the existing bench"
            )
        )
    return replace(state, bench_order=tuple(bench_order))


def optimise_xi(state: MyTeamState, expected_points: Mapping[int, float]) -> MyTeamState:
    """Best legal formation from the existing 15, by expected points
    (:func:`~features.formation.select_starting_xi`) — applied immediately, since it can only ever
    rearrange players already owned.

    Preserves the existing captain/vice if they're still in the new XI (optimising a lineup
    shouldn't casually change who's captained just because the *arrangement* changed); otherwise
    picks the highest-EV player(s) in the new XI, using ``expected_points`` directly rather than
    the XI tuple's own GK-first/DEF/MID/FWD ordering (which is not itself EV-ordered).
    """
    new_xi, new_bench = select_starting_xi(state.squad, expected_points)

    ranked = sorted(new_xi, key=lambda pid: expected_points.get(pid, 0.0), reverse=True)
    captain_id = state.captain_id if state.captain_id in new_xi else ranked[0]
    vice_captain_id = (
        state.vice_captain_id
        if state.vice_captain_id in new_xi and state.vice_captain_id != captain_id
        else next(pid for pid in ranked if pid != captain_id)
    )

    return replace(
        state,
        starting_xi=new_xi,
        bench_order=new_bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
    )
