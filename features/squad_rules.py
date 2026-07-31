"""Full FPL squad legality and the mutations that touch it (team-page plan D2/G3): the 2/5/5/3
position quota, the max-3-per-club limit, a position-legal starting XI, and the substitution/
transfer/captaincy/bench-order/optimise-XI operations a manager actually performs — none of which
existed anywhere in the repo before this page (``features.team_state.MyTeamState`` only ever
validated squad/XI *counts*, never position legality or club limits, and no mutation of any kind
was defined on it).

**Deliberately no standalone ``rebuild()``.** Real Wildcard/Free Hit do not force a full 15-player
replacement — either chip just means "transfers this gameweek cost no hit," which the caller can
achieve by calling :func:`transfer` any number of times (from 1 to 15) inside one draft. That
hit-suspension decision, and which chip (if any) is being played, is a *draft/confirm* concern —
see ``features.squad_draft`` — not something this module's own mutations need to know about;
:func:`transfer` always performs a legal swap, full stop, and :func:`transfer_hit_cost` is the one
place hit-free-ness is applied, as pure accounting with no squad mutation of its own.

Every mutation here is pure and total: it returns a new, frozen :class:`~features.team_state
.MyTeamState` or raises :class:`SquadRuleError` (never mutates its arguments), matching every other
``features/`` module's own discipline.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from engine.scoring import DEF, FWD, GK, MID, POSITIONS
from features.team_state import MyTeamState, SquadPlayer
from features.transfers import TRANSFER_HIT_COST
from simulator.formation import VALID_FORMATIONS, select_starting_xi

__all__ = [
    "SQUAD_SIZE",
    "POSITION_QUOTA",
    "MAX_PER_CLUB",
    "INITIAL_BUDGET",
    "RuleViolation",
    "SquadRuleError",
    "validate_squad",
    "validate_xi",
    "substitute",
    "transfer",
    "set_captain",
    "set_vice_captain",
    "reorder_bench",
    "optimise_xi",
    "transfer_hit_cost",
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


def validate_squad(
    squad: Sequence[SquadPlayer], team_id_by_player: Mapping[int, int]
) -> tuple[RuleViolation, ...]:
    """Every full-squad legality rule: exactly :data:`POSITION_QUOTA` per position, at most
    :data:`MAX_PER_CLUB` from any one club, and total spend (the sum of each player's own
    ``purchase_price`` — the price actually paid, not FPL's own profit-halved sell price) within
    :data:`INITIAL_BUDGET`. Returns every violation found, not just the first, so a caller building
    a squad from scratch can show every problem at once rather than one at a time.
    """
    violations: list[RuleViolation] = []

    counts: dict[str, int] = dict.fromkeys(POSITIONS, 0)
    for player in squad:
        counts[player.position] += 1
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

    club_counts: dict[int, list[int]] = {}
    for player in squad:
        team_id = team_id_by_player.get(player.player_id)
        if team_id is None:
            continue
        club_counts.setdefault(team_id, []).append(player.player_id)
    for team_id, player_ids in club_counts.items():
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

    total_spend = sum(player.purchase_price for player in squad)
    if total_spend > INITIAL_BUDGET:
        violations.append(
            RuleViolation(
                code="budget",
                message=(
                    f"squad costs {total_spend / 10:.1f}m, over the "
                    f"{INITIAL_BUDGET / 10:.1f}m budget"
                ),
            )
        )

    return tuple(violations)


def validate_xi(
    starting_xi: Sequence[int], position_by_player: Mapping[int, str]
) -> tuple[RuleViolation, ...]:
    """Position-legal starting XI: exactly 11 players, exactly 1 GK, and the outfield (DEF, MID,
    FWD) split one of :data:`~simulator.formation.VALID_FORMATIONS` — checked as an exact
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


def substitute(
    state: MyTeamState, out_id: int, in_id: int, position_by_player: Mapping[int, str]
) -> MyTeamState:
    """Swap one starting-XI player for one bench player — never hit-costed, at any time, matching
    real FPL (only bringing in a *different* player can ever cost a hit; rearranging players you
    already own never does). A goalkeeper can only be swapped with another goalkeeper — this falls
    out of :func:`validate_xi`'s own "exactly 1 GK" rule automatically rather than needing a
    special case: swapping the only starting GK for an outfielder would leave the resulting XI
    with zero GKs, which :func:`validate_xi` already rejects.
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
    in_id: int,
    in_price: int,
    in_position: str,
    team_id_by_player: Mapping[int, int],
) -> MyTeamState:
    """Sell ``out_id`` at their real FPL sell price (:func:`~features.team_state.compute_sell_price`
    — half of any profit, rounded down), buy ``in_id`` at ``in_price``, and re-validate the result.
    Whichever slot ``out_id`` occupied (starting XI or bench, captain or vice) is inherited by
    ``in_id`` — a squad invariant (``MyTeamState`` requires the captain/vice to be in the starting
    XI) must never be left dangling by a transfer.

    Never charges a hit itself — see :func:`transfer_hit_cost` and this module's own docstring for
    why that is deliberately a separate, draft-level accounting concern.
    """
    if out_id not in state.player_ids:
        raise SquadRuleError(
            RuleViolation("unknown_player", f"player {out_id} is not in the squad", (out_id,))
        )
    if in_id in state.player_ids:
        raise SquadRuleError(
            RuleViolation("duplicate", f"player {in_id} is already in the squad", (in_id,))
        )

    out_player = state.player(out_id)
    sell_price = out_player.sell_price
    new_bank = state.bank + sell_price - in_price
    if new_bank < 0:
        raise SquadRuleError(
            RuleViolation(
                "budget",
                f"transfer costs {(in_price - sell_price) / 10:.1f}m more than is in the bank",
            )
        )

    new_player = SquadPlayer(
        player_id=in_id, position=in_position, purchase_price=in_price, current_price=in_price
    )
    new_squad = tuple(new_player if p.player_id == out_id else p for p in state.squad)

    _first_or_raise(validate_squad(new_squad, team_id_by_player))

    new_xi = tuple(in_id if pid == out_id else pid for pid in state.starting_xi)
    new_bench = tuple(in_id if pid == out_id else pid for pid in state.bench_order)
    captain_id = in_id if state.captain_id == out_id else state.captain_id
    vice_captain_id = in_id if state.vice_captain_id == out_id else state.vice_captain_id

    return replace(
        state,
        squad=new_squad,
        starting_xi=new_xi,
        bench_order=new_bench,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        bank=new_bank,
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
    (:func:`~simulator.formation.select_starting_xi`) — applied immediately, with no draft/confirm
    step, since it can only ever rearrange players already owned (D22: no budget/quota/legality
    risk to weigh first).

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


def transfer_hit_cost(n_transfers: int, free_transfers: int, *, hit_free: bool) -> int:
    """Points deducted for ``n_transfers`` transfers, given ``free_transfers`` already banked.

    ``hit_free=True`` covers both real-FPL cases where a hit can never apply: before the very
    first (GW1) deadline, when the entire squad is still being freely assembled (D13), and any
    draft where a Wildcard or Free Hit chip is being played (D14) — the caller (``features
    .squad_draft.confirm_draft``) decides which case applies; this function only ever applies
    whichever flag it's given, and never touches squad legality.
    """
    if hit_free:
        return 0
    chargeable = max(0, n_transfers - free_transfers)
    return chargeable * TRANSFER_HIT_COST
