"""The draft/confirm state machine (D16-D24/G8) — the interaction model the whole team-selection
page's editing experience depends on: nothing a manager does (a substitution, a transfer, playing
a chip) is real until an explicit Confirm, and Free Hit specifically reverts the squad it changed
one gameweek later, matching real FPL exactly.

**Why this is a separate module from ``features.squad_rules``.** ``squad_rules`` answers "is this
squad legal, and what does this one mutation produce" — it has no concept of time, chips, or an
uncommitted draft. This module answers "given a real committed squad and a hypothetical draft,
what happens on Confirm, or on the next gameweek" — a genuinely different, sequencing concern.

**Two distinct lifecycles, not one.** Building the *first* squad from an empty £100m (D6/D23) has
no existing committed squad to diff against and can't be represented as a
:class:`~features.team_state.MyTeamState` mid-build (that type requires exactly 15/11/4 at every
step) — see :func:`confirm_initial_squad`. Every subsequent edit, by contrast, only ever swaps one
player at a time against an already-complete 15, so a :class:`PendingDraft` can simply wrap a real,
always-valid ``MyTeamState`` throughout (:func:`open_draft`/the ``apply_*_to_draft`` functions),
confirmed via :func:`confirm_draft`.

**Deliberately no standalone rebuild path.** Playing Wildcard or Free Hit is just a draft whose
``chip`` is set, inside which any number of ordinary :func:`~features.squad_rules.transfer` calls
(1 to 15) are hit-free — see ``features.squad_rules``'s own docstring for the reasoning.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from features.chip_calendar import (
    ChipUsage,
    available_chips_this_gameweek,
    chip_usage_from_fpl_history,
    record_chip_played,
)
from features.squad_rules import (
    INITIAL_BUDGET,
    RuleViolation,
    SquadRuleError,
    optimise_xi,
    reorder_bench,
    set_captain,
    set_vice_captain,
    substitute,
    transfer,
    transfer_hit_cost,
    validate_squad,
    validate_xi,
)
from features.team_state import CHIPS, MyTeamState, SquadPlayer

__all__ = [
    "PendingDraft",
    "CommittedSquad",
    "confirm_initial_squad",
    "confirm_imported_squad",
    "open_draft",
    "apply_substitute_to_draft",
    "apply_transfer_to_draft",
    "apply_set_captain_to_draft",
    "apply_set_vice_captain_to_draft",
    "apply_reorder_bench_to_draft",
    "apply_optimise_xi_to_draft",
    "set_draft_chip",
    "confirm_draft",
    "advance_gameweek",
]


@dataclass(frozen=True)
class PendingDraft:
    """An uncommitted edit against an *existing* committed squad — nothing here is real until
    :func:`confirm_draft` succeeds. Every mutation updates ``working_state`` directly via the
    ``apply_*_to_draft`` functions; this wrapper additionally tracks how many genuine transfers
    happened (:func:`~features.squad_rules.transfer_hit_cost` needs a count, not just a squad
    diff) and which chip, if any, this draft will play on confirm."""

    base_gameweek: int
    working_state: MyTeamState
    transfers_made: int = 0
    chip: str | None = None

    def __post_init__(self) -> None:
        if self.chip is not None and self.chip not in CHIPS:
            raise ValueError(f"unknown chip: {self.chip!r}")
        if self.transfers_made < 0:
            raise ValueError("transfers_made must be non-negative")


@dataclass(frozen=True)
class CommittedSquad:
    """The real, confirmed state — the source of truth for what's actually going in next
    gameweek. ``team_state`` is ``None`` only before the very first :func:`confirm_initial_squad`
    (D23). ``active_chip``/``active_chip_gameweek`` name whichever chip (if any) is in effect for
    ``active_chip_gameweek`` specifically — cleared by :func:`advance_gameweek` once that
    gameweek has passed. ``free_hit_snapshot`` holds the pre-chip squad while (and only while)
    Free Hit is the active chip, restored automatically by :func:`advance_gameweek` (D15).
    ``gameweek_hit_cost`` accumulates every hit :func:`confirm_draft` has charged for
    ``committed_gameweek`` specifically (a manager may confirm more than once before that
    gameweek is actually played) — Season Replay's ``POST /squad/advance`` is the one consumer
    that needs this total; :func:`advance_gameweek` resets it to 0 for the new gameweek."""

    team_state: MyTeamState | None
    chip_usage: ChipUsage = ChipUsage()
    active_chip: str | None = None
    active_chip_gameweek: int | None = None
    free_hit_snapshot: MyTeamState | None = None
    free_hit_snapshot_gameweek: int | None = None
    committed_gameweek: int = 0
    gameweek_hit_cost: int = 0


def _raise_first(violations: tuple[RuleViolation, ...]) -> None:
    if violations:
        raise SquadRuleError(violations[0])


def confirm_initial_squad(
    squad: tuple[SquadPlayer, ...],
    starting_xi: tuple[int, ...],
    bench_order: tuple[int, ...],
    captain_id: int,
    vice_captain_id: int,
    team_id_by_player: Mapping[int, int],
    gameweek: int,
) -> CommittedSquad:
    """The very first commit (D23) — going from an empty build draft to a real committed squad.
    No existing :class:`CommittedSquad` to diff against, so this validates legality directly
    (budget/quota/club-limit/XI-shape) rather than going through :func:`confirm_draft`. Never
    charges a hit — there is nothing to charge one relative to, and D13 makes pre-GW1-deadline
    squad-building free regardless.
    """
    position_by_player = {player.player_id: player.position for player in squad}
    _raise_first(validate_squad(squad, team_id_by_player))
    _raise_first(validate_xi(starting_xi, position_by_player))

    squad_ids = {player.player_id for player in squad}
    bench_ids = squad_ids - set(starting_xi)
    if set(bench_order) != bench_ids:
        raise SquadRuleError(
            RuleViolation("xi_shape", "bench_order must be exactly the squad minus the starting XI")
        )

    chip_usage = ChipUsage()
    team_state = MyTeamState(
        squad=squad,
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        bank=INITIAL_BUDGET - sum(player.purchase_price for player in squad),
        free_transfers=1,
        chips_remaining=available_chips_this_gameweek(chip_usage, gameweek),
    )
    return CommittedSquad(team_state=team_state, chip_usage=chip_usage, committed_gameweek=gameweek)


def confirm_imported_squad(
    team_state: MyTeamState,
    team_id_by_player: Mapping[int, int],
    chips_played: Sequence[Mapping[str, object]],
    gameweek: int,
) -> CommittedSquad:
    """Commit a squad imported from a real FPL manager's team ID (D6/section 18 of TEAM_PAGE_PLAN
    — deferred until GW1 locks; see :func:`~engine.data.team_state_builder.build_my_team_state`,
    the function that plan named for this job), now that it has.

    Unlike :func:`confirm_initial_squad`, this skips the from-scratch budget check
    (:data:`~features.squad_rules.INITIAL_BUDGET`) — ``team_state.bank`` already reflects this
    manager's real transfer history, not a fresh £100m, and a manager who has banked price-rise
    profit can legitimately hold a squad whose purchase-price sum exceeds it. Quota, club-limit,
    and XI-shape are still validated defensively, same as ``confirm_initial_squad``.

    ``chips_played`` is FPL's own raw ``entry/{id}/history/``'s ``chips`` list — reconstructed via
    :func:`~features.chip_calendar.chip_usage_from_fpl_history` so a chip this manager already
    played in real FPL can't be played again in this app.
    """
    position_by_player = {player.player_id: player.position for player in team_state.squad}
    _raise_first(validate_squad(team_state.squad, team_id_by_player, check_budget=False))
    _raise_first(validate_xi(team_state.starting_xi, position_by_player))

    squad_ids = {player.player_id for player in team_state.squad}
    bench_ids = squad_ids - set(team_state.starting_xi)
    if set(team_state.bench_order) != bench_ids:
        raise SquadRuleError(
            RuleViolation("xi_shape", "bench_order must be exactly the squad minus the starting XI")
        )

    chip_usage = chip_usage_from_fpl_history(chips_played)
    new_team_state = replace(
        team_state, chips_remaining=available_chips_this_gameweek(chip_usage, gameweek)
    )
    return CommittedSquad(
        team_state=new_team_state, chip_usage=chip_usage, committed_gameweek=gameweek
    )


def open_draft(committed: CommittedSquad, gameweek: int) -> PendingDraft:
    """Start editing (D16's "Edit team") — a fresh draft copying the current committed squad
    exactly, with no transfers made and no chip selected yet."""
    if committed.team_state is None:
        raise ValueError(
            "no committed squad exists yet — build and confirm the initial squad first "
            "(confirm_initial_squad)"
        )
    return PendingDraft(base_gameweek=gameweek, working_state=committed.team_state)


def apply_substitute_to_draft(
    draft: PendingDraft, out_id: int, in_id: int, position_by_player: Mapping[int, str]
) -> PendingDraft:
    return replace(
        draft, working_state=substitute(draft.working_state, out_id, in_id, position_by_player)
    )


def apply_transfer_to_draft(
    draft: PendingDraft,
    out_id: int,
    in_id: int,
    in_price: int,
    in_position: str,
    team_id_by_player: Mapping[int, int],
) -> PendingDraft:
    new_state = transfer(
        draft.working_state, out_id, in_id, in_price, in_position, team_id_by_player
    )
    return replace(draft, working_state=new_state, transfers_made=draft.transfers_made + 1)


def apply_set_captain_to_draft(draft: PendingDraft, player_id: int) -> PendingDraft:
    return replace(draft, working_state=set_captain(draft.working_state, player_id))


def apply_set_vice_captain_to_draft(draft: PendingDraft, player_id: int) -> PendingDraft:
    return replace(draft, working_state=set_vice_captain(draft.working_state, player_id))


def apply_reorder_bench_to_draft(draft: PendingDraft, bench_order: Sequence[int]) -> PendingDraft:
    return replace(draft, working_state=reorder_bench(draft.working_state, bench_order))


def apply_optimise_xi_to_draft(
    draft: PendingDraft, expected_points: Mapping[int, float]
) -> PendingDraft:
    return replace(draft, working_state=optimise_xi(draft.working_state, expected_points))


def set_draft_chip(draft: PendingDraft, chip: str | None) -> PendingDraft:
    """Set or clear which chip (if any) this draft will play on confirm (D18) — free to change
    right up until :func:`confirm_draft` actually runs; availability is only checked there, since
    that's the one moment it has to be authoritative."""
    if chip is not None and chip not in CHIPS:
        raise ValueError(f"unknown chip: {chip!r}")
    return replace(draft, chip=chip)


def confirm_draft(
    committed: CommittedSquad, draft: PendingDraft, gameweek: int, deadline_passed: bool
) -> tuple[CommittedSquad, int]:
    """Materialise ``draft`` into a new :class:`CommittedSquad`, returning
    ``(new_committed, hit_cost_charged)`` — the **only** place a hit is ever charged or a chip is
    ever spent (D16).

    Rejects (raises :class:`~features.squad_rules.SquadRuleError`) a draft opened for a different
    gameweek than ``gameweek`` (D24 — a stale draft is never force-applied) or a ``chip`` that
    isn't currently available. Transfers inside the draft are hit-free if ``not deadline_passed``
    (D13, before the very first deadline) or ``draft.chip`` is Wildcard/Free Hit (D14) — otherwise
    :func:`~features.squad_rules.transfer_hit_cost` applies normally. Playing Free Hit stashes the
    pre-chip squad as the new ``free_hit_snapshot`` (D15); playing any chip records it in
    ``chip_usage`` via :func:`~features.chip_calendar.record_chip_played`.
    """
    if draft.base_gameweek != gameweek:
        raise SquadRuleError(
            RuleViolation(
                "draft_stale",
                f"this draft was opened for gameweek {draft.base_gameweek}, not the current "
                f"gameweek {gameweek} — discard it and start again",
            )
        )

    if draft.chip is not None and draft.chip not in available_chips_this_gameweek(
        committed.chip_usage, gameweek
    ):
        raise SquadRuleError(
            RuleViolation("chip_unavailable", f"{draft.chip} is not available this gameweek")
        )

    assert committed.team_state is not None  # open_draft() already guarantees this

    hit_free = (not deadline_passed) or draft.chip in ("wildcard", "free_hit")
    hit_cost = transfer_hit_cost(
        draft.transfers_made, committed.team_state.free_transfers, hit_free=hit_free
    )

    new_chip_usage = committed.chip_usage
    free_hit_snapshot = None
    free_hit_snapshot_gameweek = None
    if draft.chip is not None:
        new_chip_usage = record_chip_played(committed.chip_usage, draft.chip, gameweek)
        if draft.chip == "free_hit":
            free_hit_snapshot = committed.team_state
            free_hit_snapshot_gameweek = gameweek

    new_free_transfers = (
        committed.team_state.free_transfers
        if hit_free
        else max(0, committed.team_state.free_transfers - draft.transfers_made)
    )
    new_team_state = replace(
        draft.working_state,
        free_transfers=new_free_transfers,
        chips_remaining=available_chips_this_gameweek(new_chip_usage, gameweek),
    )

    new_committed = CommittedSquad(
        team_state=new_team_state,
        chip_usage=new_chip_usage,
        active_chip=draft.chip,
        active_chip_gameweek=gameweek if draft.chip is not None else None,
        free_hit_snapshot=free_hit_snapshot,
        free_hit_snapshot_gameweek=free_hit_snapshot_gameweek,
        committed_gameweek=gameweek,
        gameweek_hit_cost=committed.gameweek_hit_cost + hit_cost,
    )
    return new_committed, hit_cost


def advance_gameweek(
    committed: CommittedSquad, pending: PendingDraft | None, new_gameweek: int
) -> tuple[CommittedSquad, PendingDraft | None]:
    """Called once per batch-refresh, whenever the cache's own "current gameweek" moves on.

    - If Free Hit is the active chip and ``new_gameweek`` is past the gameweek it was played for,
      restores ``team_state`` from ``free_hit_snapshot`` and clears both snapshot fields (D15).
    - Any active chip (whichever it was) is cleared once its own gameweek has passed — a chip
      other than Free Hit leaves the squad itself untouched, only that one gameweek's scoring.
    - Free transfers accrue by 1, capped at 5 (2026/27's banking rule), regardless of chip.
    - A ``pending`` draft opened for a gameweek other than ``new_gameweek`` is dropped (D24) rather
      than silently carried forward against the wrong deadline.
    - ``gameweek_hit_cost`` resets to 0 for the new gameweek (the old gameweek's hits have already
      been consumed by whoever scored that gameweek).
    """
    team_state = committed.team_state
    free_hit_snapshot = committed.free_hit_snapshot
    free_hit_snapshot_gameweek = committed.free_hit_snapshot_gameweek

    if (
        committed.active_chip == "free_hit"
        and free_hit_snapshot is not None
        and free_hit_snapshot_gameweek is not None
        and new_gameweek > free_hit_snapshot_gameweek
    ):
        team_state = free_hit_snapshot
        free_hit_snapshot = None
        free_hit_snapshot_gameweek = None

    active_chip = committed.active_chip
    active_chip_gameweek = committed.active_chip_gameweek
    if active_chip_gameweek is not None and new_gameweek > active_chip_gameweek:
        active_chip = None
        active_chip_gameweek = None

    if team_state is not None:
        team_state = replace(
            team_state,
            free_transfers=min(5, team_state.free_transfers + 1),
            chips_remaining=available_chips_this_gameweek(committed.chip_usage, new_gameweek),
        )

    new_committed = replace(
        committed,
        team_state=team_state,
        active_chip=active_chip,
        active_chip_gameweek=active_chip_gameweek,
        free_hit_snapshot=free_hit_snapshot,
        free_hit_snapshot_gameweek=free_hit_snapshot_gameweek,
        committed_gameweek=new_gameweek,
        gameweek_hit_cost=0,
    )

    new_pending = pending if pending is not None and pending.base_gameweek == new_gameweek else None

    return new_committed, new_pending
