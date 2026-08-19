"""Tests for features.squad_draft -- the preview-then-confirm state machine (D16-D24/G8),
including the full worked Free Hit example from the team-page plan's own §8.4.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from engine.scoring import DEF, FWD, GK, MID
from features.chip_calendar import ChipUsage
from features.squad_draft import (
    CommittedSquad,
    advance_gameweek,
    apply_optimise_xi_to_draft,
    apply_reorder_bench_to_draft,
    apply_set_captain_to_draft,
    apply_set_vice_captain_to_draft,
    apply_substitute_to_draft,
    apply_transfer_to_draft,
    confirm_draft,
    confirm_initial_squad,
    open_draft,
    set_draft_chip,
)
from features.squad_rules import SquadRuleError
from features.team_state import SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


def _player(player_id: int, position: str, price: int = 40) -> SquadPlayer:
    return SquadPlayer(
        player_id=player_id, position=position, purchase_price=price, current_price=price
    )


def _squad() -> tuple[SquadPlayer, ...]:
    players = [_player(GK1, GK), _player(GK2, GK)]
    players += [_player(pid, DEF) for pid in DEF_IDS]
    players += [_player(pid, MID) for pid in MID_IDS]
    players += [_player(pid, FWD) for pid in FWD_IDS]
    return tuple(players)


def _team_id_by_player() -> dict[int, int]:
    all_ids = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]
    return {pid: pid for pid in all_ids}


def _position_by_player() -> dict[int, str]:
    d = {GK1: GK, GK2: GK}
    d.update({pid: DEF for pid in DEF_IDS})
    d.update({pid: MID for pid in MID_IDS})
    d.update({pid: FWD for pid in FWD_IDS})
    return d


def _starting_xi() -> tuple[int, ...]:
    return (GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2])


def _bench_order() -> tuple[int, ...]:
    return (DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2)


def _initial_committed(gameweek: int = 1) -> CommittedSquad:
    return confirm_initial_squad(
        squad=_squad(),
        starting_xi=_starting_xi(),
        bench_order=_bench_order(),
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
        team_id_by_player=_team_id_by_player(),
        gameweek=gameweek,
    )


class TestConfirmInitialSquad:
    def test_produces_a_real_committed_squad(self):
        committed = _initial_committed()
        assert committed.team_state is not None
        assert committed.team_state.squad == _squad()

    def test_bank_is_budget_minus_total_spend(self):
        committed = _initial_committed()
        assert committed.team_state.bank == 1000 - 15 * 40

    def test_illegal_squad_raises(self):
        squad = list(_squad())
        squad[1] = _player(GK2, DEF)  # now 1 GK, 6 DEF
        with pytest.raises(SquadRuleError):
            confirm_initial_squad(
                squad=tuple(squad),
                starting_xi=_starting_xi(),
                bench_order=_bench_order(),
                captain_id=MID_IDS[0],
                vice_captain_id=MID_IDS[1],
                team_id_by_player=_team_id_by_player(),
                gameweek=1,
            )

    def test_starts_with_no_chips_used(self):
        committed = _initial_committed()
        assert committed.chip_usage == ChipUsage()
        assert committed.team_state.chips_remaining == frozenset(
            {"wildcard", "free_hit", "bench_boost", "triple_captain"}
        )


class TestOpenDraft:
    def test_requires_an_existing_committed_squad(self):
        with pytest.raises(ValueError, match="no committed squad"):
            open_draft(CommittedSquad(team_state=None), gameweek=1)

    def test_copies_the_committed_squad_exactly(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        assert draft.working_state == committed.team_state
        assert draft.transfers_made == 0
        assert draft.chip is None
        assert draft.base_gameweek == 2


class TestApplyMutationsToDraft:
    def test_substitute_does_not_increment_transfers_made(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        draft = apply_substitute_to_draft(draft, FWD_IDS[1], FWD_IDS[2], _position_by_player())
        assert draft.transfers_made == 0
        assert FWD_IDS[2] in draft.working_state.starting_xi

    def test_transfer_increments_transfers_made(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        assert draft.transfers_made == 1
        assert 9001 in draft.working_state.player_ids

    def test_two_transfers_increment_twice(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        team_ids[9002] = 9002
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        draft = apply_transfer_to_draft(draft, MID_IDS[4], 9002, 40, MID, team_ids)
        assert draft.transfers_made == 2

    def test_captaincy_and_bench_and_optimise_wrappers_work(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        draft = apply_set_captain_to_draft(draft, FWD_IDS[0])
        assert draft.working_state.captain_id == FWD_IDS[0]
        draft = apply_set_vice_captain_to_draft(draft, FWD_IDS[1])
        assert draft.working_state.vice_captain_id == FWD_IDS[1]
        reversed_bench = tuple(reversed(draft.working_state.bench_order))
        draft = apply_reorder_bench_to_draft(draft, reversed_bench)
        assert draft.working_state.bench_order == reversed_bench
        expected_points = {pid: float(pid) for pid in draft.working_state.player_ids}
        draft = apply_optimise_xi_to_draft(draft, expected_points)
        assert len(draft.working_state.starting_xi) == 11

    def test_illegal_transfer_raises_and_leaves_draft_unusable_for_that_call(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        with pytest.raises(SquadRuleError):
            apply_transfer_to_draft(draft, FWD_IDS[2], MID_IDS[0], 40, MID, _team_id_by_player())


class TestSetDraftChip:
    def test_sets_the_chip(self):
        draft = open_draft(_initial_committed(), gameweek=2)
        draft = set_draft_chip(draft, "wildcard")
        assert draft.chip == "wildcard"

    def test_clears_the_chip(self):
        draft = open_draft(_initial_committed(), gameweek=2)
        draft = set_draft_chip(draft, "wildcard")
        draft = set_draft_chip(draft, None)
        assert draft.chip is None

    def test_unknown_chip_raises(self):
        draft = open_draft(_initial_committed(), gameweek=2)
        with pytest.raises(ValueError):
            set_draft_chip(draft, "not_a_real_chip")


class TestConfirmDraft:
    def test_stale_draft_is_rejected(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        with pytest.raises(SquadRuleError) as exc_info:
            confirm_draft(committed, draft, gameweek=3, deadline_passed=True)
        assert exc_info.value.violation.code == "draft_stale"

    def test_unavailable_chip_is_rejected(self):
        committed = _initial_committed()
        used_usage = ChipUsage(first_half_played=frozenset({"wildcard"}))
        committed = CommittedSquad(
            team_state=committed.team_state, chip_usage=used_usage, committed_gameweek=1
        )
        draft = open_draft(committed, gameweek=2)
        draft = set_draft_chip(draft, "wildcard")
        with pytest.raises(SquadRuleError) as exc_info:
            confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert exc_info.value.violation.code == "chip_unavailable"

    def test_no_hit_before_deadline_regardless_of_transfer_count(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        team_ids[9002] = 9002
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        draft = apply_transfer_to_draft(draft, MID_IDS[4], 9002, 40, MID, team_ids)
        new_committed, hit_cost = confirm_draft(committed, draft, gameweek=2, deadline_passed=False)
        assert hit_cost == 0

    def test_normal_transfer_beyond_free_transfers_is_charged_after_deadline(self):
        committed = _initial_committed()
        assert committed.team_state.free_transfers == 1
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        team_ids[9002] = 9002
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        draft = apply_transfer_to_draft(draft, MID_IDS[4], 9002, 40, MID, team_ids)
        new_committed, hit_cost = confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert hit_cost == 4  # 2 transfers, 1 free -> 1 chargeable x 4

    def test_within_free_transfers_is_not_charged_after_deadline(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        new_committed, hit_cost = confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert hit_cost == 0
        assert new_committed.team_state.free_transfers == 0

    def test_wildcard_transfers_are_hit_free_even_after_the_deadline(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        draft = set_draft_chip(draft, "wildcard")
        team_ids = _team_id_by_player()
        for i, pid in enumerate([9001, 9002, 9003]):
            team_ids[pid] = pid
            out_id = [FWD_IDS[2], MID_IDS[4], DEF_IDS[4]][i]
            position = [FWD, MID, DEF][i]
            draft = apply_transfer_to_draft(draft, out_id, pid, 40, position, team_ids)
        new_committed, hit_cost = confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert hit_cost == 0

    def test_playing_wildcard_marks_it_used_and_keeps_the_other_three_available(self):
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        draft = set_draft_chip(draft, "wildcard")
        new_committed, _ = confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert "wildcard" not in new_committed.team_state.chips_remaining
        assert "bench_boost" in new_committed.team_state.chips_remaining
        assert new_committed.active_chip == "wildcard"
        assert new_committed.active_chip_gameweek == 2

    def test_free_transfers_reduced_by_chargeable_transfers_not_reset_to_zero_always(self):
        committed = _initial_committed()
        team_state = replace(committed.team_state, free_transfers=3)
        committed = CommittedSquad(
            team_state=team_state,
            chip_usage=committed.chip_usage,
            committed_gameweek=committed.committed_gameweek,
        )
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        team_ids[9002] = 9002
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        draft = apply_transfer_to_draft(draft, MID_IDS[4], 9002, 40, MID, team_ids)
        new_committed, hit_cost = confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert new_committed.team_state.free_transfers == 1  # 3 banked - 2 used
        assert hit_cost == 0

    def test_hit_cost_accumulates_across_repeated_confirms_in_the_same_gameweek(self):
        # Season Replay's POST /squad/advance needs the *total* hit charged for a gameweek, even
        # if the manager confirmed more than once before advancing.
        committed = _initial_committed()
        draft = open_draft(committed, gameweek=2)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        team_ids[9002] = 9002
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        draft = apply_transfer_to_draft(draft, MID_IDS[4], 9002, 40, MID, team_ids)
        committed, hit_cost_1 = confirm_draft(committed, draft, gameweek=2, deadline_passed=True)
        assert hit_cost_1 == 4
        assert committed.gameweek_hit_cost == 4

        draft2 = open_draft(committed, gameweek=2)
        team_ids[9003] = 9003
        draft2 = apply_transfer_to_draft(draft2, DEF_IDS[4], 9003, 40, DEF, team_ids)
        committed, hit_cost_2 = confirm_draft(committed, draft2, gameweek=2, deadline_passed=True)
        assert hit_cost_2 == 4  # this draft's own single transfer, 0 free -> 1 chargeable x 4
        assert committed.gameweek_hit_cost == 8  # accumulated across both confirms


class TestFreeHitWorkedExample:
    """The full worked example from the team-page plan's own §8.4."""

    def test_free_hit_reverts_exactly_one_gameweek_later(self):
        # GW1: initial committed squad, no chip active.
        committed = _initial_committed(gameweek=1)
        original_team_state = committed.team_state

        # GW3: play Free Hit with 4 transfers.
        draft = open_draft(committed, gameweek=3)
        draft = set_draft_chip(draft, "free_hit")
        team_ids = _team_id_by_player()
        outs = [FWD_IDS[2], MID_IDS[4], DEF_IDS[4], GK2]
        ins = [9001, 9002, 9003, 9004]
        positions = [FWD, MID, DEF, GK]
        for out_id, in_id, position in zip(outs, ins, positions, strict=True):
            team_ids[in_id] = in_id
            draft = apply_transfer_to_draft(draft, out_id, in_id, 40, position, team_ids)

        committed_gw3, hit_cost = confirm_draft(committed, draft, gameweek=3, deadline_passed=True)
        assert hit_cost == 0  # Free Hit -- no hit regardless of 4 transfers
        assert committed_gw3.active_chip == "free_hit"
        assert committed_gw3.active_chip_gameweek == 3
        assert committed_gw3.free_hit_snapshot == original_team_state
        assert committed_gw3.free_hit_snapshot_gameweek == 3
        assert set(ins).issubset(committed_gw3.team_state.player_ids)
        assert "free_hit" not in committed_gw3.team_state.chips_remaining

        # GW4 batch refresh: advance_gameweek should restore the pre-Free-Hit squad.
        committed_gw4, pending = advance_gameweek(committed_gw3, pending=None, new_gameweek=4)
        assert committed_gw4.team_state.squad == original_team_state.squad
        assert committed_gw4.active_chip is None
        assert committed_gw4.free_hit_snapshot is None
        assert committed_gw4.free_hit_snapshot_gameweek is None
        # The chip stays marked as used for the rest of the half, even after reverting.
        assert "free_hit" not in committed_gw4.team_state.chips_remaining

    def test_free_hit_does_not_revert_before_its_own_gameweek_passes(self):
        committed = _initial_committed(gameweek=1)
        draft = open_draft(committed, gameweek=3)
        draft = set_draft_chip(draft, "free_hit")
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        committed_gw3, _ = confirm_draft(committed, draft, gameweek=3, deadline_passed=True)

        # Same gameweek, not yet advanced -- must NOT revert.
        unchanged, _ = advance_gameweek(committed_gw3, pending=None, new_gameweek=3)
        assert 9001 in unchanged.team_state.player_ids

    def test_wildcard_never_reverts(self):
        committed = _initial_committed(gameweek=1)
        draft = open_draft(committed, gameweek=3)
        draft = set_draft_chip(draft, "wildcard")
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        committed_gw3, _ = confirm_draft(committed, draft, gameweek=3, deadline_passed=True)

        committed_gw4, _ = advance_gameweek(committed_gw3, pending=None, new_gameweek=4)
        assert 9001 in committed_gw4.team_state.player_ids  # permanent, unlike Free Hit


class TestAdvanceGameweek:
    def test_free_transfers_accrue_by_one_capped_at_five(self):
        committed = _initial_committed(gameweek=1)
        new_committed, _ = advance_gameweek(committed, pending=None, new_gameweek=2)
        assert new_committed.team_state.free_transfers == 2

    def test_free_transfers_cap_at_five(self):
        committed = _initial_committed(gameweek=1)
        team_state = replace(committed.team_state, free_transfers=5)
        committed = CommittedSquad(
            team_state=team_state, chip_usage=committed.chip_usage, committed_gameweek=1
        )
        new_committed, _ = advance_gameweek(committed, pending=None, new_gameweek=2)
        assert new_committed.team_state.free_transfers == 5

    def test_bench_boost_active_chip_cleared_next_gameweek_without_touching_squad(self):
        committed = _initial_committed(gameweek=1)
        draft = open_draft(committed, gameweek=1)
        draft = set_draft_chip(draft, "bench_boost")
        committed_gw1, _ = confirm_draft(committed, draft, gameweek=1, deadline_passed=False)
        assert committed_gw1.active_chip == "bench_boost"

        new_committed, _ = advance_gameweek(committed_gw1, pending=None, new_gameweek=2)
        assert new_committed.active_chip is None
        assert new_committed.team_state.squad == committed_gw1.team_state.squad

    def test_stale_pending_draft_is_dropped(self):
        committed = _initial_committed(gameweek=1)
        draft = open_draft(committed, gameweek=1)
        new_committed, new_pending = advance_gameweek(committed, pending=draft, new_gameweek=2)
        assert new_pending is None

    def test_matching_pending_draft_survives(self):
        committed = _initial_committed(gameweek=1)
        draft = open_draft(committed, gameweek=2)
        new_committed, new_pending = advance_gameweek(committed, pending=draft, new_gameweek=2)
        assert new_pending is draft

    def test_does_not_mutate_the_input_committed_squad(self):
        committed = _initial_committed(gameweek=1)
        original_free_transfers = committed.team_state.free_transfers
        advance_gameweek(committed, pending=None, new_gameweek=2)
        assert committed.team_state.free_transfers == original_free_transfers

    def test_gameweek_hit_cost_resets_to_zero_for_the_new_gameweek(self):
        committed = _initial_committed(gameweek=1)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        team_ids[9002] = 9002
        draft = open_draft(committed, gameweek=1)
        draft = apply_transfer_to_draft(draft, FWD_IDS[2], 9001, 40, FWD, team_ids)
        draft = apply_transfer_to_draft(draft, MID_IDS[4], 9002, 40, MID, team_ids)
        committed, hit_cost = confirm_draft(committed, draft, gameweek=1, deadline_passed=True)
        assert committed.gameweek_hit_cost == 4

        new_committed, _ = advance_gameweek(committed, pending=None, new_gameweek=2)
        assert new_committed.gameweek_hit_cost == 0
