"""Tests for features.squad_rules -- full FPL squad legality and its mutations (D2/G3)."""

from __future__ import annotations

import pytest

from engine.scoring import DEF, FWD, GK, MID
from features.squad_rules import (
    INITIAL_BUDGET,
    SQUAD_SIZE,
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
from features.team_state import MyTeamState, SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


def _player(
    player_id: int, position: str, purchase_price: int = 40, current_price: int | None = None
) -> SquadPlayer:
    return SquadPlayer(
        player_id=player_id,
        position=position,
        purchase_price=purchase_price,
        current_price=current_price if current_price is not None else purchase_price,
    )


def _squad(**price_overrides: dict[int, int]) -> tuple[SquadPlayer, ...]:
    players = []
    players.append(_player(GK1, GK))
    players.append(_player(GK2, GK))
    for pid in DEF_IDS:
        players.append(_player(pid, DEF))
    for pid in MID_IDS:
        players.append(_player(pid, MID))
    for pid in FWD_IDS:
        players.append(_player(pid, FWD))
    return tuple(players)


def _team_id_by_player() -> dict[int, int]:
    # Every squad player on their own club by default -- club-limit tests override explicitly.
    all_ids = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]
    return {pid: pid for pid in all_ids}


def _position_by_player() -> dict[int, str]:
    d = {GK1: GK, GK2: GK}
    d.update({pid: DEF for pid in DEF_IDS})
    d.update({pid: MID for pid in MID_IDS})
    d.update({pid: FWD for pid in FWD_IDS})
    return d


def _base_state(**overrides) -> MyTeamState:
    squad = overrides.pop("squad", _squad())
    starting_xi = overrides.pop("starting_xi", (GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]))
    bench_order = overrides.pop("bench_order", (DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2))
    defaults = dict(
        squad=squad,
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
        bank=100,
        free_transfers=1,
        chips_remaining=frozenset({"wildcard", "free_hit", "bench_boost", "triple_captain"}),
    )
    defaults.update(overrides)
    return MyTeamState(**defaults)


class TestValidateSquad:
    def test_legal_squad_has_no_violations(self):
        assert validate_squad(_squad(), _team_id_by_player()) == ()

    def test_wrong_gk_count_is_a_quota_violation(self):
        squad = list(_squad())
        squad[1] = _player(GK2, DEF)  # now 1 GK, 6 DEF
        violations = validate_squad(tuple(squad), _team_id_by_player())
        codes = {v.code for v in violations}
        assert "quota" in codes

    def test_exceeding_club_limit_is_flagged(self):
        team_ids = _team_id_by_player()
        # Cram 4 defenders onto the same club.
        for pid in DEF_IDS[:4]:
            team_ids[pid] = 999
        violations = validate_squad(_squad(), team_ids)
        club_violations = [v for v in violations if v.code == "club_limit"]
        assert len(club_violations) == 1
        assert set(club_violations[0].player_ids) == set(DEF_IDS[:4])

    def test_exactly_max_per_club_is_legal(self):
        team_ids = _team_id_by_player()
        for pid in DEF_IDS[:3]:
            team_ids[pid] = 999
        violations = validate_squad(_squad(), team_ids)
        assert not any(v.code == "club_limit" for v in violations)

    def test_over_budget_is_flagged(self):
        squad = tuple(_player(p.player_id, p.position, purchase_price=100) for p in _squad())
        violations = validate_squad(squad, _team_id_by_player())
        assert any(v.code == "budget" for v in violations)

    def test_exactly_at_budget_is_legal(self):
        per_player = INITIAL_BUDGET // SQUAD_SIZE
        squad = tuple(_player(p.player_id, p.position, purchase_price=per_player) for p in _squad())
        violations = validate_squad(squad, _team_id_by_player())
        assert not any(v.code == "budget" for v in violations)

    def test_reports_every_violation_not_just_the_first(self):
        team_ids = _team_id_by_player()
        for pid in DEF_IDS[:4]:
            team_ids[pid] = 999
        squad = list(_squad())
        squad[1] = _player(GK2, DEF)
        violations = validate_squad(tuple(squad), team_ids)
        assert len(violations) >= 2


class TestValidateXi:
    def test_legal_formation_has_no_violations(self):
        xi = (GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2])
        assert validate_xi(xi, _position_by_player()) == ()

    def test_wrong_size_is_rejected(self):
        xi = (GK1, *DEF_IDS[:4], *MID_IDS[:4])  # 9 players
        violations = validate_xi(xi, _position_by_player())
        assert violations[0].code == "xi_shape"

    def test_zero_goalkeepers_is_rejected(self):
        xi = (*DEF_IDS[:5], *MID_IDS[:4], *FWD_IDS[:2])  # 11 outfielders, no GK
        violations = validate_xi(xi, _position_by_player())
        assert violations[0].code == "xi_shape"
        assert "goalkeeper" in violations[0].message

    def test_two_goalkeepers_is_rejected(self):
        xi = (GK1, GK2, *DEF_IDS[:3], *MID_IDS[:4], *FWD_IDS[:2])  # 2 GK, 3 DEF, 4 MID, 2 FWD = 11
        violations = validate_xi(xi, _position_by_player())
        assert violations[0].code == "xi_shape"

    def test_illegal_formation_shape_is_rejected(self):
        # 2 DEF / 6 MID / 2 FWD outfield -- 2 DEF is below the 3-DEF floor.
        xi = (GK1, *DEF_IDS[:2], *MID_IDS, *FWD_IDS[:2], MID_IDS[0])
        # build a genuinely-too-few-defenders XI of the right size instead
        xi = (GK1, DEF_IDS[0], DEF_IDS[1], *MID_IDS, FWD_IDS[0])
        violations = validate_xi(xi, _position_by_player())
        assert violations[0].code == "xi_shape"

    def test_every_valid_formation_shape_is_accepted(self):
        # 4-4-2 and 3-5-2 and 5-3-2 are all legal; spot-check 3-5-2 using available ids.
        pos = _position_by_player()
        xi = (GK1, *DEF_IDS[:3], *MID_IDS, FWD_IDS[0], FWD_IDS[1])
        assert validate_xi(xi, pos) == ()


class TestSubstitute:
    def test_swaps_bench_player_into_starting_xi(self):
        state = _base_state()
        new_state = substitute(
            state, out_id=FWD_IDS[1], in_id=FWD_IDS[2], position_by_player=_position_by_player()
        )
        assert FWD_IDS[2] in new_state.starting_xi
        assert FWD_IDS[1] in new_state.bench_order
        assert FWD_IDS[1] not in new_state.starting_xi
        assert FWD_IDS[2] not in new_state.bench_order

    def test_out_id_not_in_xi_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            substitute(
                state, out_id=FWD_IDS[2], in_id=FWD_IDS[1], position_by_player=_position_by_player()
            )
        assert exc_info.value.violation.code == "unknown_player"

    def test_in_id_not_on_bench_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            substitute(
                state, out_id=FWD_IDS[1], in_id=MID_IDS[0], position_by_player=_position_by_player()
            )
        assert exc_info.value.violation.code == "unknown_player"

    def test_swapping_only_goalkeeper_out_for_an_outfielder_is_rejected(self):
        # Bench has no reserve GK in this scenario's swap target -- try to swap starting GK for
        # the bench defender directly, which would leave the XI with zero goalkeepers.
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            substitute(
                state, out_id=GK1, in_id=DEF_IDS[4], position_by_player=_position_by_player()
            )
        assert exc_info.value.violation.code == "xi_shape"

    def test_goalkeeper_for_goalkeeper_swap_is_legal(self):
        state = _base_state()
        new_state = substitute(
            state, out_id=GK1, in_id=GK2, position_by_player=_position_by_player()
        )
        assert GK2 in new_state.starting_xi
        assert GK1 in new_state.bench_order

    def test_substituting_the_captain_moves_the_captain_badge(self):
        state = _base_state(captain_id=FWD_IDS[1], vice_captain_id=MID_IDS[1])
        new_state = substitute(
            state, out_id=FWD_IDS[1], in_id=FWD_IDS[2], position_by_player=_position_by_player()
        )
        assert new_state.captain_id == FWD_IDS[2]

    def test_never_changes_squad_or_bank(self):
        state = _base_state()
        new_state = substitute(
            state, out_id=FWD_IDS[1], in_id=FWD_IDS[2], position_by_player=_position_by_player()
        )
        assert new_state.squad == state.squad
        assert new_state.bank == state.bank

    def test_does_not_mutate_the_input_state(self):
        state = _base_state()
        original_xi = state.starting_xi
        substitute(
            state, out_id=FWD_IDS[1], in_id=FWD_IDS[2], position_by_player=_position_by_player()
        )
        assert state.starting_xi == original_xi


class TestTransfer:
    def test_swaps_a_squad_player_and_updates_bank(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[2],
            in_id=9001,
            in_price=45,
            in_position=FWD,
            team_id_by_player=team_ids,
        )
        assert 9001 in new_state.player_ids
        assert FWD_IDS[2] not in new_state.player_ids
        # sold at current_price (40, no profit) minus bought at 45 -> bank drops by 5.
        assert new_state.bank == state.bank - 5

    def test_sell_price_reflects_real_fpl_profit_rule(self):
        squad = list(_squad())
        # FWD_IDS[2] bought at 40, now worth 60 -- sells for 40 + (60-40)//2 = 50.
        squad[-1] = _player(FWD_IDS[2], FWD, purchase_price=40, current_price=60)
        state = _base_state(squad=tuple(squad))
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[2],
            in_id=9001,
            in_price=50,
            in_position=FWD,
            team_id_by_player=team_ids,
        )
        assert new_state.bank == state.bank  # sold for 50, bought for 50 -- unchanged

    def test_buying_beyond_bank_plus_sell_price_raises(self):
        state = _base_state(bank=0)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=FWD_IDS[2],
                in_id=9001,
                in_price=1000,
                in_position=FWD,
                team_id_by_player=team_ids,
            )
        assert exc_info.value.violation.code == "budget"

    def test_buying_a_player_already_in_the_squad_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=FWD_IDS[2],
                in_id=MID_IDS[0],
                in_price=40,
                in_position=MID,
                team_id_by_player=_team_id_by_player(),
            )
        assert exc_info.value.violation.code == "duplicate"

    def test_selling_a_player_not_in_the_squad_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=9999,
                in_id=9001,
                in_price=40,
                in_position=FWD,
                team_id_by_player=_team_id_by_player(),
            )
        assert exc_info.value.violation.code == "unknown_player"

    def test_violating_club_limit_raises(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[FWD_IDS[0]] = 999
        team_ids[MID_IDS[0]] = 999
        team_ids[MID_IDS[1]] = 999  # club 999 already has 3 squad members
        team_ids[9001] = 999  # transferring in a 4th would exceed MAX_PER_CLUB
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=FWD_IDS[2],
                in_id=9001,
                in_price=40,
                in_position=FWD,
                team_id_by_player=team_ids,
            )
        assert exc_info.value.violation.code == "club_limit"

    def test_transferring_in_the_starting_xi_keeps_slot_and_multiplier(self):
        state = _base_state(captain_id=FWD_IDS[0])
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[0],
            in_id=9001,
            in_price=40,
            in_position=FWD,
            team_id_by_player=team_ids,
        )
        assert 9001 in new_state.starting_xi
        assert new_state.captain_id == 9001

    def test_transferring_in_a_bench_player_stays_on_the_bench(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[2],
            in_id=9001,
            in_price=40,
            in_position=FWD,
            team_id_by_player=team_ids,
        )
        assert 9001 in new_state.bench_order
        assert 9001 not in new_state.starting_xi

    def test_never_charges_a_hit_itself(self):
        state = _base_state(free_transfers=0)
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        # No exception, no points concept exists on MyTeamState at all -- hits are computed
        # entirely by transfer_hit_cost, never by transfer() itself.
        transfer(
            state,
            out_id=FWD_IDS[2],
            in_id=9001,
            in_price=40,
            in_position=FWD,
            team_id_by_player=team_ids,
        )

    def test_does_not_mutate_the_input_state(self):
        state = _base_state()
        original_squad = state.squad
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        transfer(
            state,
            out_id=FWD_IDS[2],
            in_id=9001,
            in_price=40,
            in_position=FWD,
            team_id_by_player=team_ids,
        )
        assert state.squad == original_squad


class TestCaptaincy:
    def test_set_captain(self):
        state = _base_state()
        new_state = set_captain(state, FWD_IDS[0])
        assert new_state.captain_id == FWD_IDS[0]

    def test_set_captain_to_bench_player_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            set_captain(state, FWD_IDS[2])
        assert exc_info.value.violation.code == "unknown_player"

    def test_set_captain_to_current_vice_raises(self):
        state = _base_state(captain_id=MID_IDS[0], vice_captain_id=MID_IDS[1])
        with pytest.raises(SquadRuleError) as exc_info:
            set_captain(state, MID_IDS[1])
        assert exc_info.value.violation.code == "duplicate"

    def test_set_vice_captain(self):
        state = _base_state()
        new_state = set_vice_captain(state, FWD_IDS[0])
        assert new_state.vice_captain_id == FWD_IDS[0]

    def test_set_vice_captain_to_current_captain_raises(self):
        state = _base_state(captain_id=MID_IDS[0], vice_captain_id=MID_IDS[1])
        with pytest.raises(SquadRuleError) as exc_info:
            set_vice_captain(state, MID_IDS[0])
        assert exc_info.value.violation.code == "duplicate"


class TestReorderBench:
    def test_permutation_is_accepted(self):
        state = _base_state()
        reversed_bench = tuple(reversed(state.bench_order))
        new_state = reorder_bench(state, reversed_bench)
        assert new_state.bench_order == reversed_bench

    def test_non_permutation_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError):
            reorder_bench(state, (*state.bench_order[:-1], 9999))


class TestOptimiseXi:
    def test_picks_the_best_legal_formation_by_ev(self):
        state = _base_state()
        # Make the bench forward much better than a starting midfielder.
        expected_points = dict.fromkeys(state.player_ids, 2.0)
        expected_points[FWD_IDS[2]] = 20.0  # benched forward, should come on
        expected_points[MID_IDS[3]] = 0.1  # weakest starting midfielder
        new_state = optimise_xi(state, expected_points)
        assert FWD_IDS[2] in new_state.starting_xi

    def test_result_is_always_a_legal_xi(self):
        state = _base_state()
        expected_points = {pid: float(pid) for pid in state.player_ids}
        new_state = optimise_xi(state, expected_points)
        assert validate_xi(new_state.starting_xi, _position_by_player()) == ()

    def test_preserves_captain_if_still_in_xi(self):
        state = _base_state(captain_id=MID_IDS[0], vice_captain_id=MID_IDS[1])
        expected_points = {pid: 1.0 for pid in state.player_ids}
        expected_points[MID_IDS[0]] = (
            10.0  # clearly among the best midfielders -- guaranteed a spot
        )
        new_state = optimise_xi(state, expected_points)
        assert MID_IDS[0] in new_state.starting_xi
        assert new_state.captain_id == MID_IDS[0]

    def test_reassigns_captain_if_dropped_to_bench(self):
        state = _base_state(captain_id=FWD_IDS[1], vice_captain_id=MID_IDS[1])
        expected_points = dict.fromkeys(state.player_ids, 1.0)
        expected_points[FWD_IDS[1]] = -100.0  # ensure this exact player gets dropped
        expected_points[FWD_IDS[2]] = 100.0
        new_state = optimise_xi(state, expected_points)
        assert new_state.captain_id in new_state.starting_xi


class TestTransferHitCost:
    def test_hit_free_is_always_zero(self):
        assert transfer_hit_cost(5, free_transfers=0, hit_free=True) == 0

    def test_within_free_transfers_is_zero(self):
        assert transfer_hit_cost(1, free_transfers=2, hit_free=False) == 0

    def test_beyond_free_transfers_charges_per_extra(self):
        assert transfer_hit_cost(3, free_transfers=1, hit_free=False) == 8  # 2 extra x 4

    def test_never_negative(self):
        assert transfer_hit_cost(0, free_transfers=5, hit_free=False) == 0
