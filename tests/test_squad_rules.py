"""Tests for features.squad_rules -- full FPL squad legality and its mutations, sandbox model
(no confirm step, no transfer economy)."""

from __future__ import annotations

import pytest

from engine.scoring import DEF, FWD, GK, MID
from features.squad_rules import (
    INITIAL_BUDGET,
    SQUAD_SIZE,
    SquadRuleError,
    add_player,
    assemble_team_state,
    build_team_state,
    optimise_xi,
    remove_player,
    reorder_bench,
    set_captain,
    set_vice_captain,
    substitute,
    transfer,
    validate_partial_squad,
    validate_squad,
    validate_xi,
)
from features.team_state import MyTeamState, SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


def _player(player_id: int, position: str, price: int = 40) -> SquadPlayer:
    return SquadPlayer(player_id=player_id, position=position, price=price)


def _squad() -> tuple[SquadPlayer, ...]:
    players = [_player(GK1, GK), _player(GK2, GK)]
    players += [_player(pid, DEF) for pid in DEF_IDS]
    players += [_player(pid, MID) for pid in MID_IDS]
    players += [_player(pid, FWD) for pid in FWD_IDS]
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
        squad = tuple(_player(p.player_id, p.position, price=100) for p in _squad())
        violations = validate_squad(squad, _team_id_by_player())
        assert any(v.code == "budget" for v in violations)

    def test_exactly_at_budget_is_legal(self):
        per_player = INITIAL_BUDGET // SQUAD_SIZE
        squad = tuple(_player(p.player_id, p.position, price=per_player) for p in _squad())
        violations = validate_squad(squad, _team_id_by_player())
        assert not any(v.code == "budget" for v in violations)

    def test_check_budget_false_skips_the_budget_check(self):
        squad = tuple(_player(p.player_id, p.position, price=100) for p in _squad())
        violations = validate_squad(squad, _team_id_by_player(), check_budget=False)
        assert not any(v.code == "budget" for v in violations)

    def test_custom_budget_ceiling_is_respected(self):
        # A real imported squad's personal ceiling can exceed the classic 100m.
        squad = tuple(_player(p.player_id, p.position, price=100) for p in _squad())
        violations = validate_squad(squad, _team_id_by_player(), budget=1500)
        assert not any(v.code == "budget" for v in violations)

    def test_reports_every_violation_not_just_the_first(self):
        team_ids = _team_id_by_player()
        for pid in DEF_IDS[:4]:
            team_ids[pid] = 999
        squad = list(_squad())
        squad[1] = _player(GK2, DEF)
        violations = validate_squad(tuple(squad), team_ids)
        assert len(violations) >= 2


class TestValidatePartialSquad:
    def test_empty_squad_has_no_violations(self):
        assert validate_partial_squad((), _team_id_by_player()) == ()

    def test_under_quota_is_legal(self):
        squad = (_player(GK1, GK),)
        assert validate_partial_squad(squad, _team_id_by_player()) == ()

    def test_exceeding_position_quota_is_flagged(self):
        squad = (_player(GK1, GK), _player(GK2, GK), _player(9001, GK))
        team_ids = {**_team_id_by_player(), 9001: 9001}
        violations = validate_partial_squad(squad, team_ids)
        assert any(v.code == "quota" for v in violations)

    def test_exceeding_club_limit_is_flagged(self):
        team_ids = _team_id_by_player()
        for pid in DEF_IDS[:4]:
            team_ids[pid] = 999
        squad = tuple(_player(pid, DEF) for pid in DEF_IDS[:4])
        violations = validate_partial_squad(squad, team_ids)
        assert any(v.code == "club_limit" for v in violations)

    def test_over_budget_is_flagged(self):
        squad = (_player(GK1, GK, price=1500),)
        violations = validate_partial_squad(squad, _team_id_by_player())
        assert any(v.code == "budget" for v in violations)


class TestAddPlayer:
    def test_adds_to_empty_squad(self):
        new_squad = add_player((), _player(GK1, GK), _team_id_by_player())
        assert new_squad == (_player(GK1, GK),)

    def test_duplicate_player_raises(self):
        squad = (_player(GK1, GK),)
        with pytest.raises(SquadRuleError) as exc_info:
            add_player(squad, _player(GK1, GK), _team_id_by_player())
        assert exc_info.value.violation.code == "duplicate"

    def test_full_squad_raises(self):
        with pytest.raises(SquadRuleError) as exc_info:
            add_player(_squad(), _player(9001, GK), {**_team_id_by_player(), 9001: 9001})
        assert exc_info.value.violation.code == "squad_full"

    def test_violating_quota_raises(self):
        squad = (_player(GK1, GK), _player(GK2, GK))
        with pytest.raises(SquadRuleError) as exc_info:
            add_player(squad, _player(9001, GK), {**_team_id_by_player(), 9001: 9001})
        assert exc_info.value.violation.code == "quota"

    def test_violating_budget_raises(self):
        with pytest.raises(SquadRuleError) as exc_info:
            add_player((), _player(GK1, GK, price=1500), _team_id_by_player())
        assert exc_info.value.violation.code == "budget"

    def test_custom_budget_ceiling_is_respected(self):
        new_squad = add_player((), _player(GK1, GK, price=1200), _team_id_by_player(), budget=1500)
        assert new_squad == (_player(GK1, GK, price=1200),)


class TestRemovePlayer:
    def test_removes_from_squad(self):
        squad = (_player(GK1, GK), _player(GK2, GK))
        new_squad = remove_player(squad, GK1)
        assert new_squad == (_player(GK2, GK),)

    def test_unknown_player_raises(self):
        with pytest.raises(SquadRuleError) as exc_info:
            remove_player((_player(GK1, GK),), 9999)
        assert exc_info.value.violation.code == "unknown_player"


class TestBuildTeamState:
    def test_promotes_a_caller_specified_arrangement_verbatim(self):
        state = build_team_state(
            squad=_squad(),
            starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
            bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
            captain_id=MID_IDS[0],
            vice_captain_id=MID_IDS[1],
            team_id_by_player=_team_id_by_player(),
        )
        assert state.captain_id == MID_IDS[0]
        assert state.starting_xi == (GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2])

    def test_illegal_squad_raises(self):
        squad = tuple(_player(p.player_id, p.position, price=100) for p in _squad())
        with pytest.raises(SquadRuleError):
            build_team_state(
                squad=squad,
                starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
                bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
                captain_id=MID_IDS[0],
                vice_captain_id=MID_IDS[1],
                team_id_by_player=_team_id_by_player(),
            )

    def test_check_budget_false_allows_a_squad_over_the_classic_ceiling(self):
        squad = tuple(_player(p.player_id, p.position, price=100) for p in _squad())
        state = build_team_state(
            squad=squad,
            starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
            bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
            captain_id=MID_IDS[0],
            vice_captain_id=MID_IDS[1],
            team_id_by_player=_team_id_by_player(),
            check_budget=False,
        )
        assert sum(p.price for p in state.squad) == 1500

    def test_bench_shape_mismatch_raises(self):
        with pytest.raises(SquadRuleError) as exc_info:
            build_team_state(
                squad=_squad(),
                starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
                bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK1),  # GK1 is in the XI already
                captain_id=MID_IDS[0],
                vice_captain_id=MID_IDS[1],
                team_id_by_player=_team_id_by_player(),
            )
        assert exc_info.value.violation.code == "xi_shape"


class TestAssembleTeamState:
    def test_auto_derives_best_xi_and_captain(self):
        expected_points = {pid: float(pid) for pid in _team_id_by_player()}
        state = assemble_team_state(_squad(), expected_points, _team_id_by_player())
        assert validate_xi(state.starting_xi, _position_by_player()) == ()
        ranked = sorted(state.starting_xi, key=lambda pid: expected_points[pid], reverse=True)
        assert state.captain_id == ranked[0]
        assert state.vice_captain_id == ranked[1]

    def test_preferred_captain_is_kept_if_still_eligible(self):
        expected_points = dict.fromkeys(_team_id_by_player(), 1.0)
        state = assemble_team_state(
            _squad(),
            expected_points,
            _team_id_by_player(),
            preferred_captain_id=MID_IDS[0],
            preferred_vice_captain_id=MID_IDS[1],
        )
        assert MID_IDS[0] in state.starting_xi  # tied EV, guaranteed to be picked by some formation
        assert state.captain_id == MID_IDS[0]
        assert state.vice_captain_id == MID_IDS[1]

    def test_illegal_squad_raises(self):
        squad = tuple(_player(p.player_id, p.position, price=100) for p in _squad())
        expected_points = {pid: 1.0 for pid in _team_id_by_player()}
        with pytest.raises(SquadRuleError):
            assemble_team_state(squad, expected_points, _team_id_by_player())


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

    def test_never_changes_squad(self):
        state = _base_state()
        new_state = substitute(
            state, out_id=FWD_IDS[1], in_id=FWD_IDS[2], position_by_player=_position_by_player()
        )
        assert new_state.squad == state.squad

    def test_does_not_mutate_the_input_state(self):
        state = _base_state()
        original_xi = state.starting_xi
        substitute(
            state, out_id=FWD_IDS[1], in_id=FWD_IDS[2], position_by_player=_position_by_player()
        )
        assert state.starting_xi == original_xi


class TestTransfer:
    def test_swaps_a_squad_player(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[2],
            in_player=_player(9001, FWD, price=45),
            team_id_by_player=team_ids,
        )
        assert 9001 in new_state.player_ids
        assert FWD_IDS[2] not in new_state.player_ids

    def test_is_always_free_regardless_of_price(self):
        # No hit cost or free-transfer count exists in the sandbox -- only legality matters.
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[2],
            in_player=_player(9001, FWD, price=1),
            team_id_by_player=team_ids,
        )
        assert 9001 in new_state.player_ids

    def test_exceeding_budget_raises(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=FWD_IDS[2],
                in_player=_player(9001, FWD, price=2000),
                team_id_by_player=team_ids,
            )
        assert exc_info.value.violation.code == "budget"

    def test_custom_budget_ceiling_is_respected(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state,
            out_id=FWD_IDS[2],
            in_player=_player(9001, FWD, price=2000),
            team_id_by_player=team_ids,
            budget=5000,
        )
        assert 9001 in new_state.player_ids

    def test_buying_a_player_already_in_the_squad_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=FWD_IDS[2],
                in_player=_player(MID_IDS[0], MID),
                team_id_by_player=_team_id_by_player(),
            )
        assert exc_info.value.violation.code == "duplicate"

    def test_selling_a_player_not_in_the_squad_raises(self):
        state = _base_state()
        with pytest.raises(SquadRuleError) as exc_info:
            transfer(
                state,
                out_id=9999,
                in_player=_player(9001, FWD),
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
                in_player=_player(9001, FWD),
                team_id_by_player=team_ids,
            )
        assert exc_info.value.violation.code == "club_limit"

    def test_transferring_in_the_starting_xi_keeps_slot_and_multiplier(self):
        state = _base_state(captain_id=FWD_IDS[0])
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state, out_id=FWD_IDS[0], in_player=_player(9001, FWD), team_id_by_player=team_ids
        )
        assert 9001 in new_state.starting_xi
        assert new_state.captain_id == 9001

    def test_transferring_in_a_bench_player_stays_on_the_bench(self):
        state = _base_state()
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        new_state = transfer(
            state, out_id=FWD_IDS[2], in_player=_player(9001, FWD), team_id_by_player=team_ids
        )
        assert 9001 in new_state.bench_order
        assert 9001 not in new_state.starting_xi

    def test_does_not_mutate_the_input_state(self):
        state = _base_state()
        original_squad = state.squad
        team_ids = _team_id_by_player()
        team_ids[9001] = 9001
        transfer(state, out_id=FWD_IDS[2], in_player=_player(9001, FWD), team_id_by_player=team_ids)
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
