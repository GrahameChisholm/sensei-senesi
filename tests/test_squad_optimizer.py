"""Tests for features.squad_optimizer -- the best-possible-squad ILP solver."""

from __future__ import annotations

import pytest

from engine.scoring import DEF, FWD, GK, MID
from features.formation import VALID_FORMATIONS, select_starting_xi
from features.squad_optimizer import (
    OptimizedSquad,
    PlayerCandidate,
    SquadOptimizerError,
    optimise_squad,
)
from features.squad_rules import SquadRuleError, validate_squad

# A small pool: 2 GK, 6 DEF, 6 MID, 4 FWD across 2 clubs, cheap enough for a full rebuild to fit
# comfortably inside the classic £100m (1000) budget.


def _candidate(
    player_id: int, position: str, team_id: int, price: int, expected_points: float
) -> PlayerCandidate:
    return PlayerCandidate(
        player_id=player_id,
        position=position,
        team_id=team_id,
        price=price,
        expected_points=expected_points,
    )


def _pool() -> list[PlayerCandidate]:
    # Every player on their own distinct club by default, so the max-3-per-club rule is never
    # accidentally forced into infeasibility -- dedicated tests below construct their own
    # deliberately-clustered clubs to exercise that rule specifically.
    pool = []
    for i, pid in enumerate([1, 2]):  # GK
        pool.append(_candidate(pid, GK, team_id=pid, price=40, expected_points=3.0 + i))
    for i, pid in enumerate(range(11, 17)):  # DEF
        pool.append(_candidate(pid, DEF, team_id=pid, price=40, expected_points=3.0 + i * 0.1))
    for i, pid in enumerate(range(21, 27)):  # MID
        pool.append(_candidate(pid, MID, team_id=pid, price=50, expected_points=4.0 + i * 0.1))
    for i, pid in enumerate(range(31, 35)):  # FWD
        pool.append(_candidate(pid, FWD, team_id=pid, price=55, expected_points=5.0 + i * 0.1))
    return pool


def _team_id_by_player(candidates: list[PlayerCandidate]) -> dict[int, int]:
    return {c.player_id: c.team_id for c in candidates}


class TestOptimiseSquad:
    def test_full_rebuild_is_legal(self):
        pool = _pool()
        result = optimise_squad(pool)
        assert isinstance(result, OptimizedSquad)
        assert len(result.squad) == 15
        violations = validate_squad(result.squad, _team_id_by_player(pool))
        assert violations == ()

    def test_full_rebuild_picks_the_highest_ev_legal_squad(self):
        pool = _pool()
        result = optimise_squad(pool)
        # Every position should pick its highest-EV members (all equally priced within position).
        chosen = {p.player_id for p in result.squad}
        best_def = {11, 12, 13, 14, 15, 16}  # all 6 DEF fit the quota of 5... check top 5 by EV
        top5_def = sorted(best_def, reverse=True)[:5]
        assert set(top5_def).issubset(chosen)

    def test_locked_players_are_preserved_verbatim(self):
        pool = _pool()
        locked = frozenset({1, 21})
        result = optimise_squad(pool, locked_player_ids=locked)
        chosen = {p.player_id for p in result.squad}
        assert locked.issubset(chosen)

    def test_locked_players_subsuming_full_rebuild_when_empty(self):
        pool = _pool()
        empty_result = optimise_squad(pool, locked_player_ids=frozenset())
        full_result = optimise_squad(pool)
        assert {p.player_id for p in empty_result.squad} == {p.player_id for p in full_result.squad}

    def test_respects_max_per_club(self):
        # One club fields 4 excellent defenders (more than MAX_PER_CLUB), plus enough other clubs
        # to fill the remaining quota -- the optimizer must leave one of the 4 out.
        pool = _pool()
        pool += [
            _candidate(900 + i, DEF, team_id=500, price=40, expected_points=99.0) for i in range(4)
        ]
        result = optimise_squad(pool)
        club_500_count = sum(1 for p in result.squad if p.player_id >= 900)
        assert club_500_count <= 3

    def test_infeasible_club_overload_across_positions_raises(self):
        # A club with a candidate in every position, each forced (exact GK quota, near-exact
        # DEF/MID/FWD quota), can genuinely exceed MAX_PER_CLUB with no legal way around it.
        pool = [
            _candidate(1, GK, team_id=1, price=40, expected_points=1.0),
            _candidate(2, GK, team_id=1, price=40, expected_points=1.0),
        ]
        pool += [
            _candidate(100 + i, DEF, team_id=1, price=40, expected_points=1.0) for i in range(5)
        ]
        pool += [
            _candidate(200 + i, MID, team_id=1, price=40, expected_points=1.0) for i in range(5)
        ]
        pool += [
            _candidate(300 + i, FWD, team_id=1, price=40, expected_points=1.0) for i in range(3)
        ]
        with pytest.raises(SquadOptimizerError):
            optimise_squad(pool)

    def test_prefers_higher_ev_over_cheaper_at_equal_price(self):
        # Two disjoint MID candidate sets at the same price -- one clearly higher EV. GK/DEF/FWD
        # sit on clubs 1000+, well clear of the MID candidates' clubs, so this only exercises the
        # EV-vs-EV choice, not an incidental club-limit interaction.
        pool = [
            _candidate(1, GK, 1000, 40, 1.0),
            _candidate(2, GK, 1001, 40, 1.0),
        ]
        pool += [_candidate(100 + i, DEF, 1002 + i, 40, 1.0) for i in range(5)]
        pool += [_candidate(200 + i, FWD, 1007 + i, 40, 1.0) for i in range(3)]
        # 10 MID candidates at the same price, split into a low-EV set and a high-EV set of 5,
        # each spread across its own 5 distinct clubs.
        low_ev = [_candidate(300 + i, MID, 1 + i, 45, 2.0) for i in range(5)]
        high_ev = [_candidate(310 + i, MID, 6 + i, 45, 50.0) for i in range(5)]
        pool += low_ev + high_ev

        result = optimise_squad(pool)
        chosen = {p.player_id for p in result.squad}
        assert {c.player_id for c in high_ev}.issubset(chosen)
        assert not {c.player_id for c in low_ev} & chosen

    def test_starting_xi_matches_select_starting_xi_cross_check(self):
        pool = _pool()
        result = optimise_squad(pool)
        expected_points = {c.player_id: c.expected_points for c in pool}
        expected_xi, expected_bench = select_starting_xi(result.squad, expected_points)
        assert set(result.starting_xi) == set(expected_xi)
        assert set(result.bench_order) == set(expected_bench)

    def test_starting_xi_is_a_valid_formation(self):
        pool = _pool()
        result = optimise_squad(pool)
        position_by_player = {c.player_id: c.position for c in pool}
        counts = {GK: 0, DEF: 0, MID: 0, FWD: 0}
        for pid in result.starting_xi:
            counts[position_by_player[pid]] += 1
        assert len(result.starting_xi) == 11
        assert counts[GK] == 1
        assert (counts[DEF], counts[MID], counts[FWD]) in VALID_FORMATIONS

    def test_captain_and_vice_are_the_top_two_xi_scorers(self):
        pool = _pool()
        result = optimise_squad(pool)
        expected_points = {c.player_id: c.expected_points for c in pool}
        ranked = sorted(result.starting_xi, key=lambda pid: expected_points[pid], reverse=True)
        assert result.captain_id == ranked[0]
        assert result.vice_captain_id == ranked[1]

    def test_full_squad_objective_can_change_the_bench(self):
        # Goalkeepers make this deterministic with no formation ambiguity: exactly 1 of the 2
        # squad GKs ever starts (always the highest-EV one), so the other is always the reserve,
        # regardless of formation. Base pool GK 2 (EV 4.0) always starts and is excluded from this
        # comparison; the contest is between GK 1 (EV 3.0, price 40) and a new GK 999 (EV 3.5,
        # price 200) for the one remaining, reserve-only squad slot.
        pool = _pool()
        pool.append(_candidate(999, GK, team_id=9, price=200, expected_points=3.5))
        starting_xi_result = optimise_squad(pool, objective="starting_xi")
        full_squad_result = optimise_squad(pool, objective="full_squad")
        assert {p.player_id for p in starting_xi_result.squad} & {1, 999} == {1}
        assert {p.player_id for p in full_squad_result.squad} & {1, 999} == {999}

    def test_invalid_objective_raises(self):
        with pytest.raises(ValueError):
            optimise_squad(_pool(), objective="nonsense")

    def test_duplicate_candidate_ids_raise(self):
        pool = _pool()
        pool.append(pool[0])
        with pytest.raises(ValueError):
            optimise_squad(pool)

    def test_unknown_locked_player_id_raises(self):
        with pytest.raises(ValueError):
            optimise_squad(_pool(), locked_player_ids=frozenset({999999}))

    def test_locked_selection_violating_club_limit_raises_squad_rule_error(self):
        pool = _pool()
        # Force 4 DEF candidates onto the same club, then lock all 4.
        forced_club_pool = [
            _candidate(
                c.player_id,
                c.position,
                1 if c.position == DEF else c.team_id,
                c.price,
                c.expected_points,
            )
            for c in pool
        ]
        locked = frozenset({11, 12, 13, 14})
        with pytest.raises(SquadRuleError) as exc_info:
            optimise_squad(forced_club_pool, locked_player_ids=locked)
        assert exc_info.value.violation.code == "club_limit"

    def test_locked_selection_over_budget_raises_squad_rule_error(self):
        pool = _pool()
        expensive_pool = [
            _candidate(
                c.player_id,
                c.position,
                c.team_id,
                1500 if c.player_id == 1 else c.price,
                c.expected_points,
            )
            for c in pool
        ]
        with pytest.raises(SquadRuleError) as exc_info:
            optimise_squad(expensive_pool, locked_player_ids=frozenset({1}))
        assert exc_info.value.violation.code == "budget"

    def test_infeasible_remaining_budget_raises_optimizer_error(self):
        # Every unlocked candidate is unaffordable given the tiny remaining budget.
        pool = [
            _candidate(1, GK, 1, 40, 5.0),
            _candidate(2, GK, 2, 999, 1.0),
        ]
        pool += [_candidate(100 + i, DEF, 1 + i, 999, 1.0) for i in range(5)]
        pool += [_candidate(200 + i, MID, 1 + i, 999, 1.0) for i in range(5)]
        pool += [_candidate(300 + i, FWD, 1 + i, 999, 1.0) for i in range(3)]
        with pytest.raises(SquadOptimizerError):
            optimise_squad(pool, locked_player_ids=frozenset({1}), budget=100)
