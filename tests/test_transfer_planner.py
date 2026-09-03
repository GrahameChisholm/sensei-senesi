"""Tests for features.transfer_planner (TRANSFER_BANNER) over hand-built squads, pools, and
league fixtures, no I/O.

The claims under most scrutiny are the two the module's docstring stakes itself on: that expected
swing and expected points move together exactly for a transfer (so the league can only be earning
its place through variance), and that the ranking actually changes when the manager's league
position changes while every projection stays fixed. A feature whose whole premise is "your mini
league changes the answer" is worth nothing if the answer never moves.
"""

from __future__ import annotations

import numpy as np
import pytest

from engine.aggregate import ComponentBreakdown
from engine.data.league_state_builder import LeagueEntry
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerGameweekProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.scoring import DEF, FWD, GK, MID
from engine.simulate import PlayerSimulationSummary
from features.mini_league import compute_league_ownership
from features.squad_optimizer import PlayerCandidate, optimise_squad
from features.squad_rules import validate_squad
from features.team_state import MyTeamState, SquadPlayer
from features.transfer_planner import RANK_TOLERANCE, pair_moves, plan_transfers

GAMEWEEK = 1

# A pool wide enough for a real choice at every position (4 GK, 8 DEF, 8 MID, 6 FWD), each player
# on his own club so max-3-per-club never accidentally drives the answer, and cheap enough that a
# 15 fits inside the classic 1000 budget with room to upgrade.
GK_IDS = [1, 2, 3, 4]
DEF_IDS = [11, 12, 13, 14, 15, 16, 17, 18]
MID_IDS = [21, 22, 23, 24, 25, 26, 27, 28]
FWD_IDS = [31, 32, 33, 34, 35, 36]

# The 15 the manager starts with: the first of each position block, which are deliberately the
# *lower* scoring ones, so there is always an upgrade available to find.
OWNED = GK_IDS[:2] + DEF_IDS[:5] + MID_IDS[:5] + FWD_IDS[:3]

POSITION_OF = {
    **{pid: GK for pid in GK_IDS},
    **{pid: DEF for pid in DEF_IDS},
    **{pid: MID for pid in MID_IDS},
    **{pid: FWD for pid in FWD_IDS},
}


def _points(player_id: int) -> float:
    """Rising with the index inside each position block, so the owned players are always the
    weakest and every candidate has a distinct score (no solver ties to reason about)."""
    block = [GK_IDS, DEF_IDS, MID_IDS, FWD_IDS]
    for offset, ids in enumerate(block):
        if player_id in ids:
            return 2.0 + offset * 0.5 + ids.index(player_id) * 0.4
    raise KeyError(player_id)


def _price(player_id: int) -> int:
    return 45


def _minutes() -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )


def _breakdown(total: float) -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=2.0,
        goals=total - 2.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )


def _simulation(player_id: int, std: float) -> PlayerSimulationSummary:
    return PlayerSimulationSummary(
        player_id=player_id,
        mean=0.0,
        median=0.0,
        floor=0.0,
        ceiling=0.0,
        prob_big_haul=0.0,
        raw_points=np.array([]),
        std=std,
    )


def _gameweek_projection(player_id: int, std: float = 3.0) -> PlayerGameweekProjection:
    return project_player_gameweek(
        player_id,
        POSITION_OF[player_id],
        GAMEWEEK,
        _minutes(),
        _breakdown(_points(player_id)),
        simulation=_simulation(player_id, std),
    )


def _projections() -> tuple[dict, dict]:
    """``(horizon_projections, gameweek_projections)`` over the same single gameweek, which is what
    ``api.transfer_panel`` hands the planner in the one-gameweek default case."""
    gameweek = {pid: _gameweek_projection(pid) for pid in POSITION_OF}
    horizon = {
        pid: project_player_horizon(pid, POSITION_OF[pid], {GAMEWEEK: projection})
        for pid, projection in gameweek.items()
    }
    return horizon, gameweek


def _candidates() -> list[PlayerCandidate]:
    return [
        PlayerCandidate(
            player_id=pid,
            position=POSITION_OF[pid],
            team_id=pid,
            price=_price(pid),
            expected_points=_points(pid),
        )
        for pid in POSITION_OF
    ]


def _team_state() -> MyTeamState:
    """The owned 15, with its best legal XI derived by the optimizer so the fixture is guaranteed
    legal rather than hand-asserted."""
    owned = frozenset(OWNED)
    result = optimise_squad(
        [c for c in _candidates() if c.player_id in owned],
        locked_player_ids=owned,
    )
    return MyTeamState(
        squad=result.squad,
        starting_xi=result.starting_xi,
        bench_order=result.bench_order,
        captain_id=result.captain_id,
        vice_captain_id=result.vice_captain_id,
    )


def _rival(entry_id: int, picks: dict[int, int], total_points: int) -> LeagueEntry:
    return LeagueEntry(
        entry_id=entry_id,
        manager_name=f"manager {entry_id}",
        team_name=f"team {entry_id}",
        rank=entry_id,
        total_points=total_points,
        gameweek_points=0,
        picks=picks,
        chips=(),
    )


class _Snapshot:
    """A minimal stand-in for LeagueSnapshot, since compute_league_ownership only ever reads
    ``entries``."""

    def __init__(self, entries):
        self.entries = entries


def _league(total_points: int = 100, n_rivals: int = 4):
    """A field that owns the *upgrade* targets the manager does not: the second half of each
    position block, started, so bringing one in is a template cover and leaving him out is a
    differential."""
    template = {pid: 1 for pid in DEF_IDS[5:] + MID_IDS[5:] + FWD_IDS[3:] + GK_IDS[2:3]}
    rivals = [_rival(100 + i, dict(template), total_points) for i in range(n_rivals)]
    ownership = compute_league_ownership(_Snapshot(tuple(rivals)), exclude_entry_id=-1)
    return tuple(rivals), ownership


def _plan(max_transfers: int = 1, my_total_points: int = 100, n_rivals: int = 4, **kwargs):
    horizon, gameweek = _projections()
    rivals, ownership = _league(n_rivals=n_rivals)
    return plan_transfers(
        _team_state(),
        _candidates(),
        horizon,
        [GAMEWEEK],
        gameweek,
        ownership,
        rivals,
        league_gameweek=GAMEWEEK,
        my_total_points=my_total_points,
        gameweeks_remaining=10,
        budget=1000,
        max_transfers=max_transfers,
        **kwargs,
    )


class TestPairMoves:
    def test_pairs_within_position_and_orders_by_price_delta(self):
        out_players = (SquadPlayer(1, MID, 80), SquadPlayer(2, DEF, 60))
        in_players = (SquadPlayer(3, DEF, 45), SquadPlayer(4, MID, 95))
        moves = pair_moves(out_players, in_players)
        assert [(m.out_player_id, m.in_player_id) for m in moves] == [(2, 3), (1, 4)]
        assert [m.price_delta for m in moves] == [-15, 15]

    def test_rejects_a_position_mismatch(self):
        with pytest.raises(ValueError, match="position for position"):
            pair_moves((SquadPlayer(1, MID, 80),), (SquadPlayer(2, DEF, 80),))


class TestPlanTransfers:
    def test_suggests_a_legal_squad_within_the_transfer_limit(self):
        suggestion = _plan(max_transfers=2)
        assert suggestion.plans
        for plan in suggestion.plans:
            assert plan.n_transfers <= 2
            assert len(plan.squad) == 15
            assert validate_squad(plan.squad, {pid: pid for pid in POSITION_OF}) == ()

    def test_out_and_in_partition_the_change(self):
        plan = _plan(max_transfers=2).plans[0]
        owned = frozenset(OWNED)
        new_ids = frozenset(p.player_id for p in plan.squad)
        assert frozenset(plan.out_player_ids) == owned - new_ids
        assert frozenset(plan.in_player_ids) == new_ids - owned
        assert len(plan.out_player_ids) == len(plan.in_player_ids) == plan.n_transfers

    def test_never_suggests_the_current_squad(self):
        for plan in _plan(max_transfers=2).plans:
            assert plan.n_transfers >= 1

    def test_gains_expected_points(self):
        """The owned 15 is the weakest legal 15 in the pool, so every suggestion must improve on
        it. A non-positive delta here means the solver, the scoring, or the sign convention is
        wrong, and all three are easy to get backwards."""
        plan = _plan(max_transfers=1).plans[0]
        assert plan.expected_points_delta > 0

    def test_expected_swing_delta_equals_expected_points_delta(self):
        """The module docstring's central claim, asserted rather than merely argued: the field's
        effective ownership cancels out of a transfer, so a plan's gap gain against the league is
        its points gain. Measured over a single gameweek with no chip, which is the setting the
        algebra is stated for."""
        plan = _plan(max_transfers=2).plans[0]
        assert plan.expected_gap_delta == pytest.approx(plan.expected_points_delta, abs=1e-9)

    def test_ranks_by_expected_final_rank_then_points(self):
        """The ordering contract in full: expected final rank ascending once rounded to
        RANK_TOLERANCE, and expected points descending inside any one of those buckets. Asserting
        the raw rank were monotone instead would be asserting the opposite of what the tolerance
        is for."""
        plans = _plan(max_transfers=2).plans
        buckets = [round(plan.expected_final_rank / RANK_TOLERANCE) for plan in plans]
        assert buckets == sorted(buckets)
        for bucket in set(buckets):
            deltas = [
                plan.expected_points_delta
                for plan, plan_bucket in zip(plans, buckets, strict=True)
                if plan_bucket == bucket
            ]
            assert deltas == sorted(deltas, reverse=True)

    def test_league_position_changes_the_suggestion(self):
        """The whole premise. Identical projections, identical pool, identical budget: only the
        manager's points relative to the field differ, which flips whether extra variance helps.
        If these two agree on everything, the mini-league is decorative."""
        behind = _plan(my_total_points=10)
        ahead = _plan(my_total_points=400)
        assert behind.variance_preference == "increase"
        assert ahead.variance_preference == "decrease"

    def test_a_saturated_rank_falls_back_to_points(self):
        """A manager far clear of the field has an expected final rank pinned at 1.0 whatever they
        do, so rank differences shrink to noise far below the hundredth of a place the banner
        displays. RANK_TOLERANCE exists so that noise cannot outrank a points gain the manager can
        actually see, which is what this asserts: with the rank level, plans come back best points
        first."""
        suggestion = _plan(max_transfers=2, my_total_points=100_000)
        ranks = [plan.expected_final_rank for plan in suggestion.plans]
        assert all(rank == pytest.approx(1.0, abs=RANK_TOLERANCE) for rank in ranks)
        deltas = [plan.expected_points_delta for plan in suggestion.plans]
        assert deltas == sorted(deltas, reverse=True)

    def test_expected_final_rank_is_bounded_by_the_field_size(self):
        suggestion = _plan(n_rivals=4)
        assert 1.0 <= suggestion.current_expected_final_rank <= 5.0
        for plan in suggestion.plans:
            assert 1.0 <= plan.expected_final_rank <= 5.0

    def test_marginal_transfer_never_loses_points(self):
        """Each additional transfer is a strictly looser constraint on the same program, so the
        best plan at n+1 transfers can never score below the best at n."""
        suggestion = _plan(max_transfers=3)
        deltas = [plan.expected_points_delta for plan in suggestion.best_by_transfer_count]
        assert deltas == sorted(deltas)

    def test_budget_is_respected(self):
        horizon, gameweek = _projections()
        rivals, ownership = _league()
        # The owned 15 costs exactly 15 * 45 = 675, so a budget of 675 leaves nothing spare and
        # every price here is identical, meaning any swap is still affordable but nothing above it
        # would be.
        suggestion = plan_transfers(
            _team_state(),
            _candidates(),
            horizon,
            [GAMEWEEK],
            gameweek,
            ownership,
            rivals,
            league_gameweek=GAMEWEEK,
            my_total_points=100,
            gameweeks_remaining=10,
            budget=675,
            max_transfers=2,
        )
        for plan in suggestion.plans:
            assert sum(player.price for player in plan.squad) <= 675
            assert plan.budget_remaining >= 0

    def test_falls_back_to_points_with_no_league(self):
        """No rivals means every plan carries the same expected final rank of 1.0, so the tie
        break becomes the whole ranking and the banner degrades to a plain points suggestion."""
        horizon, gameweek = _projections()
        suggestion = plan_transfers(
            _team_state(),
            _candidates(),
            horizon,
            [GAMEWEEK],
            gameweek,
            {},
            (),
            league_gameweek=GAMEWEEK,
            budget=1000,
            max_transfers=2,
        )
        assert suggestion.n_rivals == 0
        assert suggestion.variance_preference == "neutral"
        assert suggestion.plans
        assert all(plan.expected_final_rank == 1.0 for plan in suggestion.plans)
        deltas = [plan.expected_points_delta for plan in suggestion.plans]
        assert deltas == sorted(deltas, reverse=True)

    def test_rejects_a_transfer_count_below_one(self):
        with pytest.raises(ValueError, match="at least 1"):
            _plan(max_transfers=0)

    def test_plans_are_distinct(self):
        squads = [
            frozenset(p.player_id for p in plan.squad) for plan in _plan(max_transfers=2).plans
        ]
        assert len(squads) == len(set(squads))
