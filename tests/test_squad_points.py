"""Tests for features.squad_points -- the one function behind the header total, pitch-card
per-gameweek values, and every chip preview (D18)."""

from __future__ import annotations

import pytest

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID
from features.squad_points import (
    CHIP_BENCH_BOOST,
    CHIP_TRIPLE_CAPTAIN,
    projected_points,
)
from features.team_state import MyTeamState, SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


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


def _horizon(player_id: int, position: str, points_by_gw: dict[int, float]):
    return project_player_horizon(
        player_id,
        position,
        {
            gw: project_player_gameweek(player_id, position, gw, _minutes(), _breakdown(pts))
            for gw, pts in points_by_gw.items()
        },
    )


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


def _state(**overrides) -> MyTeamState:
    defaults = dict(
        squad=_squad(),
        starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
        bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
        bank=0,
        free_transfers=1,
        chips_remaining=frozenset({"wildcard", "free_hit", "bench_boost", "triple_captain"}),
    )
    defaults.update(overrides)
    return MyTeamState(**defaults)


def _flat_projections(gameweeks: list[int], points: float = 4.0) -> dict:
    """Every squad player scores `points` in every one of `gameweeks`."""
    positions = {GK1: GK, GK2: GK}
    positions.update({pid: DEF for pid in DEF_IDS})
    positions.update({pid: MID for pid in MID_IDS})
    positions.update({pid: FWD for pid in FWD_IDS})
    return {
        pid: _horizon(pid, pos, dict.fromkeys(gameweeks, points)) for pid, pos in positions.items()
    }


class TestBaselineScoring:
    def test_total_is_xi_sum_plus_captain_doubling(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1])
        # 11 XI players x 4.0, captain gets one extra 4.0 -> 11*4 + 4 = 48
        assert result.total == pytest.approx(48.0)
        assert result.starting_xi_points == pytest.approx(48.0)
        assert result.bench_points == 0.0

    def test_captain_bonus_is_the_extra_points_from_the_multiplier(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1])
        assert result.captain_bonus == pytest.approx(4.0)

    def test_bench_never_counts_without_bench_boost(self):
        state = _state()
        projections = _flat_projections([1], points=100.0)  # huge bench score, should be ignored
        result = projected_points(state, projections, [1])
        assert result.bench_points == 0.0
        for pid in state.bench_order:
            assert pid not in result.per_player

    def test_per_player_reflects_captain_multiplier_only_for_the_captain(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1])
        assert result.per_player[state.captain_id] == pytest.approx(8.0)
        other_xi = next(pid for pid in state.starting_xi if pid != state.captain_id)
        assert result.per_player[other_xi] == pytest.approx(4.0)


class TestBenchBoost:
    def test_adds_exactly_the_four_bench_players(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        baseline = projected_points(state, projections, [1])
        boosted = projected_points(state, projections, [1], chip=CHIP_BENCH_BOOST)
        assert boosted.total == pytest.approx(baseline.total + 4 * 4.0)
        assert boosted.bench_points == pytest.approx(16.0)

    def test_bench_players_appear_in_per_player(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1], chip=CHIP_BENCH_BOOST)
        for pid in state.bench_order:
            assert result.per_player[pid] == pytest.approx(4.0)

    def test_bench_boost_does_not_affect_captain_multiplier(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1], chip=CHIP_BENCH_BOOST)
        assert result.captain_bonus == pytest.approx(4.0)  # still just x2, not x3


class TestTripleCaptain:
    def test_captain_scores_triple_not_double(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1], chip=CHIP_TRIPLE_CAPTAIN)
        assert result.per_player[state.captain_id] == pytest.approx(12.0)
        assert result.captain_bonus == pytest.approx(8.0)  # 2 extra x's worth

    def test_total_reflects_triple_captain(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        baseline = projected_points(state, projections, [1])
        tripled = projected_points(state, projections, [1], chip=CHIP_TRIPLE_CAPTAIN)
        assert tripled.total == pytest.approx(baseline.total + 4.0)  # one extra captain multiple

    def test_triple_captain_does_not_add_bench(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        result = projected_points(state, projections, [1], chip=CHIP_TRIPLE_CAPTAIN)
        assert result.bench_points == 0.0


class TestWildcardAndFreeHitHaveNoScoringEffect:
    def test_wildcard_scores_identically_to_no_chip(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        baseline = projected_points(state, projections, [1])
        with_chip = projected_points(state, projections, [1], chip="wildcard")
        assert with_chip.total == pytest.approx(baseline.total)

    def test_free_hit_scores_identically_to_no_chip(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        baseline = projected_points(state, projections, [1])
        with_chip = projected_points(state, projections, [1], chip="free_hit")
        assert with_chip.total == pytest.approx(baseline.total)


class TestMultiGameweek:
    def test_sums_across_gameweeks_per_player(self):
        state = _state()
        projections = _flat_projections([1, 2, 3], points=4.0)
        result = projected_points(state, projections, [1, 2, 3])
        other_xi = next(pid for pid in state.starting_xi if pid != state.captain_id)
        assert result.per_player[other_xi] == pytest.approx(12.0)  # 4.0 x 3 gameweeks

    def test_per_gameweek_values_are_not_summed_together(self):
        state = _state()
        projections = {
            pid: _horizon(
                pid,
                (
                    "MID"
                    if pid in MID_IDS
                    else ("GK" if pid in (GK1, GK2) else ("DEF" if pid in DEF_IDS else "FWD"))
                ),
                {1: 3.0, 2: 5.0},
            )
            for pid in state.player_ids
        }
        result = projected_points(state, projections, [1, 2])
        assert result.per_gameweek[1] != result.per_gameweek[2]
        assert result.per_gameweek[1] + result.per_gameweek[2] == pytest.approx(result.total)


class TestMissingProjections:
    def test_blank_gameweek_reported_not_counted_as_zero(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)  # nobody projected for GW2
        result = projected_points(state, projections, [1, 2])
        assert set(result.missing_player_ids) == set(state.starting_xi)

    def test_missing_player_does_not_crash(self):
        state = _state()
        projections = {}  # nobody projected at all
        result = projected_points(state, projections, [1])
        assert result.total == 0.0
        assert set(result.missing_player_ids) == set(state.starting_xi)

    def test_partial_coverage_only_flags_the_gap(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        # Remove one starting player's projection entirely.
        missing_pid = next(pid for pid in state.starting_xi if pid != state.captain_id)
        del projections[missing_pid]
        result = projected_points(state, projections, [1])
        assert missing_pid in result.missing_player_ids
        assert state.captain_id not in result.missing_player_ids


class TestInvalidChip:
    def test_unknown_chip_raises(self):
        state = _state()
        projections = _flat_projections([1], points=4.0)
        with pytest.raises(ValueError, match="unknown chip"):
            projected_points(state, projections, [1], chip="not_a_real_chip")
