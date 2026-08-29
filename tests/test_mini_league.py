"""Tests for features.mini_league (MINI_LEAGUE_PLAN Phase 3) -- pure math over hand-built
LeagueSnapshot/projection fixtures, no I/O. Sign conventions get the most scrutiny here (matching
this module's own docstring warning that exposure/gap signs are the easiest thing to get backwards),
plus the M12 posture derivative claim, which is the one piece of math this whole feature leans on.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.aggregate import ComponentBreakdown
from engine.data.league_state_builder import ChipUsage, LeagueEntry, LeagueSnapshot
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection, project_player_gameweek
from engine.scoring import DEF, FWD, GK, MID
from engine.simulate import PlayerSimulationSummary
from features.mini_league import (
    KNOWN_CHIPS,
    HeadToHead,
    compute_chip_states,
    compute_coverage,
    compute_exposures,
    compute_head_to_head,
    compute_league_ownership,
    compute_posture,
    league_template_xi,
    prospective_swing,
    rank_captain_options,
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


def _team_state(**overrides) -> MyTeamState:
    defaults = dict(
        squad=_squad(),
        starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
        bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
    )
    defaults.update(overrides)
    return MyTeamState(**defaults)


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


def _projection(
    player_id: int, points: float, position: str = MID, std: float | None = None
) -> PlayerGameweekProjection:
    simulation = _simulation(player_id, std) if std is not None else None
    return project_player_gameweek(
        player_id, position, 1, _minutes(), _breakdown(points), simulation=simulation
    )


def _rival(
    entry_id: int, picks: dict[int, int], total_points: int = 0, chips: tuple = ()
) -> LeagueEntry:
    return LeagueEntry(
        entry_id=entry_id,
        manager_name=f"Manager {entry_id}",
        team_name=f"Team {entry_id}",
        rank=entry_id,
        total_points=total_points,
        gameweek_points=0,
        picks=picks,
        chips=chips,
    )


def _snapshot(entries: list[LeagueEntry], picks_gameweek: int = 7) -> LeagueSnapshot:
    return LeagueSnapshot(
        league_id=999,
        league_name="Test League",
        picks_gameweek=picks_gameweek,
        entries=tuple(entries),
    )


class TestComputeLeagueOwnership:
    def test_excludes_own_entry_from_the_field(self):
        snapshot = _snapshot(
            [
                _rival(1, {10: 1}),
                _rival(2, {10: 1}),
            ]
        )
        ownership = compute_league_ownership(snapshot, exclude_entry_id=1)
        # Only rival 2 remains -> eo_multiplier for player 10 is 1.0 / 1 rival, not averaged in
        # with entry 1's own pick.
        assert ownership[10].eo_multiplier == pytest.approx(1.0)
        assert ownership[10].owner_names == ("Manager 2",)

    def test_eo_multiplier_is_the_mean_multiplier_across_rivals(self):
        snapshot = _snapshot(
            [
                _rival(1, {10: 2}),  # captained
                _rival(2, {10: 1}),  # started
                _rival(3, {}),  # doesn't own at all
            ]
        )
        ownership = compute_league_ownership(snapshot, exclude_entry_id=None)
        assert ownership[10].eo_multiplier == pytest.approx((2 + 1 + 0) / 3)
        assert ownership[10].eo_percent == pytest.approx(100.0)

    def test_raw_ownership_counts_presence_regardless_of_multiplier(self):
        snapshot = _snapshot(
            [
                _rival(1, {10: 2}),
                _rival(2, {10: 1}),
                _rival(3, {}),
            ]
        )
        ownership = compute_league_ownership(snapshot)
        assert ownership[10].raw_ownership_percent == pytest.approx(200.0 / 3.0)

    def test_owner_count_is_the_exact_integer_the_percentage_is_computed_from(self):
        snapshot = _snapshot(
            [
                _rival(1, {10: 2}),
                _rival(2, {10: 1}),
                _rival(3, {}),
            ]
        )
        ownership = compute_league_ownership(snapshot)
        assert ownership[10].owner_count == 2

    def test_captain_share_counts_multiplier_of_at_least_two(self):
        snapshot = _snapshot(
            [
                _rival(1, {10: 2}),  # captain
                _rival(2, {10: 3}),  # triple captain
                _rival(3, {10: 1}),  # merely started
            ]
        )
        ownership = compute_league_ownership(snapshot)
        assert ownership[10].captain_share_percent == pytest.approx(200.0 / 3.0)

    def test_owner_names_lists_only_managers_who_own_the_player(self):
        snapshot = _snapshot([_rival(1, {10: 1}), _rival(2, {20: 1})])
        ownership = compute_league_ownership(snapshot)
        assert ownership[10].owner_names == ("Manager 1",)

    def test_empty_when_there_are_no_rivals(self):
        """A league of one, or your own entry excluded from a single-entry snapshot -- there is no
        field to measure ownership against."""
        snapshot = _snapshot([_rival(1, {10: 1})])
        assert compute_league_ownership(snapshot, exclude_entry_id=1) == {}


class TestComputeExposures:
    def test_captain_exposure_is_your_multiplier_minus_eo_times_xp(self):
        state = _team_state(captain_id=MID_IDS[0])
        ownership = compute_league_ownership(
            _snapshot([_rival(1, {MID_IDS[0]: 0}), _rival(2, {MID_IDS[0]: 1})])
        )
        projections = {MID_IDS[0]: _projection(MID_IDS[0], points=6.0)}

        [exposure] = compute_exposures([MID_IDS[0]], state, ownership, projections)

        assert exposure.your_multiplier == pytest.approx(2.0)
        assert exposure.ownership.eo_multiplier == pytest.approx(0.5)
        assert exposure.exposure == pytest.approx(1.5)
        assert exposure.expected_swing == pytest.approx(1.5 * 6.0)

    def test_a_player_nobody_owns_defaults_to_zero_ownership(self):
        state = _team_state()
        [exposure] = compute_exposures([FWD_IDS[0]], state, {}, {})
        assert exposure.ownership.eo_multiplier == 0.0
        assert exposure.ownership.raw_ownership_percent == 0.0

    def test_missing_projection_yields_none_expected_swing_not_a_fabricated_zero(self):
        state = _team_state()
        [exposure] = compute_exposures([FWD_IDS[0]], state, {}, {})
        assert exposure.expected_points is None
        assert exposure.expected_swing is None
        # exposure itself is still a real number -- ownership doesn't depend on projections.
        assert exposure.exposure == pytest.approx(1.0)  # you start him, league owns him at 0.0

    def test_bench_player_exposure_is_zero_against_zero_ownership(self):
        state = _team_state()
        bench_id = DEF_IDS[4]
        projections = {bench_id: _projection(bench_id, points=3.0, position=DEF)}
        [exposure] = compute_exposures([bench_id], state, {}, projections)
        assert exposure.your_multiplier == 0.0
        assert exposure.exposure == 0.0
        assert exposure.expected_swing == 0.0

    def test_bench_boost_chip_raises_bench_multiplier_to_one(self):
        state = _team_state()
        bench_id = DEF_IDS[4]
        [exposure] = compute_exposures([bench_id], state, {}, {}, chip="bench_boost")
        assert exposure.your_multiplier == 1.0

    def test_unknown_chip_raises(self):
        state = _team_state()
        with pytest.raises(ValueError):
            compute_exposures([FWD_IDS[0]], state, {}, {}, chip="not_a_real_chip")


class TestProspectiveSwing:
    def test_matches_the_one_minus_eo_formula(self):
        assert prospective_swing(eo_multiplier=0.2, expected_points=5.0) == pytest.approx(4.0)

    def test_zero_when_fully_captained_by_the_league(self):
        # A hypothetical eo_multiplier of 1.0 (everyone starts him, nobody captains or benches)
        # means buying him in nets you exactly the field -- zero prospective swing.
        assert prospective_swing(eo_multiplier=1.0, expected_points=8.0) == pytest.approx(0.0)


def _mirrored_picks(state: MyTeamState) -> dict[int, int]:
    """A rival who owns exactly your squad at exactly your own multipliers -- the neutral baseline
    every head-to-head test below starts from and then perturbs for the one differential it's
    actually testing. Without this, a sparse ``{player_id: multiplier}`` dict implicitly means "the
    rival owns none of your other 14 players," which turns every one of them into a spurious extra
    differential (your multiplier vs. the implied 0) and breaks any test that expects exactly one.
    """
    picks = {
        player_id: 2 if player_id == state.captain_id else 1 for player_id in state.starting_xi
    }
    picks.update(dict.fromkeys(state.bench_order, 0))
    return picks


class TestComputeHeadToHead:
    def test_shared_picks_are_counted_but_excluded_from_differentials(self):
        state = _team_state()
        rival = _rival(1, _mirrored_picks(state))

        result = compute_head_to_head(state, rival, projections={})

        assert result.differentials == ()
        assert result.shared_count == len(state.player_ids)
        assert result.expected_gap == 0.0

    def test_a_captain_you_hold_that_the_rival_merely_starts_is_a_positive_contribution(self):
        state = _team_state(captain_id=MID_IDS[0])
        picks = _mirrored_picks(state)
        picks[MID_IDS[0]] = 1  # the rival started him instead of captaining
        rival = _rival(1, picks)
        projections = {MID_IDS[0]: _projection(MID_IDS[0], points=6.0)}

        result = compute_head_to_head(state, rival, projections)

        [pick] = result.differentials
        assert pick.player_id == MID_IDS[0]
        assert pick.your_multiplier == 2.0
        assert pick.rival_multiplier == 1.0
        assert pick.expected_gap_contribution == pytest.approx((2.0 - 1.0) * 6.0)
        assert result.expected_gap == pytest.approx(6.0)

    def test_a_player_only_the_rival_owns_is_a_negative_contribution(self):
        state = _team_state()
        picks = _mirrored_picks(state)
        rival_only_id = 999
        picks[rival_only_id] = 1
        rival = _rival(1, picks)
        projections = {rival_only_id: _projection(rival_only_id, points=5.0, position=FWD)}

        result = compute_head_to_head(state, rival, projections)

        [pick] = result.differentials
        assert pick.player_id == rival_only_id
        assert pick.your_multiplier == 0.0
        assert pick.rival_multiplier == 1.0
        assert pick.expected_gap_contribution == pytest.approx(-5.0)

    def test_gap_std_is_the_sqrt_of_summed_variance_over_differentials(self):
        state = _team_state(captain_id=MID_IDS[0])
        second_id = FWD_IDS[0]  # part of the default starting XI, at multiplier 1.0
        picks = _mirrored_picks(state)
        picks[MID_IDS[0]] = 1  # rival started instead of captained -> diff 1.0
        picks[second_id] = 0  # rival dropped him entirely -> diff 1.0
        rival = _rival(1, picks)
        projections = {
            MID_IDS[0]: _projection(MID_IDS[0], points=6.0, std=2.0),
            second_id: _projection(second_id, points=4.0, position=FWD, std=3.0),
        }

        result = compute_head_to_head(state, rival, projections)

        assert len(result.differentials) == 2
        expected_variance = (1.0**2) * (2.0**2) + (1.0**2) * (3.0**2)
        assert result.gap_std == pytest.approx(math.sqrt(expected_variance))

    def test_p_outscore_is_a_half_when_the_gap_and_std_are_both_zero(self):
        state = _team_state()
        rival = _rival(1, _mirrored_picks(state))

        result = compute_head_to_head(state, rival, projections={})

        assert result.gap_std == 0.0
        assert result.expected_gap == 0.0
        assert result.p_outscore == pytest.approx(0.5)

    def test_p_outscore_is_one_when_gap_is_positive_with_zero_variance(self):
        state = _team_state(captain_id=MID_IDS[0])
        picks = _mirrored_picks(state)
        picks[MID_IDS[0]] = 1  # rival started him instead of captaining
        rival = _rival(1, picks)
        # std defaults to None -> 0.0 inside compute_head_to_head, i.e. deterministic.
        projections = {MID_IDS[0]: _projection(MID_IDS[0], points=6.0)}

        result = compute_head_to_head(state, rival, projections)

        assert result.p_outscore == pytest.approx(1.0)

    def test_missing_projection_contributes_zero_but_the_pick_is_still_shown(self):
        state = _team_state(captain_id=MID_IDS[0])
        picks = _mirrored_picks(state)
        picks[MID_IDS[0]] = 1  # rival started him instead of captaining
        rival = _rival(1, picks)

        result = compute_head_to_head(state, rival, projections={})

        [pick] = result.differentials
        assert pick.expected_points is None
        assert pick.expected_gap_contribution == 0.0
        assert result.expected_gap == 0.0


class TestRankCaptainOptions:
    def test_a_heavily_captained_high_xp_player_can_net_less_than_a_lightly_owned_lower_xp_one(
        self,
    ):
        """The exact inversion MINI_LEAGUE_PLAN M10 is built around."""
        low_eo_id, high_eo_id = 41, 42
        ownership = {
            low_eo_id: compute_league_ownership(_snapshot([_rival(1, {low_eo_id: 0})]))[low_eo_id],
            high_eo_id: compute_league_ownership(_snapshot([_rival(1, {high_eo_id: 3})]))[
                high_eo_id
            ],
        }
        projections = {
            low_eo_id: _projection(low_eo_id, points=6.0),
            high_eo_id: _projection(high_eo_id, points=7.4),
        }

        options = {
            option.player_id: option
            for option in rank_captain_options([low_eo_id, high_eo_id], ownership, projections)
        }

        assert options[high_eo_id].eo_multiplier == pytest.approx(3.0)
        assert options[high_eo_id].net_captain_ev == pytest.approx((2.0 - 3.0) * 7.4)
        assert options[low_eo_id].net_captain_ev == pytest.approx((2.0 - 0.0) * 6.0)
        assert options[low_eo_id].net_captain_ev > options[high_eo_id].net_captain_ev

    def test_missing_projection_yields_none_net_ev(self):
        [option] = rank_captain_options([41], {}, {})
        assert option.expected_points is None
        assert option.net_captain_ev is None
        assert option.net_captain_std is None


class TestComputeChipStates:
    def test_used_and_remaining_partition_the_known_roster(self):
        rival = _rival(1, {}, chips=(ChipUsage("bboost", 5), ChipUsage("wildcard", 2)))
        [state] = compute_chip_states([rival])
        assert set(state.used_chip_names) == {"bboost", "wildcard"}
        assert set(state.remaining_chip_names) == set(KNOWN_CHIPS) - {"bboost", "wildcard"}

    def test_unrecognised_chip_name_is_kept_in_used_but_never_in_remaining(self):
        rival = _rival(1, {}, chips=(ChipUsage("some_future_chip", 10),))
        [state] = compute_chip_states([rival])
        assert "some_future_chip" in state.used_chip_names
        assert set(state.remaining_chip_names) == set(KNOWN_CHIPS)


class TestComputePosture:
    def test_projected_behind_prefers_increasing_variance(self):
        rival = _rival(1, {}, total_points=150)
        head_to_head = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=1.0,
            gap_std=5.0,
            p_outscore=0.5,
        )
        posture = compute_posture(
            your_total_points=100, rival=rival, head_to_head=head_to_head, gameweeks_remaining=10
        )
        assert posture.projected_final_gap == pytest.approx(-50 + 10 * 1.0)
        assert posture.projected_final_gap < 0
        assert posture.variance_preference == "increase"

    def test_projected_ahead_prefers_decreasing_variance(self):
        rival = _rival(1, {}, total_points=50)
        head_to_head = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=0.0,
            gap_std=5.0,
            p_outscore=0.5,
        )
        posture = compute_posture(
            your_total_points=100, rival=rival, head_to_head=head_to_head, gameweeks_remaining=10
        )
        assert posture.projected_final_gap > 0
        assert posture.variance_preference == "decrease"

    def test_exactly_zero_projected_gap_is_neutral(self):
        rival = _rival(1, {}, total_points=100)
        head_to_head = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=0.0,
            gap_std=5.0,
            p_outscore=0.5,
        )
        posture = compute_posture(
            your_total_points=100, rival=rival, head_to_head=head_to_head, gameweeks_remaining=10
        )
        assert posture.projected_final_gap == 0.0
        assert posture.variance_preference == "neutral"
        assert posture.p_finish_ahead == pytest.approx(0.5)

    def test_derivative_direction_variance_helps_when_projected_behind(self):
        """The core M12 claim: when projected to finish behind, MORE gap_std should INCREASE
        p_finish_ahead, not decrease it."""
        rival = _rival(1, {}, total_points=150)
        low_std = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=1.0,
            gap_std=2.0,
            p_outscore=0.5,
        )
        high_std = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=1.0,
            gap_std=8.0,
            p_outscore=0.5,
        )
        posture_low = compute_posture(100, rival, low_std, gameweeks_remaining=10)
        posture_high = compute_posture(100, rival, high_std, gameweeks_remaining=10)

        assert posture_low.projected_final_gap < 0  # both are the same "projected behind" case
        assert posture_high.p_finish_ahead > posture_low.p_finish_ahead

    def test_derivative_direction_variance_hurts_when_projected_ahead(self):
        rival = _rival(1, {}, total_points=50)
        low_std = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=1.0,
            gap_std=2.0,
            p_outscore=0.5,
        )
        high_std = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=1.0,
            gap_std=8.0,
            p_outscore=0.5,
        )
        posture_low = compute_posture(100, rival, low_std, gameweeks_remaining=10)
        posture_high = compute_posture(100, rival, high_std, gameweeks_remaining=10)

        assert posture_low.projected_final_gap > 0  # both are the same "projected ahead" case
        assert posture_high.p_finish_ahead < posture_low.p_finish_ahead

    def test_zero_gap_std_falls_back_to_a_step_function(self):
        rival = _rival(1, {}, total_points=50)
        head_to_head = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=0.0,
            gap_std=0.0,
            p_outscore=0.5,
        )
        posture = compute_posture(100, rival, head_to_head, gameweeks_remaining=10)
        assert posture.p_finish_ahead == pytest.approx(1.0)  # you're already ahead, zero variance

    def test_zero_gameweeks_remaining_ignores_gap_std_entirely(self):
        rival = _rival(1, {}, total_points=50)
        head_to_head = HeadToHead(
            rival_entry_id=1,
            shared_count=0,
            differentials=(),
            expected_gap=100.0,
            gap_std=5.0,
            p_outscore=0.5,
        )
        posture = compute_posture(100, rival, head_to_head, gameweeks_remaining=0)
        # projected_final_gap ignores expected_gap entirely once no gameweeks remain.
        assert posture.projected_final_gap == pytest.approx(50.0)
        assert posture.p_finish_ahead == pytest.approx(1.0)


class TestLeagueTemplateXi:
    def test_returns_the_n_highest_eo_players(self):
        snapshot = _snapshot(
            [
                _rival(1, {10: 2, 20: 1, 30: 0}),
                _rival(2, {10: 1, 20: 1, 30: 1}),
            ]
        )
        ownership = compute_league_ownership(snapshot)
        assert league_template_xi(ownership, n=2) == (10, 20)

    def test_ties_are_broken_by_player_id_for_determinism(self):
        snapshot = _snapshot([_rival(1, {20: 1, 10: 1})])
        ownership = compute_league_ownership(snapshot)
        assert league_template_xi(ownership, n=1) == (10,)


class TestComputeCoverage:
    def test_zero_when_the_league_has_no_ownership_data(self):
        state = _team_state()
        assert compute_coverage(state, {}) == 0.0

    def test_full_coverage_when_your_multipliers_exactly_match_the_league(self):
        snapshot = _snapshot([_rival(1, {pid: 1 for pid in _team_state().starting_xi})])
        ownership = compute_league_ownership(snapshot)
        state = _team_state()
        assert compute_coverage(state, ownership) == pytest.approx(1.0)

    def test_partial_coverage_when_you_bench_a_heavily_owned_player(self):
        heavily_owned_id = DEF_IDS[4]  # benched in the default _team_state()
        snapshot = _snapshot([_rival(1, {heavily_owned_id: 1})])
        ownership = compute_league_ownership(snapshot)
        state = _team_state()
        # numerator = min(your=0.0, eo=1.0) = 0.0; denominator = eo = 1.0 -> coverage 0.0
        assert compute_coverage(state, ownership) == pytest.approx(0.0)
