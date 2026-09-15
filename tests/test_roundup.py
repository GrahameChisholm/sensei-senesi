"""Tests for features.roundup -- pure math over hand-built LeagueEntry fixtures, no I/O, mirroring
tests/test_mini_league.py's own style and conventions.
"""

from __future__ import annotations

from engine.data.league_state_builder import ChipUsage, GameweekHistoryRow, LeagueEntry
from features.roundup import (
    bench_regret,
    biggest_differential_haul,
    biggest_movers,
    captain_scoreboard,
    chips_played,
    highest_scoring_owned_player,
    standings_by_gameweek,
    template_overlap,
    top_and_bottom,
)


def _history(*points: int) -> tuple[GameweekHistoryRow, ...]:
    """One row per gameweek, points and cumulative totals both derived from the given per-gameweek
    scores -- the exact shape ``get_entry_history``'s ``current`` list carries."""
    rows = []
    running = 0
    for event, score in enumerate(points, start=1):
        running += score
        rows.append(GameweekHistoryRow(event=event, points=score, total_points=running))
    return tuple(rows)


def _rival(
    entry_id: int,
    picks: dict[int, int] | None = None,
    pick_positions: dict[int, int] | None = None,
    history: tuple[GameweekHistoryRow, ...] = (),
    chips: tuple[ChipUsage, ...] = (),
    manager_name: str | None = None,
) -> LeagueEntry:
    return LeagueEntry(
        entry_id=entry_id,
        manager_name=manager_name or f"Manager {entry_id}",
        team_name=f"Team {entry_id}",
        rank=entry_id,
        total_points=0,
        gameweek_points=0,
        picks=picks or {},
        pick_positions=pick_positions or {},
        chips=chips,
        history=history,
    )


class TestStandingsByGameweek:
    def test_ranks_by_cumulative_total_points_per_gameweek(self):
        entries = [
            _rival(1, history=_history(50, 50)),  # 50, then 100
            _rival(2, history=_history(60, 40)),  # 60, then 100
        ]
        standings = standings_by_gameweek(entries, up_to_gameweek=2)
        assert standings[1] == (2, 1)  # 60 > 50
        assert standings[2] == (1, 2)  # tied at 100 -> lower entry_id wins the tiebreak

    def test_an_entry_missing_a_history_row_is_excluded_from_that_gameweek(self):
        entries = [_rival(1, history=_history(50, 50)), _rival(2, history=_history(60))]
        standings = standings_by_gameweek(entries, up_to_gameweek=2)
        assert standings[1] == (2, 1)
        assert standings[2] == (1,)


class TestTopAndBottom:
    def test_splits_top_and_bottom_n_by_gameweek_points(self):
        entries = [
            _rival(1, history=_history(90)),
            _rival(2, history=_history(80)),
            _rival(3, history=_history(70)),
            _rival(4, history=_history(60)),
            _rival(5, history=_history(50)),
        ]
        top, bottom = top_and_bottom(entries, gameweek=1, n=2)
        assert [row.entry_id for row in top] == [1, 2]
        assert [row.entry_id for row in bottom] == [5, 4]

    def test_ties_at_the_cutoff_are_all_included(self):
        entries = [
            _rival(1, history=_history(90)),
            _rival(2, history=_history(80)),
            _rival(3, history=_history(80)),
        ]
        top, _ = top_and_bottom(entries, gameweek=1, n=2)
        assert {row.entry_id for row in top} == {1, 2, 3}

    def test_entries_missing_this_gameweeks_history_row_are_excluded(self):
        entries = [_rival(1, history=_history(90)), _rival(2, history=())]
        top, bottom = top_and_bottom(entries, gameweek=1)
        assert [row.entry_id for row in top] == [1]
        assert [row.entry_id for row in bottom] == [1]


class TestHighestScoringOwnedPlayer:
    def test_picks_the_max_scoring_owned_player_with_owner_breakdown(self):
        entries = [
            _rival(1, picks={10: 1, 20: 2}, pick_positions={10: 1, 20: 2}),
            _rival(2, picks={10: 0}, pick_positions={10: 12}),
        ]
        live_points = {10: 5, 20: 15}
        row = highest_scoring_owned_player(entries, live_points)
        assert row.player_id == 20
        assert row.live_points == 15
        assert row.owner_entry_ids == (1,)
        assert row.starter_entry_ids == (1,)
        assert row.captain_entry_ids == (1,)

    def test_bench_owner_is_not_a_starter_or_captain(self):
        entries = [_rival(1, picks={10: 0}, pick_positions={10: 15})]
        row = highest_scoring_owned_player(entries, {10: 8})
        assert row.owner_entry_ids == (1,)
        assert row.starter_entry_ids == ()
        assert row.captain_entry_ids == ()

    def test_none_when_nobody_owns_anything(self):
        assert highest_scoring_owned_player([_rival(1)], {}) is None


class TestCaptainScoreboard:
    def test_captain_points_are_live_points_times_multiplier(self):
        entries = [_rival(1, picks={10: 2, 11: 1})]
        [row] = captain_scoreboard(entries, {10: 6})
        assert row.captain_player_id == 10
        assert row.multiplier == 2
        assert row.points == 12

    def test_triple_captain_multiplies_by_three(self):
        entries = [_rival(1, picks={10: 3})]
        [row] = captain_scoreboard(entries, {10: 6})
        assert row.multiplier == 3
        assert row.points == 18

    def test_no_captain_pick_yields_zero_not_a_crash(self):
        entries = [_rival(1, picks={10: 1})]
        [row] = captain_scoreboard(entries, {10: 6})
        assert row.captain_player_id is None
        assert row.points == 0


class TestBenchRegret:
    def test_sums_points_from_zero_multiplier_picks_only(self):
        entries = [_rival(1, picks={10: 1, 11: 0, 12: 0})]
        [row] = bench_regret(entries, {10: 5, 11: 8, 12: 0})
        assert row.points_left_on_bench == 8
        assert row.contributing_player_ids == (11,)

    def test_bench_boost_week_has_no_bench_multiplier_and_so_no_regret(self):
        """M3: multiplier already reflects the chip, so a Bench Boost gameweek's picks all carry
        multiplier >= 1 and this reports zero, correctly -- nothing was actually left unscored."""
        entries = [_rival(1, picks={10: 1, 11: 1})]
        [row] = bench_regret(entries, {10: 5, 11: 8})
        assert row.points_left_on_bench == 0
        assert row.contributing_player_ids == ()


class TestBiggestMovers:
    def test_empty_at_gameweek_one(self):
        entries = [_rival(1, history=_history(50))]
        standings = standings_by_gameweek(entries, up_to_gameweek=1)
        assert biggest_movers(entries, standings, gameweek=1) == ()

    def test_climb_and_fall_are_computed_from_rank_delta(self):
        entries = [
            _rival(1, history=_history(50, 40)),  # GW1 rank 2 (50<60), GW2 total 90
            _rival(2, history=_history(60, 20)),  # GW1 rank 1 (60), GW2 total 80
        ]
        standings = standings_by_gameweek(entries, up_to_gameweek=2)
        # GW1: entry2 rank1, entry1 rank2. GW2 totals: entry1=90, entry2=80 -> ranks flip.
        movers = biggest_movers(entries, standings, gameweek=2)
        by_id = {mover.entry_id: mover for mover in movers}
        assert by_id[1].rank_before == 2 and by_id[1].rank_after == 1 and by_id[1].delta == 1
        assert by_id[2].rank_before == 1 and by_id[2].rank_after == 2 and by_id[2].delta == -1

    def test_missing_prior_gameweek_in_standings_yields_empty(self):
        entries = [_rival(1, history=_history(50))]
        standings = {2: (1,)}  # no entry for gameweek 1
        assert biggest_movers(entries, standings, gameweek=2) == ()


class TestTemplateOverlap:
    def test_counts_starters_in_the_template(self):
        entries = [_rival(1, pick_positions={10: 1, 11: 2, 12: 12})]
        [row] = template_overlap(entries, template_xi_ids=(10, 11, 99))
        assert row.overlap_count == 2
        assert row.template_size == 3

    def test_benched_template_player_does_not_count(self):
        entries = [_rival(1, pick_positions={10: 12})]
        [row] = template_overlap(entries, template_xi_ids=(10,))
        assert row.overlap_count == 0


class TestBiggestDifferentialHaul:
    def test_only_globally_unique_picks_count_as_differentials(self):
        entries = [
            _rival(1, picks={10: 1, 20: 1}),
            _rival(2, picks={10: 1, 30: 1}),  # shares player 10 with entry 1
        ]
        live_points = {10: 100, 20: 7, 30: 9}
        rows = {row.entry_id: row for row in biggest_differential_haul(entries, live_points)}
        assert rows[1].differential_player_ids == (20,)
        assert rows[1].total_points == 7
        assert rows[2].differential_player_ids == (30,)
        assert rows[2].total_points == 9

    def test_a_player_nobody_else_owns_is_still_a_differential_with_zero_others(self):
        entries = [_rival(1, picks={10: 1})]
        rows = biggest_differential_haul(entries, {10: 5})
        assert rows[0].differential_player_ids == (10,)


class TestChipsPlayed:
    def test_bench_boost_effect_is_the_sum_of_bench_position_points(self):
        entries = [
            _rival(
                1,
                pick_positions={10: 1, 11: 12, 12: 13},
                chips=(ChipUsage(name="bboost", gameweek=3),),
            )
        ]
        standings = {3: (1,)}
        [row] = chips_played(entries, {10: 5, 11: 8, 12: 2}, standings, gameweek=3)
        assert row.chip_name == "bboost"
        assert row.points_effect == 10  # positions 12 and 13 -> players 11, 12 -> 8 + 2
        assert row.resulting_rank == 1

    def test_triple_captain_effect_is_one_extra_multiplier_of_the_captains_points(self):
        entries = [
            _rival(1, picks={10: 3}, chips=(ChipUsage(name="3xc", gameweek=3),)),
        ]
        standings = {3: (1,)}
        [row] = chips_played(entries, {10: 9}, standings, gameweek=3)
        assert row.points_effect == 9

    def test_wildcard_has_no_points_effect(self):
        entries = [_rival(1, chips=(ChipUsage(name="wildcard", gameweek=3),))]
        standings = {3: (1,)}
        [row] = chips_played(entries, {}, standings, gameweek=3)
        assert row.points_effect is None
        assert row.resulting_rank == 1

    def test_a_chip_played_in_a_different_gameweek_is_excluded(self):
        entries = [_rival(1, chips=(ChipUsage(name="wildcard", gameweek=2),))]
        assert chips_played(entries, {}, {}, gameweek=3) == ()
