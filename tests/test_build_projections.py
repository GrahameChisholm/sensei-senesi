"""Tests for scripts.build_projections's pure assembly functions -- no network, no real snapshot
files. The orchestrating build_projections()/main() entry points are integration-only (real live
network + disk), matching how the rest of this repo's batch-job wiring is tested only at the
pure-function level.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from api.state import _player_actual_from_dict, _simulation_from_dict
from engine.aggregate import ComponentBreakdown
from engine.data.cold_start import fit_cold_start_priors
from engine.data.player_history import PlayerGameweekActual
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.simulate import PlayerSimulationSummary
from scripts.build_projections import (
    _deadline_times_for_gameweeks,
    _serialize_player_history,
    _serialize_simulation,
    assemble_projection_cache,
    build_fixture_list,
    merge_cold_start_projections,
    write_projection_cache,
)

TEAM_A, TEAM_B, TEAM_C = 1, 2, 3


def _breakdown(total: float = 4.0) -> ComponentBreakdown:
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


def _minutes() -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )


def _horizon_projection(player_id: int, position: str, gameweeks: list[int]) -> object:
    return project_player_horizon(
        player_id,
        position,
        {
            gw: project_player_gameweek(player_id, position, gw, _minutes(), _breakdown())
            for gw in gameweeks
        },
    )


def _live_elements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "web_name": "Engine",
                "first_name": "En",
                "second_name": "Gine",
                "team": TEAM_A,
                "element_type": 3,
                "now_cost": 75,
                "status": "a",
                "chance_of_playing_next_round": None,
            },
            {
                "id": 2,
                "web_name": "ColdStarter",
                "first_name": "Cold",
                "second_name": "Starter",
                "team": TEAM_B,
                "element_type": 4,
                "now_cost": 55,
                "status": "a",
                "chance_of_playing_next_round": 100.0,
            },
            {
                "id": 3,
                "web_name": "NoFixture",
                "first_name": "No",
                "second_name": "Fixture",
                "team": TEAM_C,
                "element_type": 4,
                "now_cost": 45,
                "status": "a",
                "chance_of_playing_next_round": 100.0,
            },
        ]
    )


def _live_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": TEAM_A, "name": "Team A", "short_name": "TMA"},
            {"id": TEAM_B, "name": "Team B", "short_name": "TMB"},
            {"id": TEAM_C, "name": "Team C", "short_name": "TMC"},
        ]
    )


def _fixtures_df() -> pd.DataFrame:
    # Team A vs Team B in GW1 and GW2; Team C has no fixture at all in the horizon.
    return pd.DataFrame(
        [
            {
                "event": 1,
                "team_h": TEAM_A,
                "team_a": TEAM_B,
                "kickoff_time": "2026-08-22T14:00:00Z",
                "team_h_difficulty": 2,
                "team_a_difficulty": 4,
            },
            {
                "event": 2,
                "team_h": TEAM_B,
                "team_a": TEAM_A,
                "kickoff_time": "2026-08-29T14:00:00Z",
                "team_h_difficulty": 3,
                "team_a_difficulty": 3,
            },
            {"event": None, "team_h": TEAM_A, "team_a": TEAM_C, "kickoff_time": None},
        ]
    )


class TestBuildFixtureList:
    def test_one_row_per_team_per_fixture(self):
        rows = build_fixture_list(_fixtures_df())
        # 2 scheduled fixtures x 2 sides = 4 rows; the unscheduled (event=None) row is skipped.
        assert len(rows) == 4

    def test_home_and_away_perspectives_are_both_present(self):
        rows = build_fixture_list(_fixtures_df())
        gw1 = [r for r in rows if r["gameweek"] == 1]
        assert {r["team_id"] for r in gw1} == {TEAM_A, TEAM_B}
        home_row = next(r for r in gw1 if r["team_id"] == TEAM_A)
        away_row = next(r for r in gw1 if r["team_id"] == TEAM_B)
        assert home_row["is_home"] is True
        assert home_row["opponent_id"] == TEAM_B
        assert away_row["is_home"] is False
        assert away_row["opponent_id"] == TEAM_A

    def test_unscheduled_fixture_is_skipped(self):
        rows = build_fixture_list(_fixtures_df())
        assert not any(r["team_id"] == TEAM_C for r in rows)

    def test_difficulty_is_carried_through_from_the_correct_side(self):
        rows = build_fixture_list(_fixtures_df())
        gw1 = [r for r in rows if r["gameweek"] == 1]
        home_row = next(r for r in gw1 if r["team_id"] == TEAM_A)
        away_row = next(r for r in gw1 if r["team_id"] == TEAM_B)
        assert home_row["difficulty"] == 2
        assert away_row["difficulty"] == 4


def _cold_start_priors():
    rows = []
    for position, value in (("FWD", 55), ("MID", 75)):
        for _ in range(10):
            rows.append(
                {
                    "position": position,
                    "value": value,
                    "minutes": 80,
                    "goals_scored": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "defensive_contribution": 0,
                    "saves": 0,
                    "bonus": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "penalties_missed": 0,
                    "own_goals": 0,
                }
            )
    return fit_cold_start_priors(pd.DataFrame(rows))


class TestMergeColdStartProjections:
    def test_engine_projected_player_is_untouched(self):
        engine_projections = {1: _horizon_projection(1, "MID", [1, 2])}
        team_id_by_player = {1: TEAM_A, 2: TEAM_B, 3: TEAM_C}
        team_gws = {(TEAM_A, 1), (TEAM_A, 2), (TEAM_B, 1), (TEAM_B, 2)}
        result, cold_start_ids = merge_cold_start_projections(
            _live_elements(),
            engine_projections,
            _cold_start_priors(),
            team_id_by_player,
            team_gws,
            [1, 2],
        )
        assert result[1] is engine_projections[1]
        assert 1 not in cold_start_ids

    def test_uncovered_player_gets_a_cold_start_projection(self):
        team_id_by_player = {1: TEAM_A, 2: TEAM_B, 3: TEAM_C}
        team_gws = {(TEAM_A, 1), (TEAM_A, 2), (TEAM_B, 1), (TEAM_B, 2)}
        result, cold_start_ids = merge_cold_start_projections(
            _live_elements(), {}, _cold_start_priors(), team_id_by_player, team_gws, [1, 2]
        )
        assert 2 in cold_start_ids
        assert set(result[2].gameweeks) == {1, 2}

    def test_player_whose_team_has_no_fixture_at_all_is_omitted(self):
        team_id_by_player = {1: TEAM_A, 2: TEAM_B, 3: TEAM_C}
        team_gws = {(TEAM_A, 1), (TEAM_A, 2), (TEAM_B, 1), (TEAM_B, 2)}
        result, cold_start_ids = merge_cold_start_projections(
            _live_elements(), {}, _cold_start_priors(), team_id_by_player, team_gws, [1, 2]
        )
        assert 3 not in result
        assert 3 not in cold_start_ids

    def test_every_covered_live_player_ends_up_in_result(self):
        team_id_by_player = {1: TEAM_A, 2: TEAM_B, 3: TEAM_C}
        team_gws = {(TEAM_A, 1), (TEAM_A, 2), (TEAM_B, 1), (TEAM_B, 2)}
        engine_projections = {1: _horizon_projection(1, "MID", [1, 2])}
        result, _ = merge_cold_start_projections(
            _live_elements(),
            engine_projections,
            _cold_start_priors(),
            team_id_by_player,
            team_gws,
            [1, 2],
        )
        assert set(result) == {1, 2}  # 3 correctly excluded (no fixture)

    def test_within_club_rank_differentiates_two_same_bucket_cold_start_players(self):
        # Prior season: two price tiers at the same club and position, so the rank-fitted prior
        # in engine.data.cold_start has real signal to key off (mirrors
        # tests/test_cold_start.py's own _club_ranked_rows fixture).
        rows = []
        for minutes, goals in ((88, 1), (10, 0)):
            for _ in range(15):
                rows.append(
                    {
                        "position": "MID",
                        "value": 60,
                        "minutes": minutes,
                        "goals_scored": goals,
                        "assists": 0,
                        "clean_sheets": 0,
                        "goals_conceded": 1,
                        "defensive_contribution": 0,
                        "saves": 0,
                        "bonus": 0,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "penalties_missed": 0,
                        "own_goals": 0,
                        "team": "Newclub",
                        "element": "first" if minutes == 88 else "second",
                    }
                )
        priors = fit_cold_start_priors(pd.DataFrame(rows))

        live_elements = pd.DataFrame(
            [
                {"id": 101, "team": TEAM_A, "element_type": 3, "now_cost": 62},
                {"id": 102, "team": TEAM_A, "element_type": 3, "now_cost": 58},
            ]
        )
        team_id_by_player = {101: TEAM_A, 102: TEAM_A}
        team_gws = {(TEAM_A, 1)}

        result, cold_start_ids = merge_cold_start_projections(
            live_elements, {}, priors, team_id_by_player, team_gws, [1]
        )

        assert cold_start_ids == {101, 102}
        assert (
            result[101].gameweeks[1].expected_points > result[102].gameweeks[1].expected_points
        ), "the pricier (rank-1) player must outproject the cheaper (rank-2) one at the same bucket"

    def test_within_club_rank_breaks_ties_on_equal_price_rather_than_collapsing_them(self):
        # Two live players at the SAME club, position, AND price -- the real case that silently
        # defeated the whole rank differentiator (`method="min"` gives every tied player rank 1),
        # concretely the backup-goalkeeper-at-the-price-floor case. With a real tiebreak, one of
        # the two must land on rank tier "1" and the other on "2", not both on "1".
        rows = []
        for minutes, goals in ((88, 1), (10, 0)):
            for _ in range(15):
                rows.append(
                    {
                        "position": "MID",
                        "value": 60,
                        "minutes": minutes,
                        "goals_scored": goals,
                        "assists": 0,
                        "clean_sheets": 0,
                        "goals_conceded": 1,
                        "defensive_contribution": 0,
                        "saves": 0,
                        "bonus": 0,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "penalties_missed": 0,
                        "own_goals": 0,
                        "team": "Newclub",
                        "element": "first" if minutes == 88 else "second",
                    }
                )
        priors = fit_cold_start_priors(pd.DataFrame(rows))

        live_elements = pd.DataFrame(
            [
                {"id": 101, "team": TEAM_A, "element_type": 3, "now_cost": 60},
                {"id": 102, "team": TEAM_A, "element_type": 3, "now_cost": 60},
            ]
        )
        team_id_by_player = {101: TEAM_A, 102: TEAM_A}
        team_gws = {(TEAM_A, 1)}

        result, cold_start_ids = merge_cold_start_projections(
            live_elements, {}, priors, team_id_by_player, team_gws, [1]
        )

        assert cold_start_ids == {101, 102}
        first = result[101].gameweeks[1].expected_points
        second = result[102].gameweeks[1].expected_points
        assert first != second, (
            "two same-price, same-club, same-position cold-start players must not collapse onto "
            "an identical projection"
        )


class TestAssembleProjectionCache:
    def _cache(self):
        team_id_by_player = {1: TEAM_A, 2: TEAM_B, 3: TEAM_C}
        team_gws = {(TEAM_A, 1), (TEAM_A, 2), (TEAM_B, 1), (TEAM_B, 2)}
        engine_projections = {1: _horizon_projection(1, "MID", [1, 2])}
        projections, cold_start_ids = merge_cold_start_projections(
            _live_elements(),
            engine_projections,
            _cold_start_priors(),
            team_id_by_player,
            team_gws,
            [1, 2],
        )
        fixture_rows = build_fixture_list(_fixtures_df())
        return assemble_projection_cache(
            season="2026-27",
            gameweek=1,
            horizon_gameweeks=[1, 2],
            projections=projections,
            cold_start_ids=cold_start_ids,
            live_elements=_live_elements(),
            live_teams=_live_teams(),
            fixture_rows=fixture_rows,
            generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
            deadline_time=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
            deadline_passed=False,
            model_version="test-version",
            diagnostics={"training_rows": 100},
            deadline_times={
                1: datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
                2: datetime(2026, 8, 28, 17, 30, tzinfo=UTC),
            },
        )

    def test_top_level_shape(self):
        cache = self._cache()
        for key in (
            "season",
            "gameweek",
            "horizon_gameweeks",
            "deadline_passed",
            "generated_at",
            "deadline_time",
            "model_version",
            "projections",
            "players",
            "teams",
            "fixtures",
            "diagnostics",
            "deadline_times",
        ):
            assert key in cache

    def test_carries_a_deadline_for_every_horizon_gameweek(self):
        # api/state.py resolves the decision gameweek from these, so every horizon gameweek needs
        # one, not just the gameweek the cache was built for.
        cache = self._cache()
        assert cache["deadline_times"] == {
            "1": "2026-08-21T17:30:00+00:00",
            "2": "2026-08-28T17:30:00+00:00",
        }

    def test_every_live_player_present_except_the_one_with_no_fixture(self):
        cache = self._cache()
        assert set(cache["players"]) == {"1", "2", "3"}
        assert set(cache["projections"]) == {"1", "2"}

    def test_fixtures_carry_the_difficulty_field(self):
        cache = self._cache()
        gw1_home_row = next(
            row for row in cache["fixtures"] if row["gameweek"] == 1 and row["team_id"] == TEAM_A
        )
        assert gw1_home_row["difficulty"] == 2

    def test_source_and_low_confidence_flags(self):
        cache = self._cache()
        assert cache["players"]["1"]["source"] == "engine"
        assert cache["players"]["1"]["low_confidence"] is False
        assert cache["players"]["2"]["source"] == "cold_start"
        assert cache["players"]["2"]["low_confidence"] is True

    def test_missing_chance_of_playing_defaults_to_100(self):
        cache = self._cache()
        assert cache["players"]["1"]["chance_of_playing_next_round"] == 100.0

    def test_projection_breakdown_round_trips(self):
        cache = self._cache()
        gw1 = cache["projections"]["1"]["gameweeks"][0]
        assert gw1["gameweek"] == 1
        assert gw1["breakdown"]["appearance"] == 2.0
        assert gw1["simulation"] is None

    def test_teams_keyed_by_id_with_short_name(self):
        cache = self._cache()
        assert cache["teams"][str(TEAM_A)] == {"name": "Team A", "short_name": "TMA"}

    def test_json_serializable(self):
        import json

        json.dumps(self._cache())


class TestWriteProjectionCache:
    def test_writes_valid_json_at_expected_path(self, tmp_path):
        cache = {"season": "2026-27", "gameweek": 1}
        path = write_projection_cache(cache, tmp_path, "2026-27", 1)
        assert path == tmp_path / "2026-27" / "gw01.json"
        import json

        assert json.loads(path.read_text()) == cache

    def test_no_leftover_temp_file(self, tmp_path):
        cache = {"season": "2026-27", "gameweek": 1}
        write_projection_cache(cache, tmp_path, "2026-27", 1)
        assert not (tmp_path / "2026-27" / "gw01.json.tmp").exists()

    def test_overwrites_an_existing_cache(self, tmp_path):
        write_projection_cache({"gameweek": 1, "v": "old"}, tmp_path, "2026-27", 1)
        write_projection_cache({"gameweek": 1, "v": "new"}, tmp_path, "2026-27", 1)
        import json

        path = tmp_path / "2026-27" / "gw01.json"
        assert json.loads(path.read_text())["v"] == "new"


class TestSerializePlayerHistoryRoundTrip:
    """DIFFERENTIALS_PLAN Phase 1: selected/starts/value/transfers_in/transfers_out/bps must
    survive the full write -> JSON -> api.state read cycle, and an old cache missing them must
    still load, with all six coming back as None rather than raising or defaulting to 0."""

    def _actual(self, **overrides) -> PlayerGameweekActual:
        base = dict(
            gameweek=1,
            minutes=90,
            goals_scored=1,
            assists=0,
            clean_sheets=0,
            goals_conceded=1,
            own_goals=0,
            penalties_saved=0,
            penalties_missed=0,
            saves=0,
            yellow_cards=0,
            red_cards=0,
            bonus=2,
            defensive_contribution=0,
            total_points=7,
            expected_goals=0.8,
            expected_assists=0.1,
            expected_goal_involvements=0.9,
            expected_goals_conceded=1.1,
            selected=123456,
            starts=1,
            value=55,
            transfers_in=1000,
            transfers_out=200,
            bps=28,
        )
        base.update(overrides)
        return PlayerGameweekActual(**base)

    def test_new_fields_round_trip_through_json(self):
        import json

        serialized = _serialize_player_history([self._actual()])
        reloaded = json.loads(json.dumps(serialized))
        actual = _player_actual_from_dict(reloaded[0])

        assert actual.selected == 123456
        assert actual.starts == 1
        assert actual.value == 55
        assert actual.transfers_in == 1000
        assert actual.transfers_out == 200
        assert actual.bps == 28

    def test_a_cache_written_before_these_fields_existed_still_loads(self):
        """Simulates an old on-disk cache: the dict simply has no keys for the six new fields,
        the same shape ``_serialize_player_history`` produced before this change."""
        old_style_row = {
            "gameweek": 1,
            "minutes": 90,
            "goals_scored": 1,
            "assists": 0,
            "clean_sheets": 0,
            "goals_conceded": 1,
            "own_goals": 0,
            "penalties_saved": 0,
            "penalties_missed": 0,
            "saves": 0,
            "yellow_cards": 0,
            "red_cards": 0,
            "bonus": 2,
            "defensive_contribution": 0,
            "total_points": 7,
            "expected_goals": 0.8,
            "expected_assists": 0.1,
            "expected_goal_involvements": 0.9,
            "expected_goals_conceded": 1.1,
        }

        actual = _player_actual_from_dict(old_style_row)

        assert actual.selected is None
        assert actual.starts is None
        assert actual.value is None
        assert actual.transfers_in is None
        assert actual.transfers_out is None
        assert actual.bps is None


class TestDeadlineTimesForGameweeks:
    def _events(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {"id": 1, "deadline_time": "2026-08-21T17:30:00Z"},
                {"id": 2, "deadline_time": "2026-08-28T17:30:00Z"},
            ]
        )

    def test_returns_one_deadline_per_scheduled_gameweek(self):
        deadlines = _deadline_times_for_gameweeks(self._events(), [1, 2])

        assert deadlines == {
            1: datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
            2: datetime(2026, 8, 28, 17, 30, tzinfo=UTC),
        }

    def test_skips_a_gameweek_fpl_has_not_scheduled_yet(self):
        # Skipping rather than raising keeps a whole build from failing over one unscheduled
        # horizon gameweek; api/state.py just falls back to that gameweek's first kickoff.
        deadlines = _deadline_times_for_gameweeks(self._events(), [1, 2, 3])

        assert set(deadlines) == {1, 2}


class TestSerializeSimulationStdRoundTrip:
    """MINI_LEAGUE_PLAN M9: ``std`` must survive the full write -> JSON -> api.state read cycle,
    and an old cache missing it must still load, falling back to the normal-distribution spread
    approximation from floor (P10)/ceiling (P90) rather than raising or defaulting to zero."""

    def _simulation(self) -> PlayerSimulationSummary:
        return PlayerSimulationSummary(
            player_id=1,
            mean=5.0,
            median=4.5,
            floor=1.0,
            ceiling=11.0,
            prob_big_haul=0.2,
            raw_points=np.array([]),
            std=3.2,
        )

    def test_std_round_trips_through_json(self):
        serialized = _serialize_simulation(self._simulation())
        reloaded = json.loads(json.dumps(serialized))
        simulation = _simulation_from_dict(reloaded, player_id=1)

        assert simulation.std == 3.2

    def test_a_cache_written_before_std_existed_falls_back_to_the_normal_approximation(self):
        """Simulates an old on-disk cache: the dict simply has no ``std`` key, the same shape
        ``_serialize_simulation`` produced before this change."""
        old_style_row = {
            "mean": 5.0,
            "median": 4.5,
            "floor": 1.0,
            "ceiling": 11.0,
            "prob_big_haul": 0.2,
        }

        simulation = _simulation_from_dict(old_style_row, player_id=1)

        assert simulation.std == pytest.approx((11.0 - 1.0) / 2.5631)
