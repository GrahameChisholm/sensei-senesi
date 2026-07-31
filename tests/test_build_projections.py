"""Tests for scripts.build_projections's pure assembly functions -- no network, no real snapshot
files. The orchestrating build_projections()/main() entry points are integration-only (real live
network + disk), matching how the rest of this repo's batch-job wiring is tested only at the
pure-function level.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from engine.aggregate import ComponentBreakdown
from engine.data.cold_start import fit_cold_start_priors
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from scripts.build_projections import (
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
            },
            {
                "event": 2,
                "team_h": TEAM_B,
                "team_a": TEAM_A,
                "kickoff_time": "2026-08-29T14:00:00Z",
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
        ):
            assert key in cache

    def test_every_live_player_present_except_the_one_with_no_fixture(self):
        cache = self._cache()
        assert set(cache["players"]) == {"1", "2", "3"}
        assert set(cache["projections"]) == {"1", "2"}

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
