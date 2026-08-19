"""Tests for engine.data.cross_season -- turning a cached prior-season vaastav frame into
current-season-keyed training rows, closing the GW1 cold-start gap live_adapter's own docstring
documents as unresolved.

All synthetic (no network) -- code/team/player joins are exercised directly against small
constructed DataFrames, mirroring tests/test_live_adapter.py's own style.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data.cross_season import (
    PRIOR_SEASON_GAMEWEEKS,
    merge_player_histories,
    player_code_map,
    prior_season_merged_gw,
    remap_player_histories,
    synthetic_team_rows,
    team_id_map,
)
from engine.data.live_adapter import MERGED_GW_COLUMNS

_OUTCOME_COLUMNS = [
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "defensive_contribution",
    "own_goals",
    "yellow_cards",
    "red_cards",
    "saves",
    "bps",
    "bonus",
    "penalties_missed",
]


def _prior_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 100, "code": 1, "name": "Prior Alpha", "short_name": "ALP"},
            {"id": 200, "code": 2, "name": "Prior Beta", "short_name": "BET"},
            {"id": 300, "code": 3, "name": "Prior Gamma", "short_name": "GAM"},  # relegated
        ]
    )


def _live_teams() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 1, "code": 1, "name": "Alpha", "short_name": "ALP"},
            {"id": 2, "code": 2, "name": "Beta", "short_name": "BET"},
        ]
    )


def _prior_players_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 11,
                "code": 1001,
                "web_name": "Survivor",
                "first_name": "S",
                "second_name": "One",
                "team": 100,
                "element_type": 1,
            },
            {
                "id": 12,
                "code": 1002,
                "web_name": "Mover",
                "first_name": "M",
                "second_name": "Two",
                "team": 300,
                "element_type": 2,
            },
            {
                "id": 13,
                "code": 9999,
                "web_name": "Departed",
                "first_name": "D",
                "second_name": "Three",
                "team": 200,
                "element_type": 3,
            },
        ]
    )


def _live_elements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": 911, "code": 1001, "team": 1, "element_type": 1},
            {"id": 912, "code": 1002, "team": 1, "element_type": 2},
        ]
    )


def _merged_gw_row(element: int, round_: int, opponent_team: int, position: str, team: str) -> dict:
    row = {
        "element": element,
        "round": round_,
        "position": position,
        "team": team,
        "opponent_team": opponent_team,
        "was_home": True,
        "kickoff_time": f"2025-08-{9 + round_ * 7:02d}T14:00:00Z",
        "team_h_score": 2,
        "team_a_score": 1,
        "value": 50,
        "selected": 100000,
        "transfers_out": 100,
        "transfers_balance": 0,
    }
    for col in _OUTCOME_COLUMNS:
        row[col] = 0
    row["minutes"] = 90
    row["total_points"] = 6
    return row


def _prior_merged_gw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _merged_gw_row(11, 1, opponent_team=300, position="GK", team="Prior Alpha"),
            _merged_gw_row(11, 2, opponent_team=200, position="GK", team="Prior Alpha"),
            _merged_gw_row(12, 1, opponent_team=100, position="DEF", team="Prior Gamma"),
            _merged_gw_row(13, 1, opponent_team=100, position="MID", team="Prior Beta"),
        ]
    )


def _relegated_team_ids() -> dict[int, int]:
    synthetic = synthetic_team_rows(_prior_teams(), _live_teams())
    prior = _prior_teams()
    relegated = prior[~prior["code"].isin(_live_teams()["code"])]
    return dict(zip(relegated["id"], synthetic["id"], strict=True))


def _build_merged_gw() -> pd.DataFrame:
    code_map = player_code_map(_prior_players_raw(), _live_elements())
    team_map = team_id_map(_prior_teams(), _live_teams())
    return prior_season_merged_gw(_prior_merged_gw(), code_map, team_map, _relegated_team_ids())


class TestPlayerCodeMap:
    def test_maps_surviving_players_by_code(self):
        mapping = player_code_map(_prior_players_raw(), _live_elements())
        assert mapping == {11: 911, 12: 912}

    def test_departed_player_absent_from_map(self):
        mapping = player_code_map(_prior_players_raw(), _live_elements())
        assert 13 not in mapping

    def test_duplicate_code_raises(self):
        dupe = pd.concat([_prior_players_raw(), _prior_players_raw().iloc[[0]]], ignore_index=True)
        with pytest.raises(ValueError, match="duplicate code"):
            player_code_map(dupe, _live_elements())


class TestTeamIdMap:
    def test_maps_surviving_clubs_by_code(self):
        mapping = team_id_map(_prior_teams(), _live_teams())
        assert mapping == {100: 1, 200: 2}

    def test_relegated_club_absent_from_map(self):
        mapping = team_id_map(_prior_teams(), _live_teams())
        assert 300 not in mapping


class TestSyntheticTeamRows:
    def test_relegated_club_gets_a_synthetic_id_above_live_max(self):
        rows = synthetic_team_rows(_prior_teams(), _live_teams())
        assert len(rows) == 1
        row = rows.iloc[0]
        assert row["name"] == "Prior Gamma"
        assert row["id"] > _live_teams()["id"].max()

    def test_no_relegated_clubs_gives_empty_frame_with_right_columns(self):
        prior_no_relegation = pd.DataFrame(
            [
                {"id": 1, "code": 1, "name": "Alpha", "short_name": "ALP"},
                {"id": 2, "code": 2, "name": "Beta", "short_name": "BET"},
            ]
        )
        rows = synthetic_team_rows(prior_no_relegation, _live_teams())
        assert rows.empty
        assert list(rows.columns) == ["id", "name", "short_name", "code"]


class TestPriorSeasonMergedGw:
    def test_output_has_exact_merged_gw_columns(self):
        result = _build_merged_gw()
        assert list(result.columns) == MERGED_GW_COLUMNS

    def test_player_ids_remapped_to_current_season(self):
        result = _build_merged_gw()
        assert set(result["player_id"]) == {911, 912}

    def test_departed_player_dropped_but_others_kept(self):
        result = _build_merged_gw()
        assert 13 not in result["player_id"].values
        assert 912 in result["player_id"].values

    def test_gameweek_offset_is_negative(self):
        result = _build_merged_gw()
        assert (result["GW"] <= 0).all()
        assert result["GW"].min() == 1 - PRIOR_SEASON_GAMEWEEKS

    def test_opponent_team_never_a_raw_prior_season_id(self):
        result = _build_merged_gw()
        assert not set(result["opponent_team"].astype(int)).intersection({100, 200, 300})

    def test_relegated_opponent_resolves_via_synthetic_id(self):
        result = _build_merged_gw()
        synthetic = synthetic_team_rows(_prior_teams(), _live_teams())
        synthetic_id = int(synthetic.iloc[0]["id"])
        row = result[(result["player_id"] == 911) & (result["GW"] == 1 - PRIOR_SEASON_GAMEWEEKS)]
        assert len(row) == 1
        assert int(row.iloc[0]["opponent_team"]) == synthetic_id

    def test_keeps_prior_season_position_and_club(self):
        result = _build_merged_gw()
        mover_row = result[result["player_id"] == 912].iloc[0]
        assert mover_row["position"] == "DEF"
        assert mover_row["team"] == "Prior Gamma"

    def test_unresolvable_opponent_raises(self):
        code_map = player_code_map(_prior_players_raw(), _live_elements())
        team_map = team_id_map(_prior_teams(), _live_teams())
        with pytest.raises(ValueError, match="opponent_team"):
            prior_season_merged_gw(
                _prior_merged_gw(), code_map, team_map, {}
            )  # no relegated mapping supplied

    def test_missing_required_column_raises(self):
        broken = _prior_merged_gw().drop(columns=["minutes"])
        code_map = player_code_map(_prior_players_raw(), _live_elements())
        team_map = team_id_map(_prior_teams(), _live_teams())
        with pytest.raises(ValueError, match="missing expected column"):
            prior_season_merged_gw(broken, code_map, team_map, _relegated_team_ids())

    def test_real_vaastav_shape_with_both_round_and_gw_columns_does_not_duplicate(self):
        # Every real vaastav merged_gw export (verified against this repo's own cached 2024/25 and
        # 2025/26 parquets) carries both `round` and `GW`, always identical -- the fixtures above
        # only ever define `round`, so this is the one case that would have caught the real
        # duplicate-"GW"-column bug (renaming `round -> GW` when `GW` already exists).
        with_both = _prior_merged_gw().assign(GW=lambda df: df["round"])
        code_map = player_code_map(_prior_players_raw(), _live_elements())
        team_map = team_id_map(_prior_teams(), _live_teams())
        result = prior_season_merged_gw(with_both, code_map, team_map, _relegated_team_ids())

        assert list(result.columns) == MERGED_GW_COLUMNS
        assert result.columns.tolist().count("GW") == 1
        assert isinstance(result["GW"], pd.Series)
        assert result["GW"].min() == 1 - PRIOR_SEASON_GAMEWEEKS


def _understat_history(player_dates: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": date, "time": 90.0, "npxG": 0.3, "xA": 0.1, "goals": 0.0, "npg": 0.0}
            for date in player_dates
        ]
    )


class TestRemapPlayerHistories:
    def test_remaps_keys_via_code_map(self):
        code_map = {11: 911, 12: 912}
        prior = {11: _understat_history(["2025-08-16"]), 12: _understat_history(["2025-08-17"])}
        result = remap_player_histories(prior, code_map)
        assert set(result) == {911, 912}

    def test_drops_players_absent_from_code_map(self):
        code_map = {11: 911}
        prior = {11: _understat_history(["2025-08-16"]), 13: _understat_history(["2025-08-17"])}
        result = remap_player_histories(prior, code_map)
        assert set(result) == {911}


class TestMergePlayerHistories:
    def test_concatenates_and_sorts_by_date(self):
        prior = {911: _understat_history(["2024-08-16", "2025-05-10"])}
        current = {911: _understat_history(["2026-08-22"])}
        result = merge_player_histories(prior, current)
        dates = result[911]["date"].tolist()
        assert dates == sorted(dates)
        assert len(result[911]) == 3

    def test_player_only_in_prior(self):
        prior = {911: _understat_history(["2025-05-10"])}
        result = merge_player_histories(prior, {})
        assert set(result) == {911}

    def test_player_only_in_current(self):
        current = {912: _understat_history(["2026-08-22"])}
        result = merge_player_histories({}, current)
        assert set(result) == {912}

    def test_player_with_no_history_anywhere_is_absent(self):
        result = merge_player_histories({911: pd.DataFrame()}, {911: pd.DataFrame()})
        assert 911 not in result
