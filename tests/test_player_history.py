"""Tests for engine/data/player_history.py -- normalizing FPL's element-summary history into
PlayerGameweekActual, and converting a gameweek's raw counts into points-per-component
(PLAYER_STATS_PLAN D13/G4)."""

from __future__ import annotations

from engine.data.player_history import (
    PlayerGameweekActual,
    actual_points_for_gameweek,
    load_live_player_history,
)
from engine.scoring import DEF, FWD, GK, MID


def _history_row(**overrides) -> dict:
    row = {
        "round": 1,
        "minutes": 90,
        "goals_scored": 0,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 0,
        "own_goals": 0,
        "penalties_saved": 0,
        "penalties_missed": 0,
        "saves": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "bonus": 0,
        "defensive_contribution": 0,
        "total_points": 2,
        "expected_goals": 0.0,
        "expected_assists": 0.0,
        "expected_goal_involvements": 0.0,
        "expected_goals_conceded": 0.0,
    }
    row.update(overrides)
    return row


class _StubClient:
    def __init__(self, summaries: dict[int, dict]):
        self._summaries = summaries

    def iter_element_summaries(self, player_ids: list[int]) -> dict[int, dict]:
        return {
            player_id: self._summaries.get(player_id, {"history": []}) for player_id in player_ids
        }


def _actual(**overrides) -> PlayerGameweekActual:
    base = dict(
        gameweek=1,
        minutes=90,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        saves=0,
        yellow_cards=0,
        red_cards=0,
        bonus=0,
        defensive_contribution=0,
        total_points=2,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        expected_goals_conceded=0.0,
    )
    base.update(overrides)
    return PlayerGameweekActual(**base)


# --- load_live_player_history ------------------------------------------------------------------


def test_load_live_player_history_normalizes_rows_sorted_by_gameweek():
    client = _StubClient(
        {
            1: {
                "history": [
                    _history_row(round=2, minutes=45, goals_scored=1),
                    _history_row(round=1, minutes=90),
                ]
            }
        }
    )

    result = load_live_player_history(client, [1])

    assert [actual.gameweek for actual in result[1]] == [1, 2]
    assert result[1][1].goals_scored == 1
    assert result[1][1].minutes == 45


def test_load_live_player_history_returns_empty_list_for_no_history():
    client = _StubClient({1: {"history": []}})

    result = load_live_player_history(client, [1])

    assert result[1] == []


def test_load_live_player_history_defaults_missing_expected_stats_fields():
    client = _StubClient({1: {"history": [_history_row()]}})

    result = load_live_player_history(client, [1])

    assert result[1][0].expected_goals == 0.0


# --- actual_points_for_gameweek --------------------------------------------------------------


def test_appearance_points_by_minutes_tier():
    assert actual_points_for_gameweek(_actual(minutes=0), MID).appearance == 0.0
    assert actual_points_for_gameweek(_actual(minutes=45), MID).appearance == 1.0
    assert actual_points_for_gameweek(_actual(minutes=90), MID).appearance == 2.0


def test_goals_points_scaled_by_position():
    assert actual_points_for_gameweek(_actual(goals_scored=2), FWD).goals == 8.0
    assert actual_points_for_gameweek(_actual(goals_scored=2), GK).goals == 20.0


def test_assist_points():
    assert actual_points_for_gameweek(_actual(assists=2), MID).assists == 6.0


def test_clean_sheet_points_by_position():
    assert actual_points_for_gameweek(_actual(clean_sheets=1), DEF).clean_sheet == 4.0
    assert actual_points_for_gameweek(_actual(clean_sheets=1), FWD).clean_sheet == 0.0


def test_goals_conceded_penalty_only_for_gk_and_def():
    assert actual_points_for_gameweek(_actual(goals_conceded=2), DEF).goals_conceded == -1.0
    assert actual_points_for_gameweek(_actual(goals_conceded=2), MID).goals_conceded == 0.0


def test_goals_conceded_penalty_is_computed_per_gameweek_not_summed_first():
    """G4's core claim: conceding 1 goal in each of two separate gameweeks scores 0 penalty in
    each -- summing raw goals_conceded to 2 first and applying the -1 penalty once would wrongly
    produce -1 total instead of 0."""
    gw1 = actual_points_for_gameweek(_actual(goals_conceded=1), DEF).goals_conceded
    gw2 = actual_points_for_gameweek(_actual(goals_conceded=1), DEF).goals_conceded

    assert gw1 + gw2 == 0.0


def test_saves_points_include_penalty_save_bonus_gk_only():
    result = actual_points_for_gameweek(_actual(saves=6, penalties_saved=1), GK)
    assert result.saves == 2.0 + 5.0

    assert actual_points_for_gameweek(_actual(saves=6, penalties_saved=1), DEF).saves == 0.0


def test_cards_points():
    result = actual_points_for_gameweek(_actual(yellow_cards=1, red_cards=1), MID)
    assert result.cards == -1.0 + -3.0


def test_penalty_miss_points():
    assert actual_points_for_gameweek(_actual(penalties_missed=1), FWD).penalty_misses == -2.0


def test_own_goal_points():
    assert actual_points_for_gameweek(_actual(own_goals=1), DEF).own_goals == -2.0


def test_bonus_and_defensive_contribution_pass_through_fpls_recorded_values():
    result = actual_points_for_gameweek(_actual(bonus=3, defensive_contribution=2), DEF)
    assert result.bonus == 3.0
    assert result.defensive_contribution == 2.0
