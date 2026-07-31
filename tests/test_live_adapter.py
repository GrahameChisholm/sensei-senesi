"""Tests for engine.data.live_adapter — the live-snapshot -> engineer_features bridge (A1).

All synthetic (no network, no real snapshot files) except the end-to-end decisive check, which
feeds the adapter's own output through the real ``backtest.run_season.engineer_features`` to prove
the shapes actually connect, not just that this module's own functions don't crash.
"""

from __future__ import annotations

import pandas as pd
import pytest

from backtest.run_season import engineer_features
from engine.data.live_adapter import (
    DEFAULT_TOTAL_MANAGERS,
    build_merged_gw,
    build_player_histories_from_live_snapshot,
)

TEAM_A, TEAM_B = 10, 20


def _teams() -> pd.DataFrame:
    return pd.DataFrame([{"id": TEAM_A, "name": "Team A"}, {"id": TEAM_B, "name": "Team B"}])


def _elements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "team": TEAM_A,
                "element_type": 3,  # MID
                "now_cost": 75,
                "selected_by_percent": "10.0",
                "transfers_in_event": 1000,
                "transfers_out_event": 200,
            },
            {
                "id": 2,
                "team": TEAM_B,
                "element_type": 3,  # MID
                "now_cost": 60,
                "selected_by_percent": "5.0",
                "transfers_in_event": 100,
                "transfers_out_event": 50,
            },
        ]
    )


def _fixtures(gameweek: int, team_h: int = TEAM_A, team_a: int = TEAM_B) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event": gameweek,
                "team_h": team_h,
                "team_a": team_a,
                "kickoff_time": "2025-09-20T14:00:00Z",
            }
        ]
    )


def _history_row(element: int, round_: int, opponent: int, was_home: bool) -> dict:
    return {
        "element": element,
        "round": round_,
        "opponent_team": opponent,
        "was_home": was_home,
        "kickoff_time": f"2025-08-{9 + round_ * 7:02d}T14:00:00Z",
        "team_h_score": 2,
        "team_a_score": 1,
        "minutes": 90,
        "total_points": 6,
        "goals_scored": 1,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "defensive_contribution": 3,
        "own_goals": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bps": 25,
        "bonus": 1,
        "penalties_missed": 0,
        "starts": 1,
        "value": 70,
        "selected": 50000,
        "transfers_out": 1000,
        "transfers_balance": 500,
    }


def _element_summary_histories() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _history_row(1, 1, TEAM_B, True),
            _history_row(1, 2, TEAM_B, False),
            _history_row(2, 1, TEAM_A, False),
            _history_row(2, 2, TEAM_A, True),
        ]
    )


def _understat_player_match(fpl_id: int, date: str, season: str = "2025") -> dict:
    return {
        "fpl_id": fpl_id,
        "date": date,
        "season": season,
        "time": "90",
        "npxG": "0.3",
        "xA": "0.2",
        "goals": "1",
        "npg": "1",
    }


def _understat_player_histories() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _understat_player_match(1, "2025-08-16"),
            _understat_player_match(1, "2025-08-23"),
            _understat_player_match(2, "2025-08-16"),
            _understat_player_match(2, "2025-08-23"),
        ]
    )


def _understat_teams_history() -> pd.DataFrame:
    rows = []
    for team_title, is_home_flag in [("Team A", "h"), ("Team B", "a")]:
        for date in ("2025-08-16", "2025-08-23"):
            rows.append(
                {"team_title": team_title, "date": date, "xG": 1.5, "xGA": 1.1, "h_a": is_home_flag}
            )
    return pd.DataFrame(rows)


def test_build_merged_gw_includes_every_played_row_and_the_target_gameweek():
    merged = build_merged_gw(
        _elements(), _teams(), _fixtures(gameweek=3), _element_summary_histories(), gameweek=3
    )

    # 4 real played rows (2 players x 2 gameweeks) + 2 target-gameweek rows (1 fixture x 2 players).
    assert len(merged) == 6
    assert set(merged[merged["GW"] == 3]["player_id"]) == {1, 2}
    played = merged[merged["GW"] < 3]
    assert len(played) == 4
    assert set(played["player_id"]) == {1, 2}


def test_build_merged_gw_target_row_uses_real_fixture_and_current_price():
    merged = build_merged_gw(
        _elements(), _teams(), _fixtures(gameweek=3), _element_summary_histories(), gameweek=3
    )
    target = merged[merged["GW"] == 3].set_index("player_id")

    assert target.loc[1, "team"] == "Team A"
    assert target.loc[1, "opponent_team"] == TEAM_B
    assert bool(target.loc[1, "was_home"]) is True
    assert target.loc[2, "team"] == "Team B"
    assert target.loc[2, "opponent_team"] == TEAM_A
    assert bool(target.loc[2, "was_home"]) is False
    # value comes straight from now_cost -- not from any played-history row.
    assert target.loc[1, "value"] == pytest.approx(75.0)
    assert target.loc[2, "value"] == pytest.approx(60.0)


def test_build_merged_gw_target_row_converts_ownership_percent_to_a_count():
    merged = build_merged_gw(
        _elements(), _teams(), _fixtures(gameweek=3), _element_summary_histories(), gameweek=3
    )
    target = merged[merged["GW"] == 3].set_index("player_id")

    assert target.loc[1, "selected"] == pytest.approx(0.10 * DEFAULT_TOTAL_MANAGERS)
    assert target.loc[1, "transfers_balance"] == pytest.approx(1000.0 - 200.0)


def test_build_merged_gw_target_row_has_no_leaked_outcome_data():
    merged = build_merged_gw(
        _elements(), _teams(), _fixtures(gameweek=3), _element_summary_histories(), gameweek=3
    )
    target = merged[merged["GW"] == 3]

    for col in ("minutes", "total_points", "goals_scored", "bonus"):
        assert (target[col] == 0).all()
    assert target["team_h_score"].isna().all()


def test_build_merged_gw_blank_gameweek_produces_no_target_rows_for_that_team():
    # Team B has no fixture this gameweek at all -- player 2 must get no target row, while
    # player 1 (Team A, still playing) does.
    fixtures = pd.DataFrame(
        [{"event": 3, "team_h": TEAM_A, "team_a": 999, "kickoff_time": "2025-09-20T14:00:00Z"}]
    )
    teams = pd.concat([_teams(), pd.DataFrame([{"id": 999, "name": "Team C"}])], ignore_index=True)

    merged = build_merged_gw(_elements(), teams, fixtures, _element_summary_histories(), gameweek=3)
    target = merged[merged["GW"] == 3]

    assert set(target["player_id"]) == {1}


def test_build_merged_gw_double_gameweek_produces_two_target_rows():
    fixtures = pd.concat(
        [_fixtures(gameweek=3), _fixtures(gameweek=3, team_h=TEAM_B, team_a=TEAM_A)],
        ignore_index=True,
    )
    merged = build_merged_gw(
        _elements(), _teams(), fixtures, _element_summary_histories(), gameweek=3
    )
    target = merged[merged["GW"] == 3]

    assert len(target[target["player_id"] == 1]) == 2
    assert len(target[target["player_id"] == 2]) == 2


def test_build_player_histories_from_live_snapshot_groups_and_sorts_by_fpl_id():
    histories = build_player_histories_from_live_snapshot(
        _understat_player_histories(), season_start_year=2025
    )

    assert set(histories) == {1, 2}
    assert list(histories[1]["date"]) == sorted(histories[1]["date"])
    assert histories[1]["npxG"].dtype == float


def test_build_player_histories_from_live_snapshot_drops_future_seasons():
    histories = pd.concat(
        [
            _understat_player_histories(),
            pd.DataFrame([_understat_player_match(1, "2026-08-01", season="2026")]),
        ],
        ignore_index=True,
    )
    result = build_player_histories_from_live_snapshot(histories, season_start_year=2025)

    assert (pd.to_datetime(result[1]["date"]) < pd.Timestamp("2026-01-01", tz="UTC")).all()


def test_adapter_output_feeds_engineer_features_end_to_end():
    # The decisive check: the adapter's own output, fed through the REAL engineer_features,
    # must produce a usable point-in-time frame for the target gameweek -- not just avoid
    # crashing, but yield non-NaN rate features for players with enough prior history.
    from engine.data.live_adapter import build_merged_gw

    merged_gw = build_merged_gw(
        _elements(), _teams(), _fixtures(gameweek=3), _element_summary_histories(), gameweek=3
    )
    team_histories = {
        "Team A": _understat_teams_history()[_understat_teams_history()["team_title"] == "Team A"]
        .assign(minutes=90.0, is_home=lambda d: d["h_a"] == "h")
        .assign(date=lambda d: pd.to_datetime(d["date"], utc=True))
        .reset_index(drop=True),
        "Team B": _understat_teams_history()[_understat_teams_history()["team_title"] == "Team B"]
        .assign(minutes=90.0, is_home=lambda d: d["h_a"] == "h")
        .assign(date=lambda d: pd.to_datetime(d["date"], utc=True))
        .reset_index(drop=True),
    }
    player_histories = build_player_histories_from_live_snapshot(
        _understat_player_histories(), season_start_year=2025
    )

    engineered = engineer_features(merged_gw, _teams(), team_histories, player_histories)

    target_rows = engineered[engineered["gameweek"] == 3]
    assert set(target_rows["player_id"]) == {1, 2}
    assert target_rows["npxg_per_90"].notna().all()
    assert target_rows["team_xg_per_90"].notna().all()
