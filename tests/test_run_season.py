"""Tests for backtest/run_season.py — the versioned real-backtest driver.

No real network calls: data-prep caching is tested against an injected ``httpx.MockTransport``
(matching ``tests/test_fpl_client.py``/``tests/test_understat_client.py``'s convention); every
feature-engineering and fit/predict function is tested against small synthetic data.
"""

from __future__ import annotations

import httpx
import pandas as pd

from backtest.harness import run_walk_forward
from backtest.run_season import (
    build_penalty_attempts_frame,
    compute_days_since_last_appearance,
    compute_fixture_congestion,
    compute_team_rotation_propensity,
    compute_zero_minute_streak_length,
    engineer_features,
    fetch_vaastav_merged_gw,
    fit_fn,
    make_predict_fn,
    score_season,
    season_label,
)


def test_season_label_formats_start_year():
    assert season_label(2025) == "2025-26"
    assert season_label(1999) == "1999-00"


def test_fetch_vaastav_merged_gw_caches_to_disk(tmp_path):
    csv_body = "element,position,team,GW\n1,MID,Team A,1\n"
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text=csv_body)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    first = fetch_vaastav_merged_gw(2025, tmp_path, client)
    second = fetch_vaastav_merged_gw(
        2025, tmp_path, client
    )  # should hit the cache, not the network

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert (tmp_path / "vaastav" / "2025-26" / "merged_gw.parquet").exists()


def test_fetch_vaastav_merged_gw_refresh_forces_refetch(tmp_path):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url.path)
        return httpx.Response(200, text="element,position,team,GW\n1,MID,Team A,1\n")

    client = httpx.Client(transport=httpx.MockTransport(handler))

    fetch_vaastav_merged_gw(2025, tmp_path, client)
    fetch_vaastav_merged_gw(2025, tmp_path, client, refresh=True)

    assert len(calls) == 2


# --------------------------------------------------------------------------------------------
# Feature-engineering helpers
# --------------------------------------------------------------------------------------------


def _gw_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    return df


def test_compute_fixture_congestion_counts_recent_team_matches():
    gw = _gw_frame(
        [
            {"player_id": 1, "team": "A", "kickoff_time": "2025-08-01"},
            {"player_id": 1, "team": "A", "kickoff_time": "2025-08-04"},  # 3 days after the first
            {"player_id": 1, "team": "A", "kickoff_time": "2025-08-20"},  # long gap after that
        ]
    )

    congestion = compute_fixture_congestion(gw, window_days=7)

    assert list(congestion) == [0, 1, 0]


def test_compute_days_since_last_appearance_tracks_gaps_and_defaults_first_row():
    gw = _gw_frame(
        [
            {"player_id": 1, "kickoff_time": "2025-08-01", "minutes": 90},
            {"player_id": 1, "kickoff_time": "2025-08-08", "minutes": 0},
            {"player_id": 1, "kickoff_time": "2025-08-22", "minutes": 90},
        ]
    )

    days_since = compute_days_since_last_appearance(gw, default_days=60.0)

    # Row 3's "last appearance" is still 2025-08-01 (the 08-08 row had 0 minutes, so it never
    # became a new last-appearance date) -- 21 days, not 14.
    assert list(days_since) == [60.0, 7.0, 21.0]


def test_compute_zero_minute_streak_length_resets_on_appearance():
    gw = _gw_frame(
        [
            {"player_id": 1, "kickoff_time": "2025-08-01", "minutes": 90},
            {"player_id": 1, "kickoff_time": "2025-08-08", "minutes": 0},
            {"player_id": 1, "kickoff_time": "2025-08-15", "minutes": 0},
            {"player_id": 1, "kickoff_time": "2025-08-22", "minutes": 90},
            {"player_id": 1, "kickoff_time": "2025-08-29", "minutes": 0},
        ]
    )

    streak = compute_zero_minute_streak_length(gw)

    assert list(streak) == [0.0, 0.0, 1.0, 2.0, 0.0]


def test_compute_team_rotation_propensity_higher_for_more_variable_squad():
    # Team A: both players start every match -- zero dispersion once a prior gameweek exists.
    # Team B: one nailed-on starter, one rotated in/out -- real dispersion.
    rows = []
    for gw_num, kickoff in enumerate(["2025-08-01", "2025-08-08", "2025-08-15"], start=1):
        rows.append(
            {"player_id": 1, "team": "A", "gameweek": gw_num, "kickoff_time": kickoff, "starts": 1}
        )
        rows.append(
            {"player_id": 2, "team": "A", "gameweek": gw_num, "kickoff_time": kickoff, "starts": 1}
        )
        rows.append(
            {"player_id": 3, "team": "B", "gameweek": gw_num, "kickoff_time": kickoff, "starts": 1}
        )
        rows.append(
            {
                "player_id": 4,
                "team": "B",
                "gameweek": gw_num,
                "kickoff_time": kickoff,
                "starts": gw_num % 2,
            }
        )
    gw = _gw_frame(rows)

    propensity = compute_team_rotation_propensity(gw)
    gw = gw.assign(team_rotation_propensity=propensity)

    # By the third gameweek (two prior gameweeks of evidence), team B's dispersion should exceed
    # team A's near-zero dispersion.
    third_gw = gw[gw["gameweek"] == 3]
    team_a_value = third_gw[third_gw["team"] == "A"]["team_rotation_propensity"].iloc[0]
    team_b_value = third_gw[third_gw["team"] == "B"]["team_rotation_propensity"].iloc[0]
    assert team_b_value > team_a_value


def test_build_penalty_attempts_frame_expands_scored_and_missed_counts():
    training_history = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "realized_penalty_goals": [2.0, 0.0, 1.0],
            "penalties_missed": [0, 1, 0],
        }
    )

    attempts = build_penalty_attempts_frame(training_history)

    assert len(attempts) == 4  # 2 scored (player 1) + 1 missed (player 2) + 1 scored (player 3)
    assert attempts[attempts["player_id"] == 1]["scored"].tolist() == [1, 1]
    assert attempts[attempts["player_id"] == 2]["scored"].tolist() == [0]
    assert attempts[attempts["player_id"] == 3]["scored"].tolist() == [1]


# --------------------------------------------------------------------------------------------
# End-to-end wiring on a small synthetic season (no network)
# --------------------------------------------------------------------------------------------


def _synthetic_season() -> tuple[pd.DataFrame, pd.DataFrame, dict, dict]:
    """Two teams, four players, five gameweeks -- just enough for every engineered feature to
    have real (non-degenerate) prior history by gameweek 3+, and for the walk-forward harness to
    have something to refit against."""
    teams = pd.DataFrame({"id": [1, 2], "name": ["Team A", "Team B"]})

    kickoffs = pd.date_range("2025-08-16", periods=5, freq="7D", tz="UTC")
    rows = []
    for i, kickoff in enumerate(kickoffs):
        gw_num = i + 1
        a_home = i % 2 == 0
        a_score, b_score = (2, 0) if a_home else (0, 2)
        for player_id, team, position, opponent_id, is_home in [
            (1, "Team A", "MID", 2, a_home),
            (2, "Team A", "DEF", 2, a_home),
            (3, "Team B", "MID", 1, not a_home),
            (4, "Team B", "FWD", 1, not a_home),
        ]:
            rows.append(
                {
                    "element": player_id,
                    "position": position,
                    "team": team,
                    "GW": gw_num,
                    "kickoff_time": kickoff.isoformat(),
                    "minutes": 90,
                    "starts": 1,
                    "was_home": is_home,
                    "opponent_team": opponent_id,
                    "total_points": 2 + player_id % 3,
                    "bonus": player_id % 3,
                    "clean_sheets": (
                        1
                        if (team == "Team A" and b_score == 0)
                        or (team == "Team B" and a_score == 0)
                        else 0
                    ),
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "defensive_contribution": 8 if position == "DEF" else 5,
                    "penalties_missed": 0,
                    "team_h_score": a_score if a_home else b_score,
                    "team_a_score": b_score if a_home else a_score,
                }
            )
    merged_gw = pd.DataFrame(rows)

    def _team_history(name: str, is_home_seq: list[bool]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": kickoffs,
                "xG": [1.5 if h else 0.8 for h in is_home_seq],
                "xGA": [0.7 if h else 1.4 for h in is_home_seq],
                "minutes": 90.0,
            }
        )

    a_home_seq = [i % 2 == 0 for i in range(5)]
    team_histories = {
        "Team A": _team_history("Team A", a_home_seq),
        "Team B": _team_history("Team B", [not h for h in a_home_seq]),
    }

    def _player_history(base_npxg: float, base_xa: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": kickoffs,
                "npxG": [base_npxg] * 5,
                "xA": [base_xa] * 5,
                "goals": [0] * 5,
                "npg": [0] * 5,
                "time": [90] * 5,
                "season": ["2025"] * 5,
            }
        )

    player_histories = {
        1: _player_history(0.3, 0.2),
        2: _player_history(0.05, 0.05),
        3: _player_history(0.25, 0.15),
        4: _player_history(0.4, 0.1),
    }
    return merged_gw, teams, team_histories, player_histories


def test_engineer_features_drops_rows_with_no_prior_history_and_fills_expected_columns():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()

    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)

    # Gameweek 1 has no prior history for anyone -- must be dropped, not left NaN.
    assert 1 not in set(engineered["gameweek"])
    assert engineered["gameweek"].min() >= 2
    for col in [
        "npxg_per_90",
        "xa_per_90",
        "team_xg_per_90",
        "opponent_xg_per_90",
        "fixture_congestion",
    ]:
        assert engineered[col].notna().all()
    assert set(engineered["position"]) == {"MID", "DEF", "FWD"}


def test_fit_fn_and_predict_fn_wire_together_via_walk_forward():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)

    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=1)

    assert not result.predictions.empty
    assert set(result.predictions["gameweek"]) <= set(gameweeks)
    assert result.predictions["expected_points"].notna().all()
    for col in ["p_zero", "p_1_to_59", "p_60_plus", "player_clean_sheet_probability"]:
        assert col in result.predictions.columns


def test_score_season_produces_a_summary_without_crashing():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=1)

    ground_truth = engineered[
        ["player_id", "gameweek", "position", "total_points", "minutes", "clean_sheets"]
    ]
    report = score_season(result.predictions, ground_truth)

    summary = report.summary()
    assert "Overall MAE" in summary
    assert isinstance(report.accuracy.overall_mae, float)
