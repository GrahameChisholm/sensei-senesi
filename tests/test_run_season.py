"""Tests for backtest/run_season.py — the versioned real-backtest driver.

No real network calls: data-prep caching is tested against an injected ``httpx.MockTransport``
(matching ``tests/test_fpl_client.py``/``tests/test_understat_client.py``'s convention); every
feature-engineering and fit/predict function is tested against small synthetic data.
"""

from __future__ import annotations

import httpx
import numpy as np
import pandas as pd
import pytest

from backtest.harness import run_walk_forward
from backtest.run_season import (
    _team_rate_asof_venue_split,
    build_penalty_attempts_frame,
    build_stand_in_squad_starting_xi,
    collapse_double_gameweeks,
    compute_days_since_last_appearance,
    compute_fixture_congestion,
    compute_team_rotation_propensity,
    compute_zero_minute_streak_length,
    engineer_features,
    fetch_understat_player_histories,
    fetch_vaastav_merged_gw,
    fit_fn,
    make_predict_fn,
    make_simulate_predict_fn,
    score_season,
    season_label,
    simulate_gameweek_pool,
)
from engine.data.crosswalk import CrosswalkEntry


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


def test_collapse_double_gameweeks_sums_outcomes_and_maxes_boolean_flags():
    # Same player, same gameweek, two fixtures (a double gameweek) -- earlier kickoff first.
    gw = _gw_frame(
        [
            {
                "player_id": 1,
                "gameweek": 5,
                "kickoff_time": "2025-09-01",
                "total_points": 2,
                "minutes": 90,
                "starts": 1,
                "clean_sheets": 1,
                "recent_start_rate": 0.9,  # a point-in-time feature: first fixture wins
            },
            {
                "player_id": 1,
                "gameweek": 5,
                "kickoff_time": "2025-09-04",
                "total_points": 5,
                "minutes": 90,
                "starts": 1,
                "clean_sheets": 0,
                "recent_start_rate": 0.5,
            },
            {
                "player_id": 2,
                "gameweek": 5,
                "kickoff_time": "2025-09-02",
                "total_points": 3,
                "minutes": 90,
                "starts": 1,
                "clean_sheets": 0,
                "recent_start_rate": 0.7,
            },
        ]
    )

    collapsed, n_removed = collapse_double_gameweeks(gw)

    assert n_removed == 1  # one extra fixture row merged away
    assert collapsed.groupby(["player_id", "gameweek"]).size().max() == 1
    player_1 = collapsed[collapsed["player_id"] == 1].iloc[0]
    assert player_1["total_points"] == 7  # summed across both fixtures
    assert player_1["minutes"] == 180  # summed, a known/accepted DGW simplification
    assert player_1["starts"] == 1  # max, not summed -- a boolean "did they start" flag
    assert player_1["clean_sheets"] == 1  # max -- kept a clean sheet in at least one fixture
    assert player_1["recent_start_rate"] == pytest.approx(0.9)  # first fixture by kickoff time


def test_fetch_understat_player_histories_keeps_prior_seasons_drops_future_ones(tmp_path):
    # ENGINE_IMPROVEMENTS_2.md C.3: a cached history file spanning three seasons must retain the
    # prior-season rows (2023, 2024) alongside the target season (2025) -- rates.py's own EWMA
    # design explicitly wants this for the cold-start problem -- while still dropping any row from
    # a season *after* season_start_year, in case a cached file is reused across driver runs.
    cache_path = tmp_path / "understat" / "2025" / "players" / "42.parquet"
    cache_path.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "season": ["2023", "2024", "2025", "2026"],
            "date": ["2023-09-01", "2024-09-01", "2025-09-01", "2026-09-01"],
            "time": [90, 90, 90, 90],
            "npxG": [0.3, 0.4, 0.5, 0.6],
            "xA": [0.1, 0.1, 0.1, 0.1],
            "goals": [0, 0, 0, 0],
            "npg": [0, 0, 0, 0],
        }
    ).to_parquet(cache_path, index=False)

    crosswalk = [CrosswalkEntry(fpl_id=1, understat_id=42, fpl_name="X", understat_name="X", matched_by="exact")]

    histories = fetch_understat_player_histories(crosswalk, 2025, tmp_path, understat=None)

    seasons_kept = sorted(int(s) for s in histories[1]["season"])
    assert seasons_kept == [2023, 2024, 2025]  # 2026 (the future) is dropped
    assert list(histories[1]["date"]) == sorted(histories[1]["date"])  # chronologically sorted


def test_team_rate_asof_venue_split_uses_only_matching_venue_history():
    # ENGINE_IMPROVEMENTS_2.md D.5: a team that only ever scores at home in its history must have
    # its home-fixture rate reflect that, not be diluted by away matches with zero goals.
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2025-08-01", "2025-08-08", "2025-08-15", "2025-08-22"], utc=True
            ),
            "xG": [2.0, 0.0, 2.0, 0.0],
            "xGA": [0.0, 0.0, 0.0, 0.0],
            "minutes": 90.0,
            "is_home": [True, False, True, False],
        }
    )
    before = pd.Timestamp("2025-09-01", tz="UTC")

    home_rate = _team_rate_asof_venue_split(history, "xG", before, is_home=True)
    away_rate = _team_rate_asof_venue_split(history, "xG", before, is_home=False)

    assert home_rate == pytest.approx(2.0)
    assert away_rate == pytest.approx(0.0)


def test_team_rate_asof_venue_split_falls_back_to_combined_when_venue_subset_empty():
    # A promoted team's very first home fixture has no prior home-only history at all -- must fall
    # back to the combined rate rather than returning NaN.
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-08-08", "2025-08-15"], utc=True),
            "xG": [1.0, 1.4],
            "xGA": [0.5, 0.5],
            "minutes": 90.0,
            "is_home": [False, False],  # no home matches yet
        }
    )
    before = pd.Timestamp("2025-09-01", tz="UTC")

    rate = _team_rate_asof_venue_split(history, "xG", before, is_home=True)

    assert rate == pytest.approx((1.0 + 1.4) / 2, abs=0.2)  # ~ the combined EWMA, not NaN


def test_build_stand_in_squad_starting_xi_picks_highest_minutes_players_who_started():
    ground_truth = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "gameweek": [1, 1, 1, 1, 1],
            "minutes": [90, 90, 90, 90, 0],
            "starts": [1, 1, 1, 0, 0],  # player 4 was an unused sub; player 5 didn't play
        }
    )
    # Season totals (across a second gameweek) determine squad membership independently of GW1.
    totals = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4, 5],
            "gameweek": [2, 2, 2, 2, 2],
            "minutes": [90, 90, 90, 90, 90],
            "starts": [1, 1, 1, 1, 1],
        }
    )
    full = pd.concat([ground_truth, totals], ignore_index=True)

    starting_xi = build_stand_in_squad_starting_xi(full, squad_size=4)

    assert 1 in starting_xi and 2 in starting_xi[1] and 3 in starting_xi[1]
    assert 4 not in starting_xi[1]  # in the squad (top-4 minutes) but didn't start GW1
    assert 5 not in starting_xi.get(1, set())  # not even in the top-4-minutes squad
    assert starting_xi[2] == {1, 2, 3, 4}  # all four squad members started GW2


def test_score_season_computes_captaincy_when_starts_column_present():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=1)

    ground_truth = engineered[
        [
            "player_id",
            "gameweek",
            "position",
            "total_points",
            "minutes",
            "clean_sheets",
            "defensive_contribution",
            "goals_scored",
            "assists",
            "bonus",
            "starts",
        ]
    ]
    report = score_season(result.predictions, ground_truth)

    assert report.captaincy is not None
    assert 0.0 <= report.captaincy.raw_hit_rate <= 1.0
    assert "Captaincy hit-rate" in report.summary()


def test_simulate_gameweek_pool_produces_valid_floor_ceiling_for_a_real_fixture():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    gameweeks = sorted(engineered["gameweek"].unique())
    training_history = engineered[engineered["gameweek"] < gameweeks[-1]]
    fitted_state = fit_fn(training_history)
    players_gw = engineered[engineered["gameweek"] == gameweeks[-1]]

    result = simulate_gameweek_pool(players_gw, fitted_state, n_runs=200, seed=0)

    assert set(result.columns) == {
        "player_id",
        "gameweek",
        "sim_mean",
        "sim_median",
        "floor",
        "ceiling",
        "prob_big_haul",
    }
    assert set(result["player_id"]) == set(players_gw["player_id"])
    assert (result["floor"] <= result["ceiling"]).all()
    assert result["prob_big_haul"].between(0.0, 1.0).all()
    # every real fixture in this synthetic season pairs exactly two teams -- both sides scored.
    assert (result["gameweek"] == gameweeks[-1]).all()


def test_simulate_gameweek_pool_empty_pool_returns_empty_frame():
    result = simulate_gameweek_pool(pd.DataFrame(), fitted_state=None, n_runs=10)
    assert result.empty
    assert list(result.columns) == [
        "player_id",
        "gameweek",
        "sim_mean",
        "sim_median",
        "floor",
        "ceiling",
        "prob_big_haul",
    ]


def test_run_walk_forward_with_simulate_predict_fn_end_to_end():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    gameweeks = sorted(engineered["gameweek"].unique())
    simulate_fn = make_simulate_predict_fn(engineered, n_runs=50, seed=0)

    result = run_walk_forward(gameweeks, engineered, fit_fn, simulate_fn, min_training_gameweeks=1)

    assert not result.predictions.empty
    assert (result.predictions["floor"] <= result.predictions["ceiling"]).all()


def test_score_season_scores_floor_ceiling_and_big_haul_when_simulation_predictions_given():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=1)
    simulate_fn = make_simulate_predict_fn(engineered, n_runs=50, seed=0)
    sim_result = run_walk_forward(gameweeks, engineered, fit_fn, simulate_fn, min_training_gameweeks=1)

    ground_truth = engineered[
        [
            "player_id",
            "gameweek",
            "position",
            "total_points",
            "minutes",
            "clean_sheets",
            "defensive_contribution",
            "goals_scored",
            "assists",
            "bonus",
        ]
    ]
    report = score_season(
        result.predictions, ground_truth, simulation_predictions=sim_result.predictions
    )

    assert report.floor_ceiling_coverage is not None
    assert 0.0 <= report.floor_ceiling_coverage <= 1.0
    assert report.big_haul_calibration is not None
    assert "Simulation layer" in report.summary()


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
                    "goals_scored": 0,
                    "assists": 0,
                    "value": 50 + player_id,
                    "selected": 10000 * player_id,
                    "transfers_in": 100,
                    "transfers_out": 50,
                    "transfers_balance": 50,
                    "clean_sheets": (
                        1
                        if (team == "Team A" and b_score == 0)
                        or (team == "Team B" and a_score == 0)
                        else 0
                    ),
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
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
                "is_home": is_home_seq,
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


def test_engineer_features_computes_crowd_features_from_value_selected_transfers():
    # ENGINE_IMPROVEMENTS_2.md B.3: price/ownership/transfer-flow features, computed from archive
    # columns that were previously fetched but never used by the minutes model.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)

    row = engineered.iloc[0]
    raw = merged_gw[
        (merged_gw["element"] == row["player_id"]) & (merged_gw["GW"] == row["gameweek"])
    ].iloc[0]

    assert row["price"] == pytest.approx(float(raw["value"]))
    assert row["ownership_log"] == pytest.approx(np.log1p(float(raw["selected"])))
    assert row["transfers_out_share"] == pytest.approx(
        float(raw["transfers_out"]) / max(float(raw["selected"]), 1.0)
    )
    assert row["transfers_balance_share"] == pytest.approx(
        float(raw["transfers_balance"]) / max(float(raw["selected"]), 1.0)
    )
    for col in ["price", "ownership_log", "transfers_out_share", "transfers_balance_share"]:
        assert engineered[col].notna().all()


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
        [
            "player_id",
            "gameweek",
            "position",
            "total_points",
            "minutes",
            "clean_sheets",
            "defensive_contribution",
            "goals_scored",
            "assists",
            "bonus",
        ]
    ]
    report = score_season(result.predictions, ground_truth)

    summary = report.summary()
    assert "Overall MAE" in summary
    assert isinstance(report.accuracy.overall_mae, float)
    assert isinstance(report.minutes_diagnostics.auc_played_at_all, float)
    assert "goals" in report.mean_calibrations
    assert "assists" in report.mean_calibrations
    assert "bonus" in report.mean_calibrations


def test_score_season_includes_pure_xg_baseline_when_player_rates_given():
    # ENGINE_IMPROVEMENTS_2.md D.2: pure_xg is the third baseline BUILD_PLAN 3.3 names but which
    # was never actually wired into the gate.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=1)

    ground_truth = engineered[
        [
            "player_id",
            "gameweek",
            "position",
            "total_points",
            "minutes",
            "clean_sheets",
            "defensive_contribution",
            "goals_scored",
            "assists",
            "bonus",
        ]
    ]
    player_rates = engineered[
        ["player_id", "position", "gameweek", "npxg_per_90", "xa_per_90", "recent_minutes_ewma"]
    ]
    report = score_season(result.predictions, ground_truth, player_rates=player_rates)

    assert "pure_xg" in report.baseline_results
    assert "pure_xg" in report.definition_of_done.baseline_results
