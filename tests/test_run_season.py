"""Tests for backtest/run_season.py — the versioned real-backtest driver.

No real network calls: data-prep caching is tested against an injected ``httpx.MockTransport``
(matching ``tests/test_fpl_client.py``/``tests/test_understat_client.py``'s convention); every
feature-engineering and fit/predict function is tested against small synthetic data.
"""

from __future__ import annotations

import json

import httpx
import numpy as np
import pandas as pd
import pytest

from backtest.harness import run_walk_forward
from backtest.run_season import (
    SeasonBacktestData,
    _composite_gameweek,
    _fit_league_avg_rate_by_position,
    _league_venue_multipliers,
    _pool_season_backtest_data,
    _team_prior_match_count,
    _team_rate_asof,
    _team_rate_asof_shrunk,
    _team_venue_multipliers,
    build_penalty_attempts_frame,
    build_stand_in_squad_starting_xi,
    collapse_double_gameweeks,
    compute_coverage_report,
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
from engine.models.minutes import encode_status


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

    crosswalk = [
        CrosswalkEntry(
            fpl_id=1, understat_id=42, fpl_name="X", understat_name="X", matched_by="exact"
        )
    ]

    histories = fetch_understat_player_histories(crosswalk, 2025, tmp_path, understat=None)

    seasons_kept = sorted(int(s) for s in histories[1]["season"])
    assert seasons_kept == [2023, 2024, 2025]  # 2026 (the future) is dropped
    assert list(histories[1]["date"]) == sorted(histories[1]["date"])  # chronologically sorted


def test_team_prior_match_count_counts_strictly_prior_matches():
    history = pd.DataFrame(
        {"date": pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-15"], utc=True)}
    )
    assert _team_prior_match_count(history, pd.Timestamp("2025-08-10", tz="UTC")) == 2
    assert _team_prior_match_count(history, pd.Timestamp("2025-07-01", tz="UTC")) == 0


def test_team_prior_match_count_empty_history_is_zero():
    empty = pd.DataFrame(columns=["date"])
    assert _team_prior_match_count(empty, pd.Timestamp("2025-08-10", tz="UTC")) == 0


def test_team_rate_asof_shrunk_pulls_a_thin_sample_toward_the_league_average():
    # ENGINE_IMPROVEMENTS_3.md A.1: one match of a wildly unusual rate should not be taken at face
    # value the way the previous per-team venue split effectively did.
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-08-01"], utc=True),
            "xG": [4.0],
            "xGA": [4.0],
            "minutes": [90.0],
            "is_home": [True],
        }
    )
    before = pd.Timestamp("2025-08-15", tz="UTC")
    raw = _team_rate_asof(history, "xG", before)
    assert raw == pytest.approx(4.0)

    shrunk = _team_rate_asof_shrunk(history, "xG", before, league_avg=1.4, shrinkage_k=4.0)
    # n_prior=1, k=4 -> weight mostly on the prior: (1*4.0 + 4*1.4) / 5 = 1.92
    assert shrunk == pytest.approx((1 * 4.0 + 4 * 1.4) / 5)
    assert 1.4 < shrunk < 4.0


def test_team_rate_asof_shrunk_barely_moves_a_deep_sample():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime([f"2025-{m:02d}-01" for m in range(1, 13)], utc=True),
            "xG": [1.5] * 12,
            "xGA": [1.5] * 12,
            "minutes": [90.0] * 12,
            "is_home": [True] * 12,
        }
    )
    before = pd.Timestamp("2026-01-01", tz="UTC")
    shrunk = _team_rate_asof_shrunk(history, "xG", before, league_avg=1.0, shrinkage_k=4.0)
    # n_prior=12, k=4 -> (12*1.5 + 4*1.0) / 16 = 1.375: most of the weight stays on the
    # individual rate, unlike the single-match case above where the prior dominates.
    assert shrunk == pytest.approx((12 * 1.5 + 4 * 1.0) / 16)
    assert shrunk > 1.375 - 1e-9 and shrunk < 1.5


def test_team_rate_asof_shrunk_returns_nan_when_no_prior_history():
    empty = pd.DataFrame(columns=["date", "xG", "xGA", "minutes", "is_home"])
    before = pd.Timestamp("2025-08-15", tz="UTC")
    assert pd.isna(_team_rate_asof_shrunk(empty, "xG", before, league_avg=1.4))


def test_league_venue_multipliers_reflects_a_real_home_advantage():
    # Every team scores 2.0 at home and 1.0 away -- a real, sizeable, league-wide venue effect.
    rows = []
    for _team_id in range(20):
        for _match in range(5):
            rows.append(
                {
                    "date": pd.Timestamp("2025-08-01", tz="UTC"),
                    "xG": 2.0,
                    "xGA": 1.0,
                    "is_home": True,
                }
            )
            rows.append(
                {
                    "date": pd.Timestamp("2025-08-01", tz="UTC"),
                    "xG": 1.0,
                    "xGA": 2.0,
                    "is_home": False,
                }
            )
    team_histories = {str(i): pd.DataFrame(rows) for i in range(20)}
    xg_mult, xga_mult = _league_venue_multipliers(
        team_histories, pd.Timestamp("2025-09-01", tz="UTC")
    )
    assert xg_mult == pytest.approx(2.0 / 1.5)
    assert xga_mult == pytest.approx(1.0 / 1.5)


def test_league_venue_multipliers_neutral_below_the_minimum_match_count():
    rows = [{"date": pd.Timestamp("2025-08-01", tz="UTC"), "xG": 2.0, "xGA": 1.0, "is_home": True}]
    team_histories = {"1": pd.DataFrame(rows)}
    xg_mult, xga_mult = _league_venue_multipliers(
        team_histories, pd.Timestamp("2025-08-15", tz="UTC")
    )
    assert (xg_mult, xga_mult) == (1.0, 1.0)


def test_league_venue_multipliers_neutral_when_no_team_histories():
    xg_mult, xga_mult = _league_venue_multipliers({}, pd.Timestamp("2025-08-15", tz="UTC"))
    assert (xg_mult, xga_mult) == (1.0, 1.0)


def _venue_history(n_home: int, home_xg: float, away_xg: float) -> pd.DataFrame:
    rows = []
    for _match in range(n_home):
        rows.append(
            {
                "date": pd.Timestamp("2025-08-01", tz="UTC"),
                "xG": home_xg,
                "xGA": 1.0,
                "is_home": True,
            }
        )
        rows.append(
            {
                "date": pd.Timestamp("2025-08-01", tz="UTC"),
                "xG": away_xg,
                "xGA": 1.0,
                "is_home": False,
            }
        )
    return pd.DataFrame(rows)


def test_team_venue_multipliers_pulls_a_thin_sample_toward_the_league_multiplier():
    # B1: a team with only 2 home matches showing an extreme home advantage should end up much
    # closer to the league-wide multiplier than to its own noisy estimate.
    history = _venue_history(n_home=2, home_xg=3.0, away_xg=1.0)  # own raw xg_mult = 3.0/2.0 = 1.5
    before = pd.Timestamp("2025-09-01", tz="UTC")

    xg_mult, _ = _team_venue_multipliers(history, before, (1.1, 0.9), shrinkage_k=10.0)

    # n=2 against k=10 -> (2*1.5 + 10*1.1) / 12 = 1.1667, i.e. a sixth of the way toward its own.
    assert xg_mult == pytest.approx((2 * 1.5 + 10 * 1.1) / 12)
    assert abs(xg_mult - 1.1) < abs(xg_mult - 1.5)


def test_team_venue_multipliers_trusts_a_deep_sample():
    history = _venue_history(n_home=40, home_xg=3.0, away_xg=1.0)
    before = pd.Timestamp("2025-09-01", tz="UTC")

    xg_mult, _ = _team_venue_multipliers(history, before, (1.1, 0.9), shrinkage_k=10.0)

    assert xg_mult == pytest.approx((40 * 1.5 + 10 * 1.1) / 50)
    assert abs(xg_mult - 1.5) < abs(xg_mult - 1.1)


def test_team_venue_multipliers_falls_back_to_the_league_pair_without_both_venues():
    before = pd.Timestamp("2025-09-01", tz="UTC")
    league = (1.1, 0.9)

    assert _team_venue_multipliers(pd.DataFrame(), before, league) == league
    home_only = pd.DataFrame(
        [{"date": pd.Timestamp("2025-08-01", tz="UTC"), "xG": 2.0, "xGA": 1.0, "is_home": True}]
    )
    # Every prior match at home -- no away baseline to form a ratio against.
    assert _team_venue_multipliers(home_only, before, league) == league


def test_fit_league_avg_rate_by_position_computes_per_90_rate():
    training_history = pd.DataFrame(
        {
            "position": ["DEF"] * 120 + ["MID"] * 120,
            "minutes": [90.0] * 120 + [90.0] * 120,
            "yellow_cards": [1] * 12 + [0] * 108 + [0] * 120,  # 12 DEF yellows, 0 MID
        }
    )
    rates = _fit_league_avg_rate_by_position(training_history, "yellow_cards")
    # DEF: 12 yellows / (120*90) minutes * 90 = 12/120 = 0.1 per 90
    assert rates["DEF"] == pytest.approx(0.1)
    assert rates["MID"] == pytest.approx(0.0)
    assert rates["FWD"] == pytest.approx(0.0)  # no FWD rows at all -> too-thin fallback


def test_fit_league_avg_rate_by_position_falls_back_to_zero_for_thin_sample():
    training_history = pd.DataFrame(
        {"position": ["DEF"] * 5, "minutes": [90.0] * 5, "red_cards": [1, 0, 0, 0, 0]}
    )
    rates = _fit_league_avg_rate_by_position(training_history, "red_cards", min_rows=100)
    assert rates["DEF"] == pytest.approx(0.0)


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

    # No position column here, so this exercises the flat top-`squad_size` fallback path.
    starting_xi = build_stand_in_squad_starting_xi(full, squad_size=4, selection_col="minutes")

    assert 1 in starting_xi and 2 in starting_xi[1] and 3 in starting_xi[1]
    assert 4 not in starting_xi[1]  # in the squad (top-4 minutes) but didn't start GW1
    assert 5 not in starting_xi.get(1, set())  # not even in the top-4-minutes squad
    assert starting_xi[2] == {1, 2, 3, 4}  # all four squad members started GW2


def test_build_stand_in_squad_starting_xi_excludes_goalkeepers():
    # ENGINE_IMPROVEMENTS_3.md D.1: a goalkeeper who's never rotated would otherwise dominate a
    # pure highest-minutes selection -- no real manager captains their keeper.
    ground_truth = pd.DataFrame(
        {
            "player_id": [1, 2, 3, 4],
            "gameweek": [1, 1, 1, 1],
            "position": ["GK", "DEF", "MID", "FWD"],
            "minutes": [90, 90, 90, 45],  # GK has the most minutes of anyone
            "starts": [1, 1, 1, 1],
        }
    )
    starting_xi = build_stand_in_squad_starting_xi(
        ground_truth, squad_size=3, selection_col="minutes", shape={"DEF": 1, "MID": 1, "FWD": 1}
    )
    assert 1 not in starting_xi.get(1, set())
    assert starting_xi[1] == {2, 3, 4}


def test_build_stand_in_squad_starting_xi_respects_positional_shape_and_picks_by_points():
    # Nine outfield players; defenders play the most minutes but score the fewest points -- the
    # exact real-data pattern that made a pure minutes sort return a defender-heavy squad no
    # manager would field. The shape must cap defenders at 1 and take the best scorer per position.
    rows = []
    for player_id, position, minutes, points in [
        (1, "DEF", 90, 2),
        (2, "DEF", 90, 3),
        (3, "DEF", 90, 1),
        (4, "MID", 60, 9),
        (5, "MID", 60, 4),
        (6, "MID", 60, 7),
        (7, "FWD", 45, 12),
        (8, "FWD", 45, 5),
        (9, "GK", 90, 20),  # highest scorer overall, but a keeper -- must never be selected
    ]:
        rows.append(
            {
                "player_id": player_id,
                "gameweek": 1,
                "position": position,
                "minutes": minutes,
                "total_points": points,
                "starts": 1,
            }
        )
    ground_truth = pd.DataFrame(rows)

    starting_xi = build_stand_in_squad_starting_xi(
        ground_truth, shape={"DEF": 1, "MID": 2, "FWD": 1}
    )

    # Best DEF by points (2), best two MID (4, 6), best FWD (7). No goalkeeper.
    assert starting_xi[1] == {2, 4, 6, 7}


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
    sim_result = run_walk_forward(
        gameweeks, engineered, fit_fn, simulate_fn, min_training_gameweeks=1
    )

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
                    "name": f"Player {player_id}",
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
                    "saves": 0,
                    "bps": 10 + player_id,
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


def test_compute_coverage_report_surfaces_unmatched_significant_players():
    # Crosswalk coverage Phase 1: points_excluded_share was previously only a percentage -- there
    # was no way to see *which* players it referred to without a fresh ad hoc script.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    # _synthetic_season gives everyone exactly 450 minutes (5 gameweeks x 90) -- bump one
    # gameweek's minutes so every player clears the strict ">450" significance bar.
    merged_gw = merged_gw.copy()
    merged_gw.loc[merged_gw["GW"] == 1, "minutes"] = 120
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    # Player 1 (>450 minutes) matched; players 2-4 unmatched.
    crosswalk = [
        CrosswalkEntry(
            fpl_id=1, understat_id=101, fpl_name="P1", understat_name="P1", matched_by="exact"
        )
    ]

    report = compute_coverage_report(merged_gw, crosswalk, engineered)

    unmatched_ids = {p.fpl_id for p in report.unmatched_significant_players}
    assert 1 not in unmatched_ids
    assert {2, 3, 4}.issubset(unmatched_ids)
    for p in report.unmatched_significant_players:
        assert p.minutes > 450
        assert p.name  # real player name, not blank
    assert "Unmatched significant players" in report.summary()


def test_drop_reasons_partition_the_dropped_rows_and_name_the_cold_start():
    # B4: the dropna removes ~46% of raw rows in the real backtest, and the total alone can't
    # distinguish "harmless structural cold start" from "quietly filtered toward established
    # players". Reasons must partition the dropped rows exactly (no row counted twice, none
    # unaccounted for) or the decomposition can't be reasoned about.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)

    reasons = pd.DataFrame(engineered.attrs["drop_reasons"])
    n_dropped = engineered.attrs["n_rows_dropped_for_missing_features"]

    assert int(reasons["n_rows"].sum()) == n_dropped
    # Everyone's gameweek 1 is a first appearance, so that reason must be non-empty and must
    # account for those rows rather than being mislabelled as an unmatched crosswalk (whose
    # Understat columns are equally NaN on a first appearance -- hence the precedence).
    first_appearance = reasons[reasons["reason"] == "first_appearance"].iloc[0]
    n_players = merged_gw["element"].nunique()
    assert int(first_appearance["n_rows"]) == n_players
    assert 0.0 < float(first_appearance["points_share"]) < 1.0


def test_drop_reasons_separates_an_unmatched_player_from_the_cold_start():
    # A player with no Understat history is dropped for every gameweek, not just their first --
    # the later rows are what "unmatched_crosswalk" must isolate.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    unmatched_id = 2
    player_histories = {k: v for k, v in player_histories.items() if k != unmatched_id}

    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    reasons = pd.DataFrame(engineered.attrs["drop_reasons"])

    assert int(reasons["n_rows"].sum()) == engineered.attrs["n_rows_dropped_for_missing_features"]
    unmatched = reasons[reasons["reason"] == "unmatched_crosswalk"].iloc[0]
    # Every gameweek after the player's first, all of which would otherwise have survived.
    n_gws = merged_gw[merged_gw["element"] == unmatched_id]["GW"].nunique()
    assert int(unmatched["n_rows"]) == n_gws - 1
    assert unmatched_id not in set(engineered["player_id"])


def test_coverage_report_summary_includes_the_drop_reason_breakdown():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)

    report = compute_coverage_report(merged_gw, [], engineered)

    assert report.drop_reasons is not None
    summary = report.summary()
    assert "Why rows were dropped" in summary
    assert "first_appearance" in summary
    assert "points_share" in summary


def test_engineer_features_includes_goalkeepers_with_zeroed_npxg_xa():
    # ENGINE_IMPROVEMENTS_3.md D.1: goalkeepers are no longer excluded entirely; their npxG/xA
    # default to 0.0 (never meaningfully register in Understat, and the crosswalk doesn't try to
    # match them) rather than the NaN an unmatched *outfield* player correctly gets.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    gk_rows = merged_gw[merged_gw["element"] == 1].copy()
    gk_rows["element"] = 99
    gk_rows["position"] = "GK"
    gk_rows["saves"] = 3
    merged_gw = pd.concat([merged_gw, gk_rows], ignore_index=True)

    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)

    assert "GK" in set(engineered["position"])
    gk = engineered[engineered["player_id"] == 99]
    assert not gk.empty
    assert (gk["npxg_per_90"] == 0.0).all()
    assert (gk["xa_per_90"] == 0.0).all()
    assert gk["own_save_rate_per_90"].notna().all()
    assert (gk["own_save_rate_per_90"] > 0).any()  # picked up the real saves history


def test_engineer_features_live_availability_overrides_only_the_target_gameweek():
    # A1: chance_of_playing_next_round/status are real, live-only fields -- overriding them for
    # the current gameweek only (never a historical row, which must stay exactly as point-in-time
    # as the backtest's own default) is the whole point of the live_availability parameter.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    live_availability = pd.DataFrame(
        [{"player_id": 1, "chance_of_playing_next_round": 25.0, "status": "d"}]
    )

    engineered = engineer_features(
        merged_gw, teams, team_histories, player_histories, live_availability=live_availability
    )

    target_gameweek = engineered["gameweek"].max()
    player_1 = engineered[engineered["player_id"] == 1].set_index("gameweek")

    assert player_1.loc[target_gameweek, "chance_of_playing_next_round"] == pytest.approx(25.0)
    assert player_1.loc[target_gameweek, "status_score"] == encode_status("d")
    earlier_gameweeks = player_1.index[player_1.index != target_gameweek]
    assert (player_1.loc[earlier_gameweeks, "chance_of_playing_next_round"] == 100.0).all()
    assert (player_1.loc[earlier_gameweeks, "status_score"] == encode_status("a")).all()


def test_engineer_features_live_availability_leaves_unlisted_players_at_the_default():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    live_availability = pd.DataFrame(
        [{"player_id": 1, "chance_of_playing_next_round": 0.0, "status": "i"}]
    )

    engineered = engineer_features(
        merged_gw, teams, team_histories, player_histories, live_availability=live_availability
    )

    target_gameweek = engineered["gameweek"].max()
    others = engineered[
        (engineered["gameweek"] == target_gameweek) & (engineered["player_id"] != 1)
    ]
    assert (others["chance_of_playing_next_round"] == 100.0).all()
    assert (others["status_score"] == encode_status("a")).all()


def test_engineer_features_without_live_availability_is_unchanged():
    merged_gw, teams, team_histories, player_histories = _synthetic_season()

    with_none = engineer_features(merged_gw, teams, team_histories, player_histories)
    explicit_none = engineer_features(
        merged_gw, teams, team_histories, player_histories, live_availability=None
    )

    pd.testing.assert_frame_equal(with_none, explicit_none)


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
    # B3: the played-only figures the gate actually reads must also be present, and separately
    # from the all-rows ones -- ground_truth here carries "minutes".
    assert "goals" in report.mean_calibrations_played
    assert "played rows only" in summary


def test_season_report_headline_summary_is_json_serializable_and_complete():
    # A4: the Model Performance screen's actual data source -- must round-trip through json.dumps
    # cleanly (no DataFrames/numpy scalars leaking through) and carry every field the screen needs.
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

    headline = report.headline_summary()
    reloaded = json.loads(json.dumps(headline))  # raises on anything non-JSON-serializable

    assert isinstance(reloaded["overall_mae"], float)
    assert isinstance(reloaded["overall_rmse"], float)
    assert isinstance(reloaded["pooled_spearman"], float)
    assert set(reloaded["top_n_mean_actual"]) == {"1", "5", "10", "20"}  # JSON keys are strings
    assert "goals" in reloaded["mean_calibrations_played"]
    assert reloaded["mean_calibrations_played"]["goals"].keys() == {
        "predicted",
        "actual",
        "relative_gap",
    }
    assert reloaded["captaincy_hit_rate"] is not None
    assert set(reloaded["gate"]) == {
        "beats_baselines",
        "no_severe_bias",
        "calibration_acceptable",
        "predictions_logged",
        "trusted_by_user",
        "passed",
    }


def test_score_season_played_only_mean_calibration_differs_from_all_rows_and_feeds_the_gate():
    # B3: on a synthetic pool with real zero-minute rows, the all-rows and played-only goals
    # calibration must differ -- if they're identical the played-row filter isn't doing anything.
    # This also confirms the gate's mean_calibration_reports actually receives the played variant.
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
    ].copy()
    # Everyone scores exactly one goal when they play, and (as in the real archive) zero minutes
    # means zero goals -- the predicted-side gap this creates for the all-rows figure comes purely
    # from the minutes model assigning some nonzero expected_goals to rows that then score zero,
    # which is exactly the effect the played-only filter is meant to strip out.
    ground_truth["goals_scored"] = 1
    zero_minute = ground_truth.index[::2]
    ground_truth.loc[zero_minute, "minutes"] = 0
    ground_truth.loc[zero_minute, "goals_scored"] = 0

    report = score_season(result.predictions, ground_truth)

    assert report.mean_calibrations["goals"].mean_actual < 1.0  # diluted by the zero-minute rows
    assert report.mean_calibrations_played["goals"].mean_actual == pytest.approx(1.0)
    assert report.definition_of_done.mean_calibration_reports == report.mean_calibrations_played


def test_score_season_computes_saves_mean_calibration_for_goalkeepers():
    # Phase 3: saves had no calibration check at all -- every other component (goals, assists,
    # bonus, clean sheets, DC) already had one.
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    gk_rows = merged_gw[merged_gw["element"] == 1].copy()
    gk_rows["element"] = 99
    gk_rows["position"] = "GK"
    gk_rows["saves"] = 3
    merged_gw = pd.concat([merged_gw, gk_rows], ignore_index=True)

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
            "saves",
        ]
    ]
    report = score_season(result.predictions, ground_truth)

    assert "saves" in report.mean_calibrations
    assert report.mean_calibrations["saves"].mean_predicted >= 0.0
    assert report.mean_calibrations["saves"].mean_actual == pytest.approx(3.0)


def test_score_season_computes_price_tier_bias_and_team_clean_sheet_when_columns_present():
    # ENGINE_IMPROVEMENTS_3.md B.2/B.3: both are computed only when ground_truth carries the
    # extra columns they need -- this is the full-column path production's own run_backtest uses.
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
            "value",
            "team",
            "was_home",
            "team_h_score",
            "team_a_score",
        ]
    ]
    report = score_season(result.predictions, ground_truth)

    assert report.bias_by_price_tier is not None
    assert "price_tier" in report.bias_by_price_tier.by_group.columns
    assert report.team_clean_sheet_calibration is not None
    assert "team_clean_sheet" in report.brier_reports
    assert "minutes_played_at_all" in report.brier_reports
    summary = report.summary()
    assert "Bias by price tier" in summary
    assert "Team-level clean-sheet MACE" in summary
    assert "Brier vs. predicting the constant base rate" in summary


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


def test_composite_gameweek_disambiguates_seasons():
    # Multi-season Phase 2: season 2024's GW5 and season 2025's GW5 must never collide once pooled.
    gws = pd.Series([1, 5, 38])
    assert list(_composite_gameweek(2024, gws)) == [202401, 202405, 202438]
    assert list(_composite_gameweek(2025, gws)) == [202501, 202505, 202538]


def _synthetic_season_backtest_data(
    season_start_year: int, dc_data_available: bool = True
) -> SeasonBacktestData:
    """A minimal, network-free SeasonBacktestData for one synthetic season (multi-season Phase 2
    pooling tests) -- reuses the same synthetic fixture and real fit/predict pipeline every other
    engineer_features/score_season test in this module does."""
    merged_gw, teams, team_histories, player_histories = _synthetic_season()
    if not dc_data_available:
        merged_gw = merged_gw.drop(columns=["defensive_contribution"])
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
            "value",
            "team",
            "was_home",
            "team_h_score",
            "team_a_score",
        ]
    ].copy()
    ground_truth["dc_data_available"] = bool(engineered.attrs.get("dc_data_available", True))
    player_rates = engineered[
        ["player_id", "position", "gameweek", "npxg_per_90", "xa_per_90", "recent_minutes_ewma"]
    ]
    coverage = compute_coverage_report(
        merged_gw, [CrosswalkEntry(1, 101, "P1", "P1", "exact")], engineered
    )
    return SeasonBacktestData(
        season_start_year=season_start_year,
        predictions=result.predictions,
        ground_truth=ground_truth,
        player_rates=player_rates,
        coverage=coverage,
    )


def test_pool_season_backtest_data_keeps_per_season_and_pools_across_seasons():
    per_season_data = {
        2024: _synthetic_season_backtest_data(2024),
        2025: _synthetic_season_backtest_data(2025),
    }

    report = _pool_season_backtest_data(per_season_data)

    assert set(report.per_season) == {2024, 2025}
    for season_report in report.per_season.values():
        assert isinstance(season_report.accuracy.overall_mae, float)
    # pooled sample is the concatenation of both seasons -- twice the single-season row count.
    single_season_n = sum(
        g["n"] for g in report.per_season[2024].accuracy.by_position.to_dict("records")
    )
    pooled_n = sum(g["n"] for g in report.pooled.accuracy.by_position.to_dict("records"))
    assert pooled_n == 2 * single_season_n
    assert isinstance(report.pooled.accuracy.overall_mae, float)


def test_pool_season_backtest_data_excludes_dc_unavailable_seasons_from_dc_calibration():
    # Multi-season Phase 2: a season lacking DC's raw archive columns gets a neutral placeholder
    # rate/outcome upstream, but that placeholder must never count toward DC calibration.
    per_season_data = {
        2024: _synthetic_season_backtest_data(2024, dc_data_available=False),
        2025: _synthetic_season_backtest_data(2025, dc_data_available=True),
    }

    report = _pool_season_backtest_data(per_season_data)

    # The pooled DC calibration bin counts must match season 2025 alone (2024's placeholder rows
    # are excluded), not the sum of both seasons.
    pooled_n = report.pooled.defensive_contribution_calibration.by_bin["n"].sum()
    season_2025_n = report.per_season[2025].defensive_contribution_calibration.by_bin["n"].sum()
    assert pooled_n == season_2025_n
