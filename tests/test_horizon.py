"""Tests for engine/horizon.py -- multi-gameweek horizon projections built from one fixed
fitted engine state."""

from __future__ import annotations

import pandas as pd

from backtest.run_season import engineer_features, fit_fn
from engine.horizon import build_horizon_predictions, build_horizon_projections


def _synthetic_season() -> pd.DataFrame:
    """Two teams, four players, seven gameweeks -- enough for `fit_fn` to have real prior history
    and for a multi-gameweek horizon to have several real future gameweeks to project."""
    teams = pd.DataFrame({"id": [1, 2], "name": ["Team A", "Team B"]})
    kickoffs = pd.date_range("2025-08-16", periods=7, freq="7D", tz="UTC")
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

    def _team_history(is_home_seq: list[bool]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": kickoffs,
                "xG": [1.5 if h else 0.8 for h in is_home_seq],
                "xGA": [0.7 if h else 1.4 for h in is_home_seq],
                "minutes": 90.0,
                "is_home": is_home_seq,
            }
        )

    a_home_seq = [i % 2 == 0 for i in range(7)]
    team_histories = {
        "Team A": _team_history(a_home_seq),
        "Team B": _team_history([not h for h in a_home_seq]),
    }

    def _player_history(base_npxg: float, base_xa: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "date": kickoffs,
                "npxG": [base_npxg] * 7,
                "xA": [base_xa] * 7,
                "goals": [0] * 7,
                "npg": [0] * 7,
                "time": [90] * 7,
                "season": ["2025"] * 7,
            }
        )

    player_histories = {
        1: _player_history(0.3, 0.2),
        2: _player_history(0.05, 0.05),
        3: _player_history(0.25, 0.15),
        4: _player_history(0.4, 0.1),
    }
    return engineer_features(merged_gw, teams, team_histories, player_histories)


def test_build_horizon_predictions_covers_every_requested_gameweek():
    engineered = _synthetic_season()
    training_history = engineered[engineered["gameweek"] < 5]
    fitted_state = fit_fn(training_history)
    horizon_gameweeks = [5, 6, 7]

    predictions = build_horizon_predictions(
        engineered, fitted_state, horizon_gameweeks, n_simulation_runs=10, seed=1
    )

    assert set(predictions["gameweek"].unique()) == set(horizon_gameweeks)
    assert {"floor", "ceiling", "prob_big_haul"}.issubset(predictions.columns)


def test_build_horizon_predictions_reuses_the_same_fitted_state_across_gameweeks():
    # Same fitted_state, same player -> the position/team-independent constants used shouldn't
    # change between horizon gameweeks (only the fixture-dependent inputs should vary), so this
    # must not raise and must return one row per player per gameweek.
    engineered = _synthetic_season()
    training_history = engineered[engineered["gameweek"] < 5]
    fitted_state = fit_fn(training_history)
    predictions = build_horizon_predictions(
        engineered, fitted_state, [5, 6], n_simulation_runs=10, seed=1
    )
    counts = predictions.groupby("gameweek")["player_id"].nunique()
    assert (counts == counts.iloc[0]).all()


def test_build_horizon_projections_merges_every_gameweek_per_player():
    engineered = _synthetic_season()
    training_history = engineered[engineered["gameweek"] < 5]
    fitted_state = fit_fn(training_history)
    predictions = build_horizon_predictions(
        engineered, fitted_state, [5, 6, 7], n_simulation_runs=10, seed=1
    )

    projections = build_horizon_projections(predictions)

    for horizon in projections.values():
        assert set(horizon.gameweeks) == {5, 6, 7}
        assert horizon.horizon_total_points == sum(
            p.expected_points for p in horizon.gameweeks.values()
        )


def test_build_horizon_predictions_empty_horizon_returns_empty_frame():
    engineered = _synthetic_season()
    training_history = engineered[engineered["gameweek"] < 5]
    fitted_state = fit_fn(training_history)
    predictions = build_horizon_predictions(
        engineered, fitted_state, [999], n_simulation_runs=10, seed=1
    )
    assert predictions.empty
