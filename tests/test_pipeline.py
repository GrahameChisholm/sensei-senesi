"""Tests for the full-player-pool per-gameweek orchestrator (engine/pipeline.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.bonus import BonusModel, build_features
from engine.models.minutes import MinutesModel, encode_status
from engine.pipeline import project_gameweek_pool


@pytest.fixture(scope="module")
def fitted_minutes_model() -> MinutesModel:
    rng = np.random.default_rng(0)
    n = 200
    features = pd.DataFrame(
        {
            "recent_start_rate": rng.uniform(0.3, 1.0, n),
            "recent_minutes_ewma": rng.uniform(40, 90, n),
            "fixture_congestion": rng.integers(0, 3, n).astype(float),
            "chance_of_playing_next_round": np.full(n, 100.0),
            "status_score": np.full(n, encode_status("a")),
        }
    )
    started = pd.Series(rng.choice([0, 1], size=n, p=[0.2, 0.8]))
    minutes = pd.Series(np.where(started == 1, rng.choice([90, 75, 45], size=n), 0.0))
    return MinutesModel().fit(features, started, minutes)


@pytest.fixture(scope="module")
def fitted_bonus_model() -> BonusModel:
    rng = np.random.default_rng(1)
    rows, targets = [], []
    for _ in range(150):
        position = rng.choice(["GK", "DEF", "MID", "FWD"])
        eg, ea, cs, dc = rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 1), rng.uniform(0, 15)
        rows.append(build_features(eg, ea, cs, dc, position))
        targets.append(float(np.clip(1.5 * eg + ea + cs, 0, 3)))
    return BonusModel().fit(pd.DataFrame(rows), pd.Series(targets))


def _base_row(player_id: int, position: str) -> dict:
    return {
        "player_id": player_id,
        "position": position,
        "recent_start_rate": 0.9,
        "recent_minutes_ewma": 85.0,
        "fixture_congestion": 0.0,
        "chance_of_playing_next_round": 100.0,
        "status_score": encode_status("a"),
        "npxg_per_90": 0.4,
        "xa_per_90": 0.2,
        "team_xg_per_90": 1.5,
        "team_xga_per_90": 1.1,
        "opponent_xg_per_90": 1.3,
        "opponent_xga_per_90": 1.6,
        "league_avg_xga_per_90": 1.4,
        "yellow_card_rate_per_90": 0.15,
        "red_card_rate_per_90": 0.01,
        "dc_per_90": 6.0,
        "opponent_possession_share": 0.5,
        "opponent_shots_on_target_per_90": 4.0,
        "is_home": True,
    }


@pytest.fixture
def synthetic_pool() -> pd.DataFrame:
    rows = [
        _base_row(1, "GK"),
        _base_row(2, "DEF"),
        _base_row(3, "DEF"),
        _base_row(4, "MID"),
        _base_row(5, "MID"),
        _base_row(6, "FWD"),
    ]
    return pd.DataFrame(rows)


def test_project_gameweek_pool_returns_one_row_per_player(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    predictions = project_gameweek_pool(synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model)

    assert len(predictions) == len(synthetic_pool)
    assert set(predictions["player_id"]) == set(synthetic_pool["player_id"])
    assert (predictions["gameweek"] == 1).all()
    assert predictions["expected_points"].apply(np.isfinite).all()


def test_goalkeeper_gets_saves_not_defensive_contribution(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    predictions = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")

    assert predictions.loc[1, "saves"] > 0  # GK
    assert predictions.loc[1, "defensive_contribution"] == 0.0
    assert predictions.loc[2, "saves"] == 0.0  # DEF
    assert predictions.loc[2, "defensive_contribution"] > 0.0


def test_component_breakdown_sums_to_expected_points(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    predictions = project_gameweek_pool(synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model)

    component_cols = [
        "appearance",
        "goals",
        "assists",
        "clean_sheet",
        "goals_conceded",
        "defensive_contribution",
        "saves",
        "bonus",
        "cards",
        "penalty_misses",
    ]
    summed = predictions[component_cols].sum(axis=1)
    pd.testing.assert_series_equal(
        summed, predictions["expected_points"], check_names=False, atol=1e-9
    )


def test_clean_sheet_probability_in_valid_range(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    predictions = project_gameweek_pool(synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model)

    assert predictions["clean_sheet_probability"].between(0.0, 1.0).all()


def test_empty_pool_raises(fitted_minutes_model, fitted_bonus_model):
    with pytest.raises(ValueError):
        project_gameweek_pool(pd.DataFrame(), 1, fitted_minutes_model, fitted_bonus_model)
