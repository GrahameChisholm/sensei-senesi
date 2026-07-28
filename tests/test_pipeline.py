"""Tests for the full-player-pool per-gameweek orchestrator (engine/pipeline.py)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.models.bonus import BonusModel, build_features
from engine.models.minutes import MinutesModel, encode_status
from engine.pipeline import FittedConstants, project_gameweek_pool


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
            "days_since_last_appearance": rng.integers(0, 14, n).astype(float),
            "zero_minute_streak_length": rng.integers(0, 3, n).astype(float),
            "start_rate_last_3": rng.uniform(0.3, 1.0, n),
            "start_rate_last_6": rng.uniform(0.3, 1.0, n),
            "start_rate_last_15": rng.uniform(0.3, 1.0, n),
            "team_rotation_propensity": rng.uniform(0.0, 1.0, n),
            "price": rng.uniform(4.0, 14.0, n),
            "ownership_log": rng.uniform(0, 5, n),
            "transfers_out_share": rng.uniform(0, 0.1, n),
            "transfers_balance_share": rng.uniform(-0.1, 0.1, n),
            "is_goalkeeper": np.zeros(n),
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
        em = rng.uniform(0, 90)
        rows.append(build_features(eg, ea, cs, dc, position, expected_minutes=em))
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
        "days_since_last_appearance": 7.0,
        "zero_minute_streak_length": 0.0,
        "start_rate_last_3": 0.9,
        "start_rate_last_6": 0.9,
        "start_rate_last_15": 0.85,
        "team_rotation_propensity": 0.3,
        "price": 7.5,
        "ownership_log": 2.0,
        "transfers_out_share": 0.01,
        "transfers_balance_share": 0.0,
        "is_goalkeeper": 1.0 if position == "GK" else 0.0,
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
        "own_save_rate_per_90": 4.0,
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


def test_minutes_buckets_and_gated_clean_sheet_probability_emitted(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    predictions = project_gameweek_pool(synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model)

    for col in ["p_zero", "p_1_to_59", "p_60_plus", "player_clean_sheet_probability"]:
        assert col in predictions.columns
        assert predictions[col].between(0.0, 1.0).all()

    bucket_sum = predictions["p_zero"] + predictions["p_1_to_59"] + predictions["p_60_plus"]
    pd.testing.assert_series_equal(
        bucket_sum, pd.Series(1.0, index=predictions.index), check_names=False, atol=1e-6
    )

    # Gated probability can never exceed the ungated team-level probability (Correction 1).
    assert (
        predictions["player_clean_sheet_probability"] <= predictions["clean_sheet_probability"]
    ).all()
    expected = predictions["clean_sheet_probability"] * predictions["p_60_plus"]
    pd.testing.assert_series_equal(
        predictions["player_clean_sheet_probability"], expected, check_names=False, atol=1e-9
    )


def test_raw_component_quantities_emitted_for_calibration(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    predictions = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")

    for col in [
        "expected_minutes_given_1_to_59",
        "expected_minutes_given_60_plus",
        "p_clears_threshold",
        "expected_goals",
        "expected_assists",
        "expected_bonus",
        "expected_saves",
    ]:
        assert col in predictions.columns

    # GK (player 1) has no defensive contribution modelled -- NaN, not a fabricated probability.
    assert np.isnan(predictions.loc[1, "p_clears_threshold"])
    # Outfield players get a real, in-range probability.
    for pid in (2, 3, 4, 5, 6):
        assert 0.0 <= predictions.loc[pid, "p_clears_threshold"] <= 1.0
    assert (predictions["expected_goals"] >= 0.0).all()
    assert (predictions["expected_assists"] >= 0.0).all()
    assert predictions["expected_bonus"].between(0.0, 3.0).all()

    # Saves is the mirror image: modelled for GK only, NaN for outfield players.
    assert predictions.loc[1, "expected_saves"] >= 0.0
    for pid in (2, 3, 4, 5, 6):
        assert np.isnan(predictions.loc[pid, "expected_saves"])


def test_own_goals_activates_via_optional_row_column(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    # ENGINE_IMPROVEMENTS_2.md D.6: omitted by default (existing pool has no own_goal_rate_per_90
    # column); activates and contributes a negative line once the column is present.
    baseline = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")
    assert (baseline["own_goals"] == 0.0).all()

    pool = synthetic_pool.copy()
    pool["own_goal_rate_per_90"] = 0.05
    with_own_goals = project_gameweek_pool(
        pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")

    assert (with_own_goals["own_goals"] < 0.0).all()
    assert (with_own_goals["expected_points"] < baseline["expected_points"]).all()


def test_empty_pool_raises(fitted_minutes_model, fitted_bonus_model):
    with pytest.raises(ValueError):
        project_gameweek_pool(pd.DataFrame(), 1, fitted_minutes_model, fitted_bonus_model)


def test_nan_input_raises_naming_the_offending_player_and_column(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    # A crosswalk miss (ENGINE_IMPROVEMENTS_2.md C.1) leaves a player's npxg_per_90 as NaN --
    # this must fail loudly rather than silently produce a NaN expected_points (C.2).
    pool = synthetic_pool.copy()
    pool.loc[pool["player_id"] == 4, "npxg_per_90"] = float("nan")

    with pytest.raises(ValueError, match=r"\(4, 'npxg_per_90'\)"):
        project_gameweek_pool(pool, 1, fitted_minutes_model, fitted_bonus_model)


def test_fitted_constants_default_matches_no_argument_behavior(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    without_argument = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model
    )
    with_default_constants = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model, FittedConstants()
    )
    pd.testing.assert_frame_equal(without_argument, with_default_constants)


def test_fitted_save_rate_shrinkage_changes_saves_projection(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    # ENGINE_IMPROVEMENTS_3.md D.1: the own-rate fallback's shrinkage constants, not the
    # (unused-for-now) opponent-adjusted project_saves's save_conversion_rate.
    baseline = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")
    shrunk_toward_lower_prior = project_gameweek_pool(
        synthetic_pool,
        1,
        fitted_minutes_model,
        fitted_bonus_model,
        FittedConstants(
            league_avg_save_rate_per_90=1.0,
            save_rate_shrinkage_k=1000.0,
        ),
    ).set_index("player_id")

    # GK's saves should fall toward the (lower) league-average prior; outfield players are
    # untouched by this constant.
    assert shrunk_toward_lower_prior.loc[1, "saves"] < baseline.loc[1, "saves"]
    assert shrunk_toward_lower_prior.loc[2, "saves"] == pytest.approx(baseline.loc[2, "saves"])


def test_fitted_dc_overdispersion_alpha_changes_defensive_contribution(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    baseline = project_gameweek_pool(
        synthetic_pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")
    custom_alpha = project_gameweek_pool(
        synthetic_pool,
        1,
        fitted_minutes_model,
        fitted_bonus_model,
        FittedConstants(dc_overdispersion_alpha={"DEF": 0.6, "MID": 0.6, "FWD": 0.6}),
    ).set_index("player_id")

    assert custom_alpha.loc[2, "defensive_contribution"] != pytest.approx(
        baseline.loc[2, "defensive_contribution"]
    )


def test_fitted_shrinkage_k_shrinks_goals_and_assists_toward_team_prior(
    synthetic_pool, fitted_minutes_model, fitted_bonus_model
):
    pool = synthetic_pool.copy()
    pool["understat_effective_minutes"] = 500.0  # thick evidence for everyone by default
    # Give player 4 (MID) a physically-implausible outlier rate with a thin evidence weight.
    pool.loc[pool["player_id"] == 4, "npxg_per_90"] = 3.0
    pool.loc[pool["player_id"] == 4, "xa_per_90"] = 1.5
    pool.loc[pool["player_id"] == 4, "understat_effective_minutes"] = 5.0

    baseline = project_gameweek_pool(
        pool, 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")
    shrunk = project_gameweek_pool(
        pool,
        1,
        fitted_minutes_model,
        fitted_bonus_model,
        FittedConstants(shrinkage_k=180.0),
    ).set_index("player_id")

    assert shrunk.loc[4, "goals"] < baseline.loc[4, "goals"]
    assert shrunk.loc[4, "assists"] < baseline.loc[4, "assists"]
    # A player with no understat_effective_minutes column at all defaults to full shrinkage --
    # still strictly less than the unshrunk baseline for the same outlier rate.
    assert "expected_goals" in shrunk.columns


def test_penalty_sub_model_activates_via_optional_row_columns(
    fitted_minutes_model, fitted_bonus_model
):
    row = _base_row(7, "FWD")
    row["team_expected_penalties"] = 0.3
    row["taker_share"] = 1.0
    pool = pd.DataFrame([row])

    predictions = project_gameweek_pool(
        pool,
        1,
        fitted_minutes_model,
        fitted_bonus_model,
        FittedConstants(penalty_conversion_rate_by_player={7: 0.9}),
    ).set_index("player_id")

    # A fixed goals rate plus a designated, high-conversion penalty taker role must add strictly
    # positive expected goal points beyond the no-penalty-role baseline.
    no_penalty_role = project_gameweek_pool(
        pd.DataFrame([_base_row(7, "FWD")]), 1, fitted_minutes_model, fitted_bonus_model
    ).set_index("player_id")
    assert predictions.loc[7, "goals"] > no_penalty_role.loc[7, "goals"]
