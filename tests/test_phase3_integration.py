"""End-to-end integration test for Phase 3's quality gate (BUILD_PLAN):

Wires every module built for Phase 3 together over one synthetic multi-gameweek "season": the
walk-forward harness (3.1) refitting a real Phase 2 module — ``engine.regression`` — every
gameweek on an expanding window; accuracy/bias/calibration/captaincy metrics (3.2); the required
baselines and their statistical significance tests (3.3); immutable, version-tagged prediction
logging (3.4); and the Definition-of-Done gate (3.6) that decides whether the engine has earned
the right to move on to Phase 4/5.

The synthetic "engine" here is deliberately simple (a single true per-player rate feature fit via
real OLS regression) — the point of this test is that the Phase 3 *scaffolding* wires together
correctly and produces a sensible pass/fail verdict, not to validate the real Phase 2 engine's
actual predictive skill (that happens against real historical data, not synthetic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.baselines import (
    naive_form_predictions,
    paired_bootstrap_test,
    permutation_test_hit_rate,
    template_captain_predictions,
)
from backtest.gate import evaluate_definition_of_done
from backtest.harness import run_walk_forward
from backtest.metrics import (
    bias_by_group,
    captaincy_hit_rate,
    component_calibration,
    player_accuracy,
)
from backtest.prediction_log import current_model_version, load_logged_predictions, log_predictions
from engine.regression import PerPositionRegression

N_GAMEWEEKS = 30
N_PLAYERS_PER_POSITION = 4
TRUE_INTERCEPT = 2.0
TRUE_WEIGHT = 10.0
NOISE_STD = 1.0
NAIVE_FORM_N_RECENT = 3


@pytest.fixture(scope="module")
def synthetic_season():
    rng = np.random.default_rng(42)
    player_ids = list(range(1, 2 * N_PLAYERS_PER_POSITION + 1))
    positions = ["MID"] * N_PLAYERS_PER_POSITION + ["FWD"] * N_PLAYERS_PER_POSITION
    true_rate90 = rng.uniform(0.1, 0.9, size=len(player_ids))
    # Ownership deliberately uncorrelated with true skill -- "captaining the highest-owned player"
    # (BUILD_PLAN 3.3) is a fan-popularity signal, not a statistically-informed one.
    ownership = rng.uniform(0, 100, size=len(player_ids))

    player_features = pd.DataFrame(
        {"player_id": player_ids, "position": positions, "rate90": true_rate90}
    )

    history_rows = []
    for gw in range(1, N_GAMEWEEKS + 1):
        noise = rng.normal(0, NOISE_STD, size=len(player_ids))
        actual_points = TRUE_INTERCEPT + TRUE_WEIGHT * true_rate90 + noise
        for i, player_id in enumerate(player_ids):
            history_rows.append(
                {
                    "player_id": player_id,
                    "position": positions[i],
                    "gameweek": gw,
                    "rate90": true_rate90[i],
                    "total_points": actual_points[i],
                    "minutes": 90,
                }
            )
    history = pd.DataFrame(history_rows)

    ownership_rows = [
        {
            "player_id": player_id,
            "position": positions[i],
            "gameweek": gw,
            "selected_by_percent": ownership[i],
        }
        for gw in range(1, N_GAMEWEEKS + 1)
        for i, player_id in enumerate(player_ids)
    ]
    ownership_df = pd.DataFrame(ownership_rows)

    return {
        "player_features": player_features,
        "history": history,
        "ownership": ownership_df,
        "player_ids": player_ids,
    }


def _fit_fn(training_history: pd.DataFrame) -> PerPositionRegression:
    model = PerPositionRegression(feature_columns=["rate90"], kind="linear")
    model.fit(training_history, target_col="total_points")
    return model


def _make_predict_fn(player_features: pd.DataFrame):
    def predict_fn(fitted_state: PerPositionRegression, gameweek: int) -> pd.DataFrame:
        preds = fitted_state.predict(player_features)
        return player_features.assign(expected_points=preds, expected_minutes=90.0)[
            ["player_id", "position", "expected_points", "expected_minutes"]
        ]

    return predict_fn


def test_phase3_pipeline_runs_end_to_end_and_the_engine_clears_the_dod_gate(
    synthetic_season, tmp_path
):
    history = synthetic_season["history"]
    player_features = synthetic_season["player_features"]
    gameweeks = list(range(1, N_GAMEWEEKS + 1))

    # --- 3.1 Walk-forward harness, refitting a real Phase 2 regression every gameweek ---
    walk_result = run_walk_forward(
        gameweeks=gameweeks,
        history=history,
        fit_fn=_fit_fn,
        predict_fn=_make_predict_fn(player_features),
        min_training_gameweeks=1,
    )
    assert walk_result.skipped_gameweeks == (1,)  # gameweek 1 has no prior training data
    engine_predictions = walk_result.predictions

    # --- 3.3 Baselines ---
    naive_form_baseline = naive_form_predictions(history, n_recent=NAIVE_FORM_N_RECENT)
    template_baseline = template_captain_predictions(synthetic_season["ownership"])

    # --- 3.2 Player-level accuracy: the engine should materially beat the naive-form baseline ---
    engine_accuracy = player_accuracy(engine_predictions, history)
    baseline_accuracy = player_accuracy(naive_form_baseline, history)
    assert engine_accuracy.overall_mae < baseline_accuracy.overall_mae

    # --- 3.3 Statistical significance test on the MAE difference (not just the point estimate) ---
    paired = engine_predictions.merge(
        naive_form_baseline,
        on=["player_id", "gameweek"],
        suffixes=("_engine", "_baseline"),
    ).merge(history[["player_id", "gameweek", "total_points"]], on=["player_id", "gameweek"])
    engine_abs_errors = (paired["expected_points_engine"] - paired["total_points"]).abs().to_numpy()
    baseline_abs_errors = (
        (paired["expected_points_baseline"] - paired["total_points"]).abs().to_numpy()
    )
    naive_form_test = paired_bootstrap_test(engine_abs_errors, baseline_abs_errors, seed=7)
    assert naive_form_test.beats_baseline

    # --- 3.2 Bias detection: no severe systematic over/under-rating by position ---
    bias_report = bias_by_group(engine_predictions, history, group_col="position")
    assert not bias_report.by_group["severe"].any()

    # --- 3.2 Component calibration (a synthetic well-calibrated probability, e.g. clean sheets) ---
    rng = np.random.default_rng(1)
    predicted_prob = rng.uniform(0.1, 0.6, size=2000)
    actual_outcome = (rng.uniform(0, 1, size=2000) < predicted_prob).astype(float)
    calibration_report = component_calibration(pd.Series(predicted_prob), pd.Series(actual_outcome))
    assert calibration_report.mean_absolute_calibration_error < 0.15

    # --- 3.2 Captaincy hit-rate: eligible options = the full squad every gameweek here ---
    starting_xi = {
        gw: set(synthetic_season["player_ids"]) for gw in engine_predictions["gameweek"].unique()
    }
    engine_captaincy = captaincy_hit_rate(engine_predictions, history, starting_xi)
    template_captaincy = captaincy_hit_rate(template_baseline, history, starting_xi)
    assert engine_captaincy.raw_hit_rate > template_captaincy.raw_hit_rate

    # --- 3.3 Statistical significance test on the captaincy hit-rate difference ---
    aligned = engine_captaincy.per_gameweek.merge(
        template_captaincy.per_gameweek, on="gameweek", suffixes=("_engine", "_template")
    )
    captaincy_test = permutation_test_hit_rate(
        aligned["hit_engine"].to_numpy(dtype=float),
        aligned["hit_template"].to_numpy(dtype=float),
        seed=7,
    )

    # --- 3.4 Immutable, version-tagged prediction logging ---
    model_version = current_model_version()
    last_gw = int(engine_predictions["gameweek"].max())
    last_gw_predictions = engine_predictions[engine_predictions["gameweek"] == last_gw]
    log_entry = log_predictions(
        last_gw_predictions, gameweek=last_gw, model_version=model_version, log_dir=tmp_path
    )
    assert log_entry.path.exists()
    reloaded = load_logged_predictions(log_dir=tmp_path, gameweek=last_gw)
    assert len(reloaded) == len(last_gw_predictions)

    # --- 3.6 Definition-of-Done gate ---
    dod_report = evaluate_definition_of_done(
        baseline_results={"naive_form": naive_form_test, "template_captain": captaincy_test},
        bias_reports={"position": bias_report},
        calibration_reports={"clean_sheet": calibration_report},
        predictions_logged=True,
        trusted_by_user=True,
    )
    assert dod_report.passed, dod_report.summary()
