"""Tests for the walk-forward harness (3.1): expanding window, no-leakage, refit-every-gameweek."""

from __future__ import annotations

import pandas as pd

from backtest.harness import run_walk_forward


def _history(n_gameweeks: int = 5, n_players: int = 3) -> pd.DataFrame:
    rows = []
    for gw in range(1, n_gameweeks + 1):
        for player_id in range(1, n_players + 1):
            rows.append({"player_id": player_id, "gameweek": gw, "total_points": gw + player_id})
    return pd.DataFrame(rows)


def test_fit_fn_only_ever_sees_gameweeks_strictly_before_the_predicted_one():
    history = _history(n_gameweeks=5)
    seen_max_gameweeks = {}

    def fit_fn(training_history: pd.DataFrame) -> object:
        return training_history["gameweek"].max() if not training_history.empty else 0

    def predict_fn(fitted_state: object, gameweek: int) -> pd.DataFrame:
        seen_max_gameweeks[gameweek] = fitted_state
        return pd.DataFrame({"player_id": [1, 2, 3], "expected_points": [1.0, 2.0, 3.0]})

    result = run_walk_forward(
        gameweeks=[2, 3, 4, 5],
        history=history,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        min_training_gameweeks=1,
    )

    assert seen_max_gameweeks == {2: 1, 3: 2, 4: 3, 5: 4}
    assert result.skipped_gameweeks == ()


def test_gameweeks_without_enough_training_history_are_skipped_not_dropped_silently():
    history = _history(n_gameweeks=5)

    def fit_fn(training_history: pd.DataFrame) -> object:
        return object()

    def predict_fn(fitted_state: object, gameweek: int) -> pd.DataFrame:
        return pd.DataFrame({"player_id": [1], "expected_points": [1.0]})

    result = run_walk_forward(
        gameweeks=[1, 2, 3],
        history=history,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
        min_training_gameweeks=2,
    )

    assert result.skipped_gameweeks == (1, 2)
    assert sorted(result.predictions["gameweek"].unique()) == [3]


def test_predictions_are_concatenated_with_gameweek_column_attached():
    history = _history(n_gameweeks=3)

    def fit_fn(training_history: pd.DataFrame) -> object:
        return None

    def predict_fn(fitted_state: object, gameweek: int) -> pd.DataFrame:
        return pd.DataFrame(
            {"player_id": [1, 2], "expected_points": [gameweek * 1.0, gameweek * 2.0]}
        )

    result = run_walk_forward(
        gameweeks=[2, 3],
        history=history,
        fit_fn=fit_fn,
        predict_fn=predict_fn,
    )

    assert list(result.predictions["gameweek"]) == [2, 2, 3, 3]
    assert list(result.predictions["expected_points"]) == [2.0, 4.0, 3.0, 6.0]


def test_gameweeks_are_walked_in_ascending_order_regardless_of_input_order():
    history = _history(n_gameweeks=4)
    fit_order = []

    def fit_fn(training_history: pd.DataFrame) -> object:
        fit_order.append(training_history["gameweek"].max() if not training_history.empty else 0)
        return None

    def predict_fn(fitted_state: object, gameweek: int) -> pd.DataFrame:
        return pd.DataFrame({"player_id": [1], "expected_points": [1.0]})

    run_walk_forward(gameweeks=[4, 2, 3], history=history, fit_fn=fit_fn, predict_fn=predict_fn)

    assert fit_order == [1, 2, 3]


def test_empty_gameweek_list_returns_empty_predictions_frame():
    history = _history(n_gameweeks=2)

    result = run_walk_forward(
        gameweeks=[],
        history=history,
        fit_fn=lambda h: None,
        predict_fn=lambda s, gw: pd.DataFrame({"player_id": [1], "expected_points": [1.0]}),
    )

    assert result.predictions.empty
    assert result.skipped_gameweeks == ()
