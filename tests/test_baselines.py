"""Tests for the required baselines and the statistical significance tests (3.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest.baselines import (
    constant_predictions,
    naive_form_predictions,
    paired_bootstrap_test,
    permutation_test_hit_rate,
    pure_xg_predictions,
    template_captain_predictions,
    training_median,
)


def test_template_captain_predictions_uses_ownership_as_expected_points():
    ownership = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "position": ["MID", "MID", "FWD"],
            "gameweek": [1, 1, 1],
            "selected_by_percent": [45.0, 10.0, 60.0],
        }
    )

    out = template_captain_predictions(ownership)

    assert list(out["expected_points"]) == [45.0, 10.0, 60.0]
    assert list(out.columns) == ["player_id", "position", "gameweek", "expected_points"]


def test_naive_form_predictions_shifts_forward_with_no_leakage():
    history = pd.DataFrame(
        {
            "player_id": [1, 1, 1, 1],
            "position": ["MID"] * 4,
            "gameweek": [1, 2, 3, 4],
            "total_points": [2.0, 4.0, 6.0, 100.0],
        }
    )

    out = naive_form_predictions(history, n_recent=2).set_index("gameweek")

    # GW2 projection uses only GW1 (mean=2.0); GW3 uses GW1-2 (mean=3.0); GW4 uses GW2-3 (mean=5.0).
    # Critically, GW4's huge actual (100.0) must never leak into any projection for GW4 itself.
    assert out.loc[2, "expected_points"] == pytest.approx(2.0)
    assert out.loc[3, "expected_points"] == pytest.approx(3.0)
    assert out.loc[4, "expected_points"] == pytest.approx(5.0)
    assert 1 not in out.index  # first gameweek has no prior history to project from


def test_pure_xg_predictions_no_opponent_adjustment_no_minutes_model():
    player_rates = pd.DataFrame(
        {
            "player_id": [1, 2],
            "position": ["FWD", "MID"],
            "gameweek": [1, 1],
            "npxg_per_90": [0.5, 0.2],
            "xa_per_90": [0.1, 0.3],
            "recent_minutes_ewma": [90.0, 45.0],
        }
    )

    out = pure_xg_predictions(player_rates).set_index("player_id")

    # FWD: goal points = 4, assist points = 3. (0.5*4 + 0.1*3) * (90/90) = 2.3
    assert out.loc[1, "expected_points"] == pytest.approx(0.5 * 4 + 0.1 * 3)
    # MID: goal points = 5, assist points = 3, scaled by 45/90 = 0.5
    assert out.loc[2, "expected_points"] == pytest.approx((0.2 * 5 + 0.3 * 3) * 0.5)


def test_training_median_computes_median_of_training_history():
    history = pd.DataFrame({"total_points": [1.0, 1.0, 2.0, 5.0, 100.0]})

    assert training_median(history) == pytest.approx(2.0)


def test_constant_predictions_assigns_flat_value_to_every_row():
    players = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "position": ["MID", "FWD", "DEF"],
            "gameweek": [5, 5, 5],
        }
    )

    out = constant_predictions(players, value=1.0)

    assert list(out["expected_points"]) == [1.0, 1.0, 1.0]
    assert list(out.columns) == ["player_id", "position", "gameweek", "expected_points"]


def test_paired_bootstrap_test_detects_engine_beating_baseline():
    rng = np.random.default_rng(0)
    n = 200
    engine_errors = np.abs(rng.normal(1.0, 0.3, n))  # consistently lower error
    baseline_errors = np.abs(rng.normal(3.0, 0.3, n))  # consistently higher error

    result = paired_bootstrap_test(engine_errors, baseline_errors, n_bootstrap=2000, seed=1)

    assert result.mean_diff < 0
    assert result.ci_high < 0
    assert result.beats_baseline


def test_paired_bootstrap_test_no_edge_does_not_beat_baseline():
    rng = np.random.default_rng(0)
    n = 200
    errors_a = np.abs(rng.normal(2.0, 0.5, n))
    errors_b = np.abs(rng.normal(2.0, 0.5, n))

    result = paired_bootstrap_test(errors_a, errors_b, n_bootstrap=2000, seed=1)

    assert not result.beats_baseline


def test_paired_bootstrap_test_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        paired_bootstrap_test(np.array([1.0, 2.0]), np.array([1.0]))


def test_paired_bootstrap_test_block_by_widens_interval_but_keeps_conclusion():
    # ENGINE_IMPROVEMENTS_2.md D.2: blocking by a grouping where rows share a per-block shock
    # should widen the CI relative to the naive i.i.d. version without reversing a real edge.
    rng = np.random.default_rng(0)
    n_blocks = 40
    rows_per_block = 5
    block_shock = rng.normal(0, 1.5, n_blocks)
    engine_errors, baseline_errors, block_ids = [], [], []
    for b in range(n_blocks):
        for _ in range(rows_per_block):
            engine_errors.append(abs(1.0 + block_shock[b] + rng.normal(0, 0.1)))
            baseline_errors.append(abs(3.0 + block_shock[b] + rng.normal(0, 0.1)))
            block_ids.append(b)
    engine_errors, baseline_errors, block_ids = map(np.array, (engine_errors, baseline_errors, block_ids))

    iid_result = paired_bootstrap_test(engine_errors, baseline_errors, n_bootstrap=3000, seed=1)
    blocked_result = paired_bootstrap_test(
        engine_errors, baseline_errors, n_bootstrap=3000, seed=1, block_by=block_ids
    )

    assert iid_result.beats_baseline
    assert blocked_result.beats_baseline  # the real edge survives
    iid_width = iid_result.ci_high - iid_result.ci_low
    blocked_width = blocked_result.ci_high - blocked_result.ci_low
    assert blocked_width > iid_width  # blocking must not understate uncertainty


def test_paired_bootstrap_test_block_by_rejects_mismatched_length():
    with pytest.raises(ValueError):
        paired_bootstrap_test(
            np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0]), block_by=np.array([1, 2])
        )


def test_permutation_test_hit_rate_detects_genuine_edge():
    rng = np.random.default_rng(0)
    n = 300
    engine_hits = (rng.uniform(0, 1, n) < 0.55).astype(float)
    baseline_hits = (rng.uniform(0, 1, n) < 0.30).astype(float)

    result = permutation_test_hit_rate(engine_hits, baseline_hits, n_permutations=2000, seed=1)

    assert result.observed_diff > 0
    assert result.beats_baseline
    assert result.p_value < 0.05


def test_permutation_test_hit_rate_no_edge_not_significant():
    rng = np.random.default_rng(0)
    n = 50
    engine_hits = (rng.uniform(0, 1, n) < 0.4).astype(float)
    baseline_hits = (rng.uniform(0, 1, n) < 0.4).astype(float)

    result = permutation_test_hit_rate(engine_hits, baseline_hits, n_permutations=2000, seed=1)

    assert not result.beats_baseline
