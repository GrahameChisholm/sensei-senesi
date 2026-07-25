"""End-to-end integration test for Phase 2's Definition of Done (BUILD_PLAN):

"the engine produces, for every player, a per-gameweek and multi-gameweek projection with a full
distribution and an attached component breakdown, and it runs end-to-end on a historical snapshot
without leakage."

This wires every module built for Phase 2 together for one player: point-in-time EWMA rates from
match history (1.1/engine.rates) -> the minutes model (2.1) -> goals/assists/clean-sheets/
defensive-contribution/cards (2.2-2.6) -> aggregation (2.7) -> the top-level projection (2.9/
projections.py), plus a full-fixture Monte Carlo simulation (2.9) for the outcome distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from engine.aggregate import aggregate_gameweek
from engine.models.assists import project_assists
from engine.models.bonus import BonusModel, build_features
from engine.models.cards import project_cards
from engine.models.clean_sheets import project_clean_sheet
from engine.models.defensive_contribution import project_defensive_contribution
from engine.models.goals import project_goals
from engine.models.minutes import FEATURE_COLUMNS, MinutesModel, encode_status
from engine.projections import project_player_gameweek
from engine.rates import ewma_rate_asof, latest_ewma_rate
from engine.simulate import PlayerMatchInputs, TeamMatchInputs, simulate_fixture


def _player_match_history(n_matches: int = 40, seed: int = 0) -> pd.DataFrame:
    """A synthetic point-in-time match history for one attacking midfielder, standing in for a
    real ingested (Understat + FPL) history (Phase 1)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "npxG": rng.gamma(shape=2.0, scale=0.15, size=n_matches),
            "xA": rng.gamma(shape=2.0, scale=0.1, size=n_matches),
            "defensive_actions": rng.poisson(6, size=n_matches),
            "time": rng.choice([90, 90, 90, 75, 60, 20], size=n_matches),
        }
    )


def test_phase2_pipeline_runs_end_to_end_without_leakage():
    history = _player_match_history()

    # Point-in-time correctness (no leakage): the rate "as of" the most recent match must not
    # depend on that match's own outcome -- swapping its value shouldn't change the prior rate.
    as_of_rates = ewma_rate_asof(history, "npxG", minutes_col="time")
    mutated_history = history.copy()
    mutated_history.loc[len(mutated_history) - 1, "npxG"] = 99.0
    as_of_rates_mutated = ewma_rate_asof(mutated_history, "npxG", minutes_col="time")
    assert as_of_rates.iloc[:-1].equals(as_of_rates_mutated.iloc[:-1])

    player_npxg90 = latest_ewma_rate(history, "npxG", minutes_col="time")
    player_xa90 = latest_ewma_rate(history, "xA", minutes_col="time")
    player_dc90 = latest_ewma_rate(history, "defensive_actions", minutes_col="time")

    # --- 2.1 Minutes model ---
    training_features = pd.DataFrame(
        {
            "recent_start_rate": np.random.default_rng(1).uniform(0.5, 1.0, 100),
            "recent_minutes_ewma": np.random.default_rng(2).uniform(60, 90, 100),
            "fixture_congestion": np.random.default_rng(3).integers(0, 3, 100).astype(float),
            "chance_of_playing_next_round": np.full(100, 100.0),
            "status_score": np.full(100, encode_status("a")),
        }
    )
    started = pd.Series(np.random.default_rng(4).choice([0, 1], size=100, p=[0.2, 0.8]))
    minutes_actual = pd.Series(
        np.where(started == 1, np.random.default_rng(5).choice([90, 88, 45], size=100), 0.0)
    )
    minutes_model = MinutesModel().fit(training_features, started, minutes_actual)
    this_player_features = pd.DataFrame(
        [
            {
                "recent_start_rate": 0.9,
                "recent_minutes_ewma": 85.0,
                "fixture_congestion": 0.0,
                "chance_of_playing_next_round": 100.0,
                "status_score": encode_status("a"),
            }
        ]
    )
    assert list(this_player_features.columns) == FEATURE_COLUMNS
    minutes_distribution = minutes_model.predict(this_player_features)[0]

    league_avg_xga90 = 1.4
    opponent_xga90 = 1.6

    # --- 2.2-2.6 Components ---
    goals = project_goals(
        player_npxg_per_90=player_npxg90,
        opponent_xga_per_90=opponent_xga90,
        league_avg_xga_per_90=league_avg_xga90,
        expected_minutes=minutes_distribution.expected_minutes,
    )
    assists = project_assists(
        player_xa_per_90=player_xa90,
        opponent_xga_per_90=opponent_xga90,
        league_avg_xga_per_90=league_avg_xga90,
        expected_minutes=minutes_distribution.expected_minutes,
    )
    clean_sheet = project_clean_sheet(
        team_xg_per_90=1.5,
        team_xga_per_90=1.1,
        opponent_xg_per_90=1.3,
        opponent_xga_per_90=opponent_xga90,
        league_avg_xga_per_90=league_avg_xga90,
        expected_minutes=minutes_distribution.expected_minutes,
    )
    defensive_contribution = project_defensive_contribution(
        position="MID",
        player_actions_per_90=player_dc90,
        opponent_possession_share=0.55,
        expected_minutes=minutes_distribution.expected_minutes,
    )
    cards = project_cards(
        yellow_card_rate_per_90=0.15,
        red_card_rate_per_90=0.01,
        expected_minutes=minutes_distribution.expected_minutes,
    )

    # --- 2.6 Bonus regression proxy ---
    bonus_training_rows, bonus_training_targets = [], []
    rng = np.random.default_rng(6)
    for _ in range(150):
        position = rng.choice(["GK", "DEF", "MID", "FWD"])
        eg, ea, cs, dc = (
            rng.uniform(0, 1),
            rng.uniform(0, 1),
            rng.uniform(0, 1),
            rng.uniform(0, 15),
        )
        bonus_training_rows.append(build_features(eg, ea, cs, dc, position))
        bonus_training_targets.append(float(np.clip(1.5 * eg + ea + cs, 0, 3)))
    bonus_model = BonusModel().fit(
        pd.DataFrame(bonus_training_rows), pd.Series(bonus_training_targets)
    )
    bonus_features = pd.DataFrame(
        [
            build_features(
                expected_goals=goals.expected_goals,
                expected_assists=assists.expected_assists,
                clean_sheet_probability=clean_sheet.clean_sheet_probability,
                defensive_action_rate=player_dc90,
                position="MID",
            )
        ]
    )
    bonus = bonus_model.predict(bonus_features)[0]

    # --- 2.7 Aggregation ---
    breakdown = aggregate_gameweek(
        "MID",
        minutes_distribution,
        goals,
        assists,
        clean_sheet,
        bonus,
        cards,
        defensive_contribution=defensive_contribution,
    )
    assert np.isfinite(breakdown.total)

    # --- Top-level projection (2.9/projections.py) ---
    projection = project_player_gameweek(
        player_id=42, position="MID", gameweek=1, minutes=minutes_distribution, breakdown=breakdown
    )
    assert projection.expected_points == pytest.approx(breakdown.total)

    # --- Full-fixture Monte Carlo simulation (2.9), for the outcome distribution ---
    this_player = PlayerMatchInputs(
        player_id=42,
        position="MID",
        minutes_distribution=minutes_distribution,
        adjusted_goal_rate_per_90=player_npxg90 * (opponent_xga90 / league_avg_xga90),
        adjusted_assist_rate_per_90=player_xa90 * (opponent_xga90 / league_avg_xga90),
        adjusted_defensive_action_rate_per_90=player_dc90,
        yellow_card_rate_per_90=0.15,
        red_card_rate_per_90=0.01,
    )
    teammates = [
        PlayerMatchInputs(
            player_id=1000 + i,
            position="DEF",
            minutes_distribution=minutes_distribution,
            adjusted_goal_rate_per_90=0.05,
            adjusted_assist_rate_per_90=0.05,
            adjusted_defensive_action_rate_per_90=8.0,
        )
        for i in range(4)
    ] + [
        PlayerMatchInputs(
            player_id=2000,
            position="GK",
            minutes_distribution=minutes_distribution,
            adjusted_goal_rate_per_90=0.0,
            adjusted_assist_rate_per_90=0.0,
            expected_saves_full_match=3.0,
        )
    ]
    opponents = [
        PlayerMatchInputs(
            player_id=3000 + i,
            position="FWD",
            minutes_distribution=minutes_distribution,
            adjusted_goal_rate_per_90=0.3,
            adjusted_assist_rate_per_90=0.1,
        )
        for i in range(2)
    ] + [
        PlayerMatchInputs(
            player_id=4000,
            position="GK",
            minutes_distribution=minutes_distribution,
            adjusted_goal_rate_per_90=0.0,
            adjusted_assist_rate_per_90=0.0,
            expected_saves_full_match=3.0,
        )
    ]

    home = TeamMatchInputs(players=[this_player, *teammates])
    away = TeamMatchInputs(players=opponents)
    simulation = simulate_fixture(home, away, bonus_model, n_runs=500, seed=99)

    summary = simulation.player_summaries[42]
    full_projection = project_player_gameweek(
        player_id=42,
        position="MID",
        gameweek=1,
        minutes=minutes_distribution,
        breakdown=breakdown,
        simulation=summary,
    )
    assert full_projection.simulation is not None
    assert full_projection.simulation.floor <= full_projection.simulation.ceiling
