"""Builds a multi-gameweek "planning horizon" of point-estimate plus simulated projections,
reusing one fixed fitted engine state across every horizon gameweek.
``backtest.run_season.make_predict_fn``/``make_simulate_predict_fn`` already read a fixed,
already-fitted ``FittedEngineState`` and a fixed ``engineered`` frame, this just calls them once
per horizon gameweek instead of once per fit, the *same* fitted models throughout (fit strictly
on history before the real "current" decision gameweek), only the fixture-dependent feature row
per player changes gameweek to gameweek, matching how a real manager plans: you know the next
five weeks' fixtures, never the next five weeks' results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest.run_season import FittedEngineState, make_predict_fn, make_simulate_predict_fn
from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerGameweekProjection,
    PlayerHorizonProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.simulate import PlayerSimulationSummary

__all__ = ["SIMULATION_COLUMNS", "build_horizon_predictions", "build_horizon_projections"]

SIMULATION_COLUMNS = ["sim_mean", "sim_median", "floor", "ceiling", "prob_big_haul"]


def build_horizon_predictions(
    engineered: pd.DataFrame,
    fitted_state: FittedEngineState,
    horizon_gameweeks: list[int],
    n_simulation_runs: int = 200,
    seed: int | None = None,
) -> pd.DataFrame:
    """One row per (player, gameweek) across ``horizon_gameweeks``, point estimate plus simulated
    floor/ceiling/prob_big_haul, all from the same ``fitted_state`` — concatenated, not merged,
    since each gameweek is its own row (matching :class:`~engine.projections
    .PlayerHorizonProjection`'s own shape: one :class:`~engine.projections
    .PlayerGameweekProjection` per gameweek).
    """
    predict_fn = make_predict_fn(engineered)
    simulate_fn = make_simulate_predict_fn(engineered, n_runs=n_simulation_runs, seed=seed)

    frames = []
    for gameweek in horizon_gameweeks:
        predictions = predict_fn(fitted_state, gameweek)
        if predictions.empty:
            continue
        simulation = simulate_fn(fitted_state, gameweek)
        if not simulation.empty:
            predictions = predictions.merge(
                simulation[["player_id", "gameweek", *SIMULATION_COLUMNS]],
                on=["player_id", "gameweek"],
                how="left",
            )
        else:
            for col in SIMULATION_COLUMNS:
                predictions[col] = float("nan")
        frames.append(predictions)

    if not frames:
        return pd.DataFrame(columns=["player_id", "position", "gameweek", "expected_points"])
    return pd.concat(frames, ignore_index=True)


def build_horizon_projections(predictions: pd.DataFrame) -> dict[int, PlayerHorizonProjection]:
    """Like ``scripts.weekly_refresh.build_player_horizon_projections``, but correctly merges
    *every* gameweek in a multi-gameweek ``predictions`` frame into one
    :class:`~engine.projections.PlayerHorizonProjection` per player — that function only ever
    keeps the last gameweek's row per player (fine for its own single-gameweek caller, wrong here
    where ``predictions`` spans the whole horizon).
    """
    by_player: dict[int, dict[int, PlayerGameweekProjection]] = {}
    position_by_player: dict[int, str] = {}
    for row in predictions.itertuples():
        minutes = MinutesDistribution(
            p_zero=row.p_zero,
            p_1_to_59=row.p_1_to_59,
            p_60_plus=row.p_60_plus,
            expected_minutes_given_1_to_59=row.expected_minutes_given_1_to_59,
            expected_minutes_given_60_plus=row.expected_minutes_given_60_plus,
        )
        breakdown = ComponentBreakdown(
            appearance=row.appearance,
            goals=row.goals,
            assists=row.assists,
            clean_sheet=row.clean_sheet,
            goals_conceded=row.goals_conceded,
            defensive_contribution=row.defensive_contribution,
            saves=row.saves,
            bonus=row.bonus,
            cards=row.cards,
            penalty_misses=row.penalty_misses,
            own_goals=row.own_goals,
        )
        simulation = None
        floor = getattr(row, "floor", float("nan"))
        if pd.notna(floor):
            simulation = PlayerSimulationSummary(
                player_id=row.player_id,
                mean=row.sim_mean,
                median=row.sim_median,
                floor=floor,
                ceiling=row.ceiling,
                prob_big_haul=row.prob_big_haul,
                raw_points=np.array([]),
            )
        projection = project_player_gameweek(
            row.player_id, row.position, row.gameweek, minutes, breakdown, simulation
        )
        by_player.setdefault(row.player_id, {})[row.gameweek] = projection
        position_by_player[row.player_id] = row.position

    return {
        player_id: project_player_horizon(player_id, position_by_player[player_id], gameweeks)
        for player_id, gameweeks in by_player.items()
    }
