"""Live multi-gameweek planning horizon.

:mod:`engine.horizon` provides the shared "fit once, predict/simulate per gameweek" helper for a
multi-gameweek horizon, given a frame where every horizon gameweek's fixture-dependent features
already exist as real rows in an already-engineered frame. This module applies that helper to a
live snapshot of a season in progress, where the horizon's gameweeks (the current one, plus
however many gameweeks ahead) haven't been played yet and need
:func:`~engine.data.live_adapter.snapshot_to_feature_inputs`'s target-row synthesis first.

**A season's true opening gameweek(s) need one more piece.** ``snapshot_to_feature_inputs`` can
pool *team-level* Understat rates across prior seasons (its own ``understat_client``/
``prior_season_cache_dir`` parameters), but ``player_histories`` (Understat *per-player* rates,
feeding ``npxg_per_90``/``xa_per_90``) is built from the live snapshot alone, with no such pooling
— empty at a real season's start, same as the team-rate gap used to be. Worse, ``merged_gw`` alone
being real (e.g. via ``engine.data.cross_season.prior_season_merged_gw``) doesn't fix this: with
``player_histories`` still empty, every row — including the prior-season *training* rows —
computes a ``NaN`` ``npxg_per_90``/``xa_per_90``, and ``engineer_features``' dropna still empties
the training frame. :func:`augment_feature_inputs_with_prior_season` closes both gaps together —
see ``engine/data/cross_season.py``'s own module docstring for the full reasoning.

Deliberately does not import ``backtest.run_season``/``engine.horizon`` at module scope — same
documented exception ``engine/data/live_adapter.py`` already makes to "engine/ never depends on
backtest/": this module calls :mod:`engine.horizon`'s shared fit/predict pattern rather than
duplicating it, so the dependency is contained to the one function that needs it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

from engine.data.cross_season import merge_player_histories
from engine.data.live_adapter import (
    DEFAULT_BASE_DIR,
    DEFAULT_TOTAL_MANAGERS,
    FeatureInputs,
    snapshot_to_feature_inputs,
)
from engine.data.understat_client import UnderstatClient
from engine.projections import PlayerHorizonProjection

__all__ = [
    "DEFAULT_HORIZON_LENGTH",
    "DEFAULT_MIN_TRAINING_GAMEWEEKS",
    "LiveHorizonResult",
    "augment_feature_inputs_with_prior_season",
    "build_live_horizon",
    "build_live_horizon_from_feature_inputs",
]

# Current gameweek + next 2 — a manager plans transfers/captaincy/chips against this window.
DEFAULT_HORIZON_LENGTH = 3

# Below this many real training gameweeks, there isn't enough real history yet for `fit_fn` to
# fit a meaningful model (the GW1/GW2 cold-start case engine/data/live_adapter.py's own docstring
# already documents as a real, unresolved gap).
DEFAULT_MIN_TRAINING_GAMEWEEKS = 3


@dataclass(frozen=True)
class LiveHorizonResult:
    """One live horizon call's full output: the raw per-(player, gameweek) prediction/simulation
    rows, and the same data reshaped into one :class:`~engine.projections.PlayerHorizonProjection`
    per player — ``projections[player_id].gameweeks[gw].breakdown`` gives the full per-component
    breakdown for one player, one horizon gameweek."""

    predictions: pd.DataFrame
    projections: dict[int, PlayerHorizonProjection]


def augment_feature_inputs_with_prior_season(
    feature_inputs: FeatureInputs,
    prior_merged_gw: pd.DataFrame,
    prior_teams_extra: pd.DataFrame,
    prior_player_histories: dict[int, pd.DataFrame],
) -> FeatureInputs:
    """Prepend cross-season history onto one live snapshot's own :class:`FeatureInputs`, closing
    the GW1 cold-start gap this module's own docstring describes.

    Every argument here is expected to already be re-keyed onto **this season's** player/team ids
    — ``prior_merged_gw`` via :func:`~engine.data.cross_season.prior_season_merged_gw`,
    ``prior_player_histories`` via :func:`~engine.data.cross_season.remap_player_histories`,
    ``prior_teams_extra`` via :func:`~engine.data.cross_season.synthetic_team_rows` (one row per
    relegated club, so a remapped prior fixture's opponent still resolves to a real name). This
    function's own job is purely to assemble the pieces, not to do any re-keying itself.

    Team-level Understat rates are **not** touched here — that pooling already exists in
    :func:`~engine.data.live_adapter.snapshot_to_feature_inputs` itself (its own
    ``understat_client``/``prior_season_cache_dir`` parameters), which the caller should already
    have passed when building ``feature_inputs`` in the first place.
    """
    merged_gw = pd.concat([prior_merged_gw, feature_inputs.merged_gw], ignore_index=True)
    teams = pd.concat([feature_inputs.teams, prior_teams_extra], ignore_index=True)
    player_histories = merge_player_histories(
        prior_player_histories, feature_inputs.player_histories
    )
    return FeatureInputs(
        merged_gw=merged_gw,
        teams=teams,
        team_histories=feature_inputs.team_histories,
        player_histories=player_histories,
    )


def build_live_horizon_from_feature_inputs(
    feature_inputs: FeatureInputs,
    current_gameweek: int,
    target_gameweeks: Sequence[int],
    min_training_gameweeks: int = DEFAULT_MIN_TRAINING_GAMEWEEKS,
    n_simulation_runs: int = 200,
    seed: int | None = None,
    live_availability: pd.DataFrame | None = None,
) -> LiveHorizonResult:
    """The disk-free core: given already-assembled ``feature_inputs`` (e.g. from
    :func:`~engine.data.live_adapter.snapshot_to_feature_inputs`, or built directly in a test the
    way ``tests/test_live_adapter.py`` already does), fit once on real history strictly before
    ``current_gameweek`` and predict/simulate every gameweek in ``target_gameweeks`` from that one
    fit, exactly :func:`engine.horizon.build_horizon_predictions`'s own pattern.

    Raises ``ValueError`` rather than letting ``fit_fn`` crash obscurely on a near-empty frame
    when fewer than ``min_training_gameweeks`` real gameweeks of history are available.

    ``live_availability`` (T-F) forwards straight through to
    :func:`~backtest.run_season.engineer_features`'s own parameter of the same name, see that
    function's docstring for the full contract, and :func:`~engine.data.live_adapter.
    build_live_availability` for building it from a live snapshot's ``elements`` table. ``None``
    (the default) leaves every row at the "fully fit" placeholder, exactly as before.
    """
    from backtest.run_season import engineer_features, fit_fn
    from engine.horizon import build_horizon_predictions, build_horizon_projections

    engineered = engineer_features(
        feature_inputs.merged_gw,
        feature_inputs.teams,
        feature_inputs.team_histories,
        feature_inputs.player_histories,
        live_availability=live_availability,
    )
    training_history = engineered[engineered["gameweek"] < current_gameweek]
    n_training_gameweeks = training_history["gameweek"].nunique()
    if n_training_gameweeks < min_training_gameweeks:
        raise ValueError(
            f"only {n_training_gameweeks} real gameweek(s) of history available (need >= "
            f"{min_training_gameweeks}) — too early in the season to fit the engine live"
        )

    fitted_state = fit_fn(training_history)
    predictions = build_horizon_predictions(
        engineered, fitted_state, list(target_gameweeks), n_simulation_runs, seed
    )
    projections = build_horizon_projections(predictions)
    return LiveHorizonResult(predictions=predictions, projections=projections)


def build_live_horizon(
    season: str,
    current_gameweek: int,
    captured_at: datetime,
    understat_season_start_year: int,
    base_dir: Path = DEFAULT_BASE_DIR,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
    horizon_length: int = DEFAULT_HORIZON_LENGTH,
    understat_client: UnderstatClient | None = None,
    prior_season_cache_dir: Path | None = None,
    n_prior_seasons: int | None = None,
    min_training_gameweeks: int = DEFAULT_MIN_TRAINING_GAMEWEEKS,
    n_simulation_runs: int = 200,
    seed: int | None = None,
    live_availability: pd.DataFrame | None = None,
) -> LiveHorizonResult:
    """Load one live snapshot and produce a ``horizon_length``-gameweek planning horizon starting
    at ``current_gameweek`` (default: the current gameweek plus the next 2).

    Thin wrapper: builds ``target_gameweeks``, calls
    :func:`~engine.data.live_adapter.snapshot_to_feature_inputs` to synthesize target rows for all
    of them from the one snapshot, then delegates to
    :func:`build_live_horizon_from_feature_inputs`. See that function and
    :func:`~engine.data.live_adapter.snapshot_to_feature_inputs` for what every argument does.

    ``live_availability`` (T-F) is forwarded unchanged to
    :func:`build_live_horizon_from_feature_inputs`; this wrapper does not build it automatically
    from the snapshot, a caller with the snapshot's own ``elements`` table already in scope (e.g.
    ``scripts/build_projections.py``) builds it via
    :func:`~engine.data.live_adapter.build_live_availability` and supplies it here.
    """
    target_gameweeks = list(range(current_gameweek, current_gameweek + horizon_length))
    feature_inputs = snapshot_to_feature_inputs(
        season,
        current_gameweek,
        captured_at,
        understat_season_start_year,
        base_dir,
        total_managers,
        understat_client,
        prior_season_cache_dir,
        n_prior_seasons,
        target_gameweeks=target_gameweeks,
    )
    return build_live_horizon_from_feature_inputs(
        feature_inputs,
        current_gameweek,
        target_gameweeks,
        min_training_gameweeks,
        n_simulation_runs,
        seed,
        live_availability=live_availability,
    )
