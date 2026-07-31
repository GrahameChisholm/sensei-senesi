"""Phase 6 weekly refresh job — the "keep it running and honest across the season" operational
loop the app depends on once real users are on it (BUILD_PLAN Phase 6): capture the pre-deadline
snapshot, regenerate projections, log predictions immutably, pull the Phase 4b odds snapshot, and
push the refreshed state to the API.

**``build_pool_projections`` is real as of ENGINE_IMPROVEMENTS_4.md's Track A1.**
``engine.data.live_adapter.snapshot_to_feature_inputs`` maps one live snapshot into the exact
shape ``backtest.run_season.engineer_features`` expects — the adapter this module's own docstring
used to say didn't exist. :func:`make_build_pool_projections` wires that adapter through
``engineer_features`` → ``fit_fn`` → ``make_predict_fn``/``make_simulate_predict_fn`` into one
factory producing the hook this function needs, with no leakage concern in the fit/predict step
itself: ``fit_fn(training_history)`` fits on "everything strictly before the target gameweek"
(exactly what live deployment wants too), and ``predict_fn(fitted_state, gameweek)`` only reads
pre-match features for that gameweek, never an actual outcome.

**Still a one-gameweek horizon.** :func:`build_player_horizon_projections` builds one
:class:`~engine.projections.PlayerHorizonProjection` per player from a *single* gameweek's
predictions — every existing single-gameweek feature (captaincy, chips) only ever reads one
horizon gameweek at a time regardless, so this is already useful, but ``features.transfers``'
multi-gameweek planning value needs the full horizon. Extending to it is mechanical, not blocked:
call :func:`make_build_pool_projections`'s closure once per horizon gameweek (same
``fitted_state``, since it's fit once on data before the *current* real gameweek regardless of
which future gameweek is being projected) and merge the resulting frames before building
projections — not implemented here because nothing downstream needs it yet.

**Squad and prices are real now too (Track A3).** :func:`make_build_app_state` combines
``engine.data.team_state_builder.build_my_team_state`` (pulled live via
``engine.data.fpl_client.FPLClient``'s manager endpoints) with
:func:`build_player_horizon_projections` into one :class:`~api.state.AppState` — ``my_team``,
``team_id_by_player``, and ``buy_prices`` are all real. ``fixtures``/``team_rates`` are the one
piece still supplied by the caller: computing a live :class:`~features.fixtures.TeamRates` per
team needs the same kind of Understat-team-history work ``engine.data.live_adapter`` already does
for player rates, just not yet built for teams specifically — see
:func:`make_build_app_state`'s own docstring.

**Odds are best-effort, never blocking.** A failed odds pull is caught and reported on the result
rather than aborting the refresh — the market overlay is explicitly a separate, optional layer
(BUILD_PLAN 4b), so projections and their logging must never depend on it succeeding.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from api.state import AppState, set_state
from backtest.prediction_log import current_model_version, log_predictions
from backtest.run_season import engineer_features, fit_fn, make_predict_fn, make_simulate_predict_fn
from engine.aggregate import ComponentBreakdown
from engine.data.fpl_client import FPLClient, bootstrap_to_dataframes
from engine.data.ingest import capture_current_gameweek
from engine.data.live_adapter import DEFAULT_TOTAL_MANAGERS, snapshot_to_feature_inputs
from engine.data.snapshots import DEFAULT_BASE_DIR, SnapshotManifest, load_snapshot_tables
from engine.data.team_state_builder import build_my_team_state
from engine.data.understat_client import UnderstatClient
from engine.models.minutes import MinutesDistribution
from engine.projections import (
    PlayerHorizonProjection,
    project_player_gameweek,
    project_player_horizon,
)
from engine.simulate import PlayerSimulationSummary
from features.fixtures import TeamFixture, TeamRates
from features.team_state import MyTeamState
from market_overlay.odds_client import OddsClient, OddsClientError

logger = logging.getLogger(__name__)

BuildPoolProjections = Callable[[SnapshotManifest, int], pd.DataFrame]
BuildAppState = Callable[[pd.DataFrame, datetime], AppState]

# Columns simulate_gameweek_pool attaches, merged onto the point-estimate predictions frame.
_SIMULATION_COLUMNS = ["sim_mean", "sim_median", "floor", "ceiling", "prob_big_haul"]


def make_build_pool_projections(
    understat_season_start_year: int,
    base_dir: Path = DEFAULT_BASE_DIR,
    n_simulation_runs: int = 200,
    seed: int | None = None,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
    understat_client: UnderstatClient | None = None,
    prior_season_cache_dir: Path | None = None,
    n_prior_seasons: int | None = None,
) -> BuildPoolProjections:
    """Factory for the ``build_pool_projections`` hook :func:`run_weekly_refresh` expects.

    The returned closure: loads the snapshot's live ``chance_of_playing_next_round``/``status``
    (the strongest injury signal the backtest could never reconstruct — see
    ``engineer_features``'s own ``live_availability`` parameter) from the same snapshot's ``fpl``
    source, builds the ``engineer_features`` inputs via ``engine.data.live_adapter``, fits on
    every gameweek strictly before ``gameweek``, and returns one point-estimate-plus-simulation row
    per player for ``gameweek``.

    ``understat_client``/``prior_season_cache_dir``/``n_prior_seasons`` pass straight through to
    ``snapshot_to_feature_inputs``' own cold-start fix (A6 — a live pull against the real 2026/27
    pre-season Understat endpoint found it returns zero players/teams before a season's first
    match, which without this would drop every player from a season's opening gameweek(s) for
    lack of any team-level rate at all). Supply a real ``understat_client`` for any gameweek early
    enough in a season that this could matter; omitting it (the default) is a real gap only in
    exactly that window — see that function's own docstring.
    """

    def build(snapshot: SnapshotManifest, gameweek: int) -> pd.DataFrame:
        fpl = load_snapshot_tables(base_dir, snapshot.season, gameweek, snapshot.captured_at, "fpl")
        elements = fpl["elements"]
        live_availability = elements[["id", "chance_of_playing_next_round", "status"]].rename(
            columns={"id": "player_id"}
        )
        live_availability["chance_of_playing_next_round"] = live_availability[
            "chance_of_playing_next_round"
        ].fillna(100.0)

        feature_inputs = snapshot_to_feature_inputs(
            snapshot.season,
            gameweek,
            snapshot.captured_at,
            understat_season_start_year,
            base_dir,
            total_managers,
            understat_client,
            prior_season_cache_dir,
            n_prior_seasons,
        )
        engineered = engineer_features(
            feature_inputs.merged_gw,
            feature_inputs.teams,
            feature_inputs.team_histories,
            feature_inputs.player_histories,
            live_availability=live_availability,
        )
        training_history = engineered[engineered["gameweek"] < gameweek]
        fitted_state = fit_fn(training_history)

        predict_fn = make_predict_fn(engineered)
        predictions = predict_fn(fitted_state, gameweek)

        simulate_predict_fn = make_simulate_predict_fn(
            engineered, n_runs=n_simulation_runs, seed=seed
        )
        simulation = simulate_predict_fn(fitted_state, gameweek)
        if not simulation.empty:
            predictions = predictions.merge(
                simulation[["player_id", "gameweek", *_SIMULATION_COLUMNS]],
                on=["player_id", "gameweek"],
                how="left",
            )
        else:
            for col in _SIMULATION_COLUMNS:
                predictions[col] = float("nan")
        return predictions

    return build


def build_player_horizon_projections(
    predictions: pd.DataFrame,
) -> dict[int, PlayerHorizonProjection]:
    """One :class:`~engine.projections.PlayerHorizonProjection` per player, from a single
    gameweek's predictions frame — see this module's own docstring on why this is a one-gameweek
    horizon for now. ``raw_points`` on the reconstructed
    :class:`~engine.simulate.PlayerSimulationSummary` is an empty placeholder, not the real
    per-run array — ``simulate_gameweek_pool`` never retains it past its own floor/ceiling/
    prob_big_haul summary, and no real consumer (``features.captaincy`` included) reads it.
    """
    projections: dict[int, PlayerHorizonProjection] = {}
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
        gameweek_projection = project_player_gameweek(
            row.player_id, row.position, row.gameweek, minutes, breakdown, simulation
        )
        projections[row.player_id] = project_player_horizon(
            row.player_id, row.position, {row.gameweek: gameweek_projection}
        )
    return projections


def build_app_state_from_predictions(
    predictions: pd.DataFrame,
    my_team: MyTeamState,
    team_id_by_player: dict[int, int],
    buy_prices: dict[int, int],
    fixtures: list[TeamFixture],
    team_rates: dict[int, TeamRates],
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
    horizon_gameweeks: list[int],
    player_names: dict[int, str] | None = None,
    generated_at: datetime | None = None,
) -> AppState:
    """Assemble a full :class:`~api.state.AppState` from real projections plus the squad/fixture
    state a caller supplies directly (see this module's own docstring on why that's still
    ``api.demo_data`` today, pending Track A3's real FPL-manager-endpoint squad).
    """
    return AppState(
        my_team=my_team,
        projections=build_player_horizon_projections(predictions),
        team_id_by_player=team_id_by_player,
        buy_prices=buy_prices,
        fixtures=fixtures,
        team_rates=team_rates,
        league_avg_xg_per_90=league_avg_xg_per_90,
        league_avg_xga_per_90=league_avg_xga_per_90,
        horizon_gameweeks=horizon_gameweeks,
        player_names=player_names or {},
        generated_at=generated_at,
    )


def make_build_app_state(
    fpl_client: FPLClient,
    entry_id: int,
    current_gameweek: int,
    horizon_gameweeks: list[int],
    fixtures: list[TeamFixture],
    team_rates: dict[int, TeamRates],
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
    free_transfers: int | None = None,
) -> BuildAppState:
    """Factory for the ``build_app_state`` hook :func:`run_weekly_refresh` expects — the piece
    that finally combines Track A3's real squad (:func:`~engine.data.team_state_builder
    .build_my_team_state`) with Track A1/A2's real projections
    (:func:`build_player_horizon_projections`) into one :class:`~api.state.AppState`.

    ``team_id_by_player``/``buy_prices`` are real, pulled fresh from the same bootstrap
    ``elements`` this needs anyway for the squad. ``fixtures``/``team_rates`` are **not** — this
    factory has no live source for them (``build_app_state``'s own signature, unlike
    ``build_pool_projections``'s, never receives the snapshot manifest, only the predictions
    frame, so there's nothing here to derive them from without a live client call this function
    doesn't make) and must be supplied by the caller. Computing them for real needs a
    :class:`~features.fixtures.TeamRates` per team from live Understat team histories — legitimate
    follow-up work, not attempted here.
    """

    def build(predictions: pd.DataFrame, generated_at: datetime) -> AppState:
        entry = fpl_client.get_entry(entry_id)
        picks = fpl_client.get_entry_picks(entry_id, current_gameweek)
        transfers = fpl_client.get_entry_transfers(entry_id)
        history = fpl_client.get_entry_history(entry_id)
        elements = bootstrap_to_dataframes(fpl_client.get_bootstrap_static())["elements"]

        my_team = build_my_team_state(
            picks, entry, transfers, history, elements, current_gameweek, free_transfers
        )
        team_id_by_player = {int(row.id): int(row.team) for row in elements.itertuples()}
        buy_prices = {int(row.id): int(row.now_cost) for row in elements.itertuples()}
        player_names = {int(row.id): row.web_name for row in elements.itertuples()}

        return build_app_state_from_predictions(
            predictions,
            my_team,
            team_id_by_player,
            buy_prices,
            fixtures,
            team_rates,
            league_avg_xg_per_90,
            league_avg_xga_per_90,
            horizon_gameweeks,
            player_names,
            generated_at,
        )

    return build


@dataclass(frozen=True)
class WeeklyRefreshResult:
    snapshot: SnapshotManifest
    predictions: pd.DataFrame
    prediction_log_path: Path
    odds_pulled: bool
    odds_error: str | None


def run_weekly_refresh(
    fpl_client: FPLClient,
    understat_client: UnderstatClient,
    odds_client: OddsClient,
    season: str,
    understat_season_start_year: int,
    gameweek: int,
    build_pool_projections: BuildPoolProjections,
    build_app_state: BuildAppState,
) -> WeeklyRefreshResult:
    """Run one gameweek's full weekly refresh.

    ``build_pool_projections(snapshot, gameweek)`` must return the same
    ``engine.pipeline.GAMEWEEK_POOL_COLUMNS``-shaped-then-projected DataFrame
    ``backtest.prediction_log.log_predictions`` expects (see that module and
    ``engine.pipeline.project_gameweek_pool``'s own return shape).
    ``build_app_state(predictions, generated_at)`` turns those predictions into a full
    :class:`~api.state.AppState` (squad, fixtures, prices, ...) for the API to start serving
    immediately — ``generated_at`` is this snapshot's own ``captured_at`` (A6: the "data as of"
    timestamp the UI shows), passed here rather than baked into the ``build_app_state`` closure
    since the snapshot doesn't exist yet at the point a caller constructs that closure.
    """
    logger.info("capturing pre-deadline snapshot for %s gw%d", season, gameweek)
    snapshot = capture_current_gameweek(
        fpl_client, understat_client, season, understat_season_start_year, gameweek
    )

    logger.info("regenerating projections for gw%d", gameweek)
    predictions = build_pool_projections(snapshot, gameweek)

    logger.info("logging predictions immutably")
    log_entry = log_predictions(predictions, gameweek, current_model_version())

    odds_pulled = False
    odds_error: str | None = None
    try:
        logger.info("pulling Phase 4b odds snapshot")
        odds_client.get_match_odds()
        odds_pulled = True
    except OddsClientError as exc:
        logger.warning("odds pull failed, continuing without it: %s", exc)
        odds_error = str(exc)

    logger.info("refreshing API state")
    set_state(build_app_state(predictions, snapshot.captured_at))

    return WeeklyRefreshResult(
        snapshot=snapshot,
        predictions=predictions,
        prediction_log_path=log_entry.path,
        odds_pulled=odds_pulled,
        odds_error=odds_error,
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", required=True, help='snapshot season key, e.g. "2026-27"')
    parser.add_argument("--understat-season-start-year", type=int, required=True)
    parser.add_argument("--gameweek", type=int, required=True)
    parser.parse_args(argv)

    raise SystemExit(
        "scripts.weekly_refresh's build_pool_projections is real now (make_build_pool_projections) "
        "-- but build_app_state still needs a real squad this CLI has no source for (Track A3: no "
        "FPL manager endpoints exist yet). Call run_weekly_refresh(...) directly, supplying "
        "make_build_pool_projections(...) for build_pool_projections and a closure over "
        "build_app_state_from_predictions(...) with a real (or, for now, api.demo_data-sourced) "
        "squad/fixtures/prices for build_app_state, instead of invoking this CLI until A3 lands."
    )


if __name__ == "__main__":
    raise SystemExit(main())
