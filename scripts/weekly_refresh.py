"""Phase 6 weekly refresh job — the "keep it running and honest across the season" operational
loop the app depends on once real users are on it (BUILD_PLAN Phase 6): capture the pre-deadline
snapshot, regenerate projections, log predictions immutably, pull the Phase 4b odds snapshot, and
push the refreshed state to the API.

**Scope note — precisely why "regenerate projections" is a caller-supplied hook, not a direct
call into ``run_season.py``.** ``backtest.run_season.fit_fn``/``make_predict_fn`` themselves are
actually fine for live use as-is: ``fit_fn(training_history)`` fits on "everything strictly
before the target gameweek" (exactly what live deployment wants too), and the resulting
``predict_fn(fitted_state, gameweek)`` only reads pre-match features for that gameweek, never an
actual outcome, so there's no leakage concern in the fit/predict step itself. The real gap is one
level upstream: ``engineer_features`` builds its ``training_history``/``engineered`` frames from
``merged_gw`` — the vaastav GitHub CSV archive of **already-played** gameweeks (see
``fetch_vaastav_merged_gw``), not from ``engine.data.snapshots``/``engine.data.ingest``'s live
pre-deadline snapshot tables. No adapter from "live snapshot" to "engineer_features' expected
input shape" exists yet, and building one correctly (matching column names, join keys, and
point-in-time semantics for a gameweek that hasn't happened) is real integration work this script
doesn't attempt. ``build_pool_projections`` is a hook so that adapter can be supplied once it
exists, without this orchestration script needing to change.

**Odds are best-effort, never blocking.** A failed odds pull is caught and reported on the result
rather than aborting the refresh — the market overlay is explicitly a separate, optional layer
(BUILD_PLAN 4b), so projections and their logging must never depend on it succeeding.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from api.state import AppState, set_state
from backtest.prediction_log import current_model_version, log_predictions
from engine.data.fpl_client import FPLClient
from engine.data.ingest import capture_current_gameweek
from engine.data.snapshots import SnapshotManifest
from engine.data.understat_client import UnderstatClient
from market_overlay.odds_client import OddsClient, OddsClientError

logger = logging.getLogger(__name__)

BuildPoolProjections = Callable[[SnapshotManifest, int], pd.DataFrame]
BuildAppState = Callable[[pd.DataFrame], AppState]


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
    ``engine.pipeline.project_gameweek_pool``'s own return shape). ``build_app_state(predictions)``
    turns those predictions into a full :class:`~api.state.AppState` (squad, fixtures, prices,
    ...) for the API to start serving immediately.
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
    set_state(build_app_state(predictions))

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
        "scripts.weekly_refresh has no wired `build_pool_projections`/`build_app_state` yet -- "
        "see this module's docstring for why. Call run_weekly_refresh(...) directly with those "
        "hooks supplied instead of invoking this CLI until that wiring is confirmed safe."
    )


if __name__ == "__main__":
    raise SystemExit(main())
