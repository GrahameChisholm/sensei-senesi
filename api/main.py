"""FastAPI backend (BUILD_PLAN Phase 5.1) — thin endpoints over ``engine/``/``features/``, no
logic reimplemented. Every route reads the precomputed :class:`~api.state.AppState` (see that
module's docstring) and calls straight into a ``features/`` pure function; a ``ValueError`` from
any of them (bad gameweek, empty pool, ...) becomes a 400 rather than a 500, since these are all
caller-input problems, not server faults.
"""

from __future__ import annotations

import os

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy import Engine

from api.model_performance import ModelPerformanceData, get_default_model_performance
from api.schemas import (
    CaptaincyRecommendationOut,
    ChipEvaluationOut,
    DataStatusOut,
    FixtureDifficultyOut,
    ModelPerformanceOut,
    MyTeamStateOut,
    PlayerDetailOut,
    PlayerSummaryOut,
    SettingsIn,
    SettingsOut,
    SquadPlayerOut,
    TransferPlanOut,
    WildcardEvaluationOut,
)
from api.settings import AppSettingsData, get_db_engine, get_settings, save_settings
from api.state import AppState, get_state
from features.captaincy import rank_captaincy_pool
from features.chips import (
    evaluate_bench_boost,
    evaluate_free_hit,
    evaluate_triple_captain,
    evaluate_wildcard,
)
from features.fixtures import project_fixture_difficulties
from features.players import get_player_detail, search_players
from features.transfers import find_transfer_candidates

app = FastAPI(title="FPL Assistant API", version="0.1.0")

# The web app (web/) is served from Vite on a different origin (port 5173) than this API (port
# 8000), so without CORS the browser blocks every request outright -- this is a single-user local
# tool with no cookie-based auth, so a fixed allowlist of local dev origins (overridable via env
# for e.g. a LAN-hosted setup) is sufficient; there's no session to leak cross-site.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FPL_API_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(","),
    allow_methods=["GET", "PUT"],
    allow_headers=["*"],
)


@app.exception_handler(ValueError)
def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/data-status", response_model=DataStatusOut)
def get_data_status(state: AppState = Depends(get_state)) -> DataStatusOut:
    """The "data as of" freshness indicator (A6/BUILD_PLAN Phase 6) -- `generated_at` is `None`
    for `api.demo_data`'s synthetic state (there is no real "as of" time for numbers that were
    never actually computed from live data), real for anything built via
    `scripts.weekly_refresh.build_app_state_from_predictions`."""
    return DataStatusOut(
        generated_at=state.generated_at.isoformat() if state.generated_at else None,
        is_demo_data=state.generated_at is None,
    )


@app.get("/team", response_model=MyTeamStateOut)
def get_team(state: AppState = Depends(get_state)) -> MyTeamStateOut:
    team = state.my_team
    return MyTeamStateOut(
        squad=[SquadPlayerOut.model_validate(player) for player in team.squad],
        starting_xi=list(team.starting_xi),
        bench_order=list(team.bench_order),
        captain_id=team.captain_id,
        vice_captain_id=team.vice_captain_id,
        bank=team.bank,
        free_transfers=team.free_transfers,
        # Sorted for a stable response -- `chips_remaining` is a frozenset, whose iteration order
        # isn't something callers should ever depend on.
        chips_remaining=sorted(team.chips_remaining),
        total_sell_value=team.total_sell_value,
    )


@app.get("/fixtures", response_model=list[FixtureDifficultyOut])
def get_fixtures(
    gameweek: int | None = None, state: AppState = Depends(get_state)
) -> list[FixtureDifficultyOut]:
    """Every team's custom difficulty rating (BUILD_PLAN 4), optionally filtered to one
    gameweek."""
    fixtures = state.fixtures
    if gameweek is not None:
        fixtures = [f for f in fixtures if f.gameweek == gameweek]
    results = project_fixture_difficulties(
        fixtures, state.team_rates, state.league_avg_xg_per_90, state.league_avg_xga_per_90
    )
    return [FixtureDifficultyOut.model_validate(r) for r in results]


@app.get("/captaincy", response_model=CaptaincyRecommendationOut)
def get_captaincy(
    gameweek: int, state: AppState = Depends(get_state)
) -> CaptaincyRecommendationOut:
    gw_projections = [
        horizon.gameweeks[gameweek]
        for horizon in state.projections.values()
        if gameweek in horizon.gameweeks
    ]
    if not gw_projections:
        raise HTTPException(status_code=400, detail=f"no projections for gameweek {gameweek}")
    result = rank_captaincy_pool(state.my_team, gw_projections)
    return CaptaincyRecommendationOut.model_validate(result)


def _split_owned(state: AppState) -> tuple[dict, dict]:
    owned_ids = set(state.my_team.player_ids)
    current = {pid: h for pid, h in state.projections.items() if pid in owned_ids}
    pool = {pid: h for pid, h in state.projections.items() if pid not in owned_ids}
    return current, pool


@app.get("/transfers", response_model=TransferPlanOut)
def get_transfers(state: AppState = Depends(get_state)) -> TransferPlanOut:
    current, pool = _split_owned(state)
    plan = find_transfer_candidates(state.my_team, current, pool, state.buy_prices)
    return TransferPlanOut.model_validate(plan)


@app.get("/chips/bench-boost", response_model=ChipEvaluationOut)
def get_bench_boost(gameweek: int, state: AppState = Depends(get_state)) -> ChipEvaluationOut:
    result = evaluate_bench_boost(state.my_team, state.projections, gameweek)
    return ChipEvaluationOut.model_validate(result)


@app.get("/chips/triple-captain", response_model=ChipEvaluationOut)
def get_triple_captain(gameweek: int, state: AppState = Depends(get_state)) -> ChipEvaluationOut:
    result = evaluate_triple_captain(state.my_team, state.projections, gameweek)
    return ChipEvaluationOut.model_validate(result)


@app.get("/chips/free-hit", response_model=ChipEvaluationOut)
def get_free_hit(gameweek: int, state: AppState = Depends(get_state)) -> ChipEvaluationOut:
    result = evaluate_free_hit(
        state.my_team,
        state.team_id_by_player,
        state.fixtures,
        state.horizon_gameweeks,
        gameweek,
    )
    return ChipEvaluationOut.model_validate(result)


@app.get("/chips/wildcard", response_model=WildcardEvaluationOut)
def get_wildcard(state: AppState = Depends(get_state)) -> WildcardEvaluationOut:
    current, pool = _split_owned(state)
    result = evaluate_wildcard(state.my_team, current, pool, state.buy_prices)
    return WildcardEvaluationOut.model_validate(result)


@app.get("/settings", response_model=SettingsOut)
def get_app_settings(db_engine: Engine = Depends(get_db_engine)) -> SettingsOut:
    """FPL team ID, mini-league IDs, and planning-horizon default (BUILD_PLAN 5.2's Settings
    screen) — persisted independently of `AppState`, so they survive a weekly refresh/restart."""
    return SettingsOut.model_validate(get_settings(db_engine))


@app.put("/settings", response_model=SettingsOut)
def put_app_settings(
    payload: SettingsIn, db_engine: Engine = Depends(get_db_engine)
) -> SettingsOut:
    data = AppSettingsData(
        fpl_team_id=payload.fpl_team_id,
        mini_league_ids=tuple(payload.mini_league_ids),
        planning_horizon_gameweeks=payload.planning_horizon_gameweeks,
    )
    save_settings(db_engine, data)
    return SettingsOut.model_validate(data)


@app.get("/players", response_model=list[PlayerSummaryOut])
def get_players(
    search: str | None = None,
    position: str | None = None,
    max_price: int | None = None,
    gameweek: int | None = None,
    state: AppState = Depends(get_state),
) -> list[PlayerSummaryOut]:
    results = search_players(
        state.projections,
        state.player_names,
        state.buy_prices,
        search=search,
        position=position,
        max_price=max_price,
        gameweek=gameweek,
    )
    return [PlayerSummaryOut.model_validate(r) for r in results]


@app.get("/players/{player_id}", response_model=PlayerDetailOut)
def get_player(
    player_id: int, gameweek: int | None = None, state: AppState = Depends(get_state)
) -> PlayerDetailOut:
    try:
        detail = get_player_detail(
            player_id, state.projections, state.player_names, state.buy_prices, gameweek=gameweek
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return PlayerDetailOut.model_validate(detail)


@app.get("/model-performance", response_model=ModelPerformanceOut)
def get_model_performance(
    report: ModelPerformanceData = Depends(get_default_model_performance),
) -> ModelPerformanceOut:
    """The stored season-backtest headline report (BUILD_PLAN 5.2's Model Performance screen) --
    `headline` is `null` if no backtest has ever been run and stored, which is itself a real,
    honest state to show rather than an error."""
    return ModelPerformanceOut.model_validate(report)
