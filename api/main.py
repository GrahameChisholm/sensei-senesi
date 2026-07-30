"""FastAPI backend (BUILD_PLAN Phase 5.1) — thin endpoints over ``engine/``/``features/``, no
logic reimplemented. Every route reads the precomputed :class:`~api.state.AppState` (see that
module's docstring) and calls straight into a ``features/`` pure function; a ``ValueError`` from
any of them (bad gameweek, empty pool, ...) becomes a 400 rather than a 500, since these are all
caller-input problems, not server faults.
"""

from __future__ import annotations

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from api.schemas import (
    CaptaincyRecommendationOut,
    ChipEvaluationOut,
    FixtureDifficultyOut,
    MyTeamStateOut,
    SquadPlayerOut,
    TransferPlanOut,
    WildcardEvaluationOut,
)
from api.state import AppState, get_state
from features.captaincy import rank_captaincy_pool
from features.chips import (
    evaluate_bench_boost,
    evaluate_free_hit,
    evaluate_triple_captain,
    evaluate_wildcard,
)
from features.fixtures import project_fixture_difficulties
from features.transfers import find_transfer_candidates

app = FastAPI(title="FPL Assistant API", version="0.1.0")


@app.exception_handler(ValueError)
def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


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
