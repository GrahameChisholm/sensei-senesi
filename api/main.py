"""FastAPI backend (Phase 4) — thin endpoints over ``features/``, no logic reimplemented. Every
mutation delegates to ``features.squad_rules``/``features.squad_draft``; a
:class:`~features.squad_rules.SquadRuleError` (or any other ``ValueError``) becomes a 400 with a
``RuleViolationOut`` body, never a 500 — these are all caller-input problems, not server faults.
"""

from __future__ import annotations

import os
from dataclasses import replace

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import schemas
from api.differentials_panel import build_differential_rows
from api.fixtures_view import DEFAULT_FIXTURE_TICKER_HORIZON, build_fixture_ticker_rows
from api.panel import build_panel_rows, build_team_fixture_map
from api.player_stats_panel import build_player_stats_rows
from api.squad_state import SquadState
from api.state import get_app_state, get_squad_state, set_squad_state
from engine.data.fpl_client import FPLClient, FPLClientError
from engine.data.team_state_builder import build_my_team_state
from features.differentials import DEFAULT_WINDOW_GAMEWEEKS, build_differentials
from features.player_stats import build_actual_stats_by_player
from features.players import get_player_detail
from features.squad_optimizer import PlayerCandidate, SquadOptimizerError, optimise_squad
from features.squad_points import projected_points
from features.squad_rules import (
    INITIAL_BUDGET,
    SQUAD_SIZE,
    RuleViolation,
    SquadRuleError,
    add_player,
    assemble_team_state,
    optimise_xi,
    remove_player,
    set_captain,
    set_vice_captain,
    validate_xi,
)
from features.team_state import MyTeamState, SquadPlayer

app = FastAPI(title="FPL Assistant API", version="0.1.0")

# The web app is served from Vite on a different origin (port 5173) than this API -- this is a
# single-user local tool with no cookie-based auth, so a fixed allowlist of local dev origins
# (overridable via env for e.g. a LAN-hosted setup) is sufficient.
_DEFAULT_CORS_ORIGINS = "http://localhost:5173,http://127.0.0.1:5173"
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("FPL_API_CORS_ORIGINS", _DEFAULT_CORS_ORIGINS).split(","),
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


@app.exception_handler(SquadRuleError)
def _squad_rule_error_handler(request: Request, exc: SquadRuleError) -> JSONResponse:
    return JSONResponse(
        status_code=400,
        content=schemas.RuleViolationOut(
            code=exc.violation.code,
            message=exc.violation.message,
            player_ids=list(exc.violation.player_ids),
        ).model_dump(),
    )


@app.exception_handler(SquadOptimizerError)
def _squad_optimizer_error_handler(request: Request, exc: SquadOptimizerError) -> JSONResponse:
    """A distinct 500, not 400: an infeasible optimizer call means the player pool (given the
    caller's locked picks, budget, and real prices) has no legal completion -- a data/state
    problem, not a malformed request."""
    return JSONResponse(
        status_code=500, content={"code": "infeasible", "message": str(exc), "player_ids": []}
    )


@app.exception_handler(ValueError)
def _value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(
        status_code=400, content={"code": "invalid", "message": str(exc), "player_ids": []}
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/teams", response_model=list[schemas.TeamOut])
def list_teams() -> list[schemas.TeamOut]:
    app_state = get_app_state()
    return [
        schemas.TeamOut(team_id=team_id, name=data["name"], short_name=data["short_name"])
        for team_id, data in app_state.teams.items()
    ]


@app.get("/fixtures", response_model=list[schemas.FixtureTickerRowOut])
def list_fixture_ticker(
    horizon: int = DEFAULT_FIXTURE_TICKER_HORIZON,
) -> list[schemas.FixtureTickerRowOut]:
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    app_state = get_app_state()
    gameweeks = list(range(app_state.gameweek, app_state.gameweek + horizon))
    rows = build_fixture_ticker_rows(app_state.fixtures, app_state.teams.keys(), gameweeks)
    return [
        schemas.FixtureTickerRowOut(
            team_id=row.team_id,
            gameweeks=[
                schemas.FixtureTickerCellOut(
                    gameweek=cell.gameweek,
                    fixtures=[
                        schemas.FixtureTickerCellFixtureOut(
                            opponent_id=entry.opponent_id,
                            is_home=entry.is_home,
                            difficulty=entry.difficulty,
                        )
                        for entry in cell.fixtures
                    ],
                )
                for cell in row.gameweeks
            ],
            average_difficulty=row.average_difficulty,
        )
        for row in rows
    ]


@app.get("/gameweek", response_model=schemas.GameweekOut)
def get_gameweek() -> schemas.GameweekOut:
    state = get_app_state()
    return schemas.GameweekOut(
        season=state.season,
        gameweek=state.gameweek,
        deadline_time=state.deadline_time.isoformat(),
        deadline_passed=state.deadline_passed,
        generated_at=state.generated_at.isoformat(),
        model_version=state.model_version,
        horizon_gameweeks=state.horizon_gameweeks,
    )


def _squad_player_out(player: SquadPlayer) -> schemas.SquadPlayerOut:
    return schemas.SquadPlayerOut(
        player_id=player.player_id, position=player.position, price=player.price
    )


def _squad_out() -> schemas.SquadOut:
    state = get_squad_state()
    is_complete = (
        len(state.squad) == SQUAD_SIZE
        and state.captain_id is not None
        and state.vice_captain_id is not None
    )
    budget_remaining = state.budget_ceiling - sum(player.price for player in state.squad)
    return schemas.SquadOut(
        squad=[_squad_player_out(player) for player in state.squad],
        starting_xi=list(state.starting_xi),
        bench_order=list(state.bench_order),
        captain_id=state.captain_id,
        vice_captain_id=state.vice_captain_id,
        is_complete=is_complete,
        budget_ceiling=state.budget_ceiling,
        budget_remaining=budget_remaining,
    )


def _reconcile_after_squad_change(
    state: SquadState, new_squad: tuple[SquadPlayer, ...]
) -> SquadState:
    """After an add/remove changes the squad's membership, keep starting_xi/bench_order/captain/
    vice consistent with it. Below 15 players there's no legal starting XI to speak of, so those
    fields are cleared; back at 15, the best XI is auto-derived (:func:`~features.squad_rules
    .assemble_team_state`), preferring to keep the existing captain/vice if they're still eligible
    so an incidental swap elsewhere doesn't reset the armband."""
    if len(new_squad) != SQUAD_SIZE:
        return replace(
            state,
            squad=new_squad,
            starting_xi=(),
            bench_order=(),
            captain_id=None,
            vice_captain_id=None,
        )
    app_state = get_app_state()
    team_state = assemble_team_state(
        new_squad,
        app_state.expected_points(),
        app_state.team_id_by_player,
        budget=state.budget_ceiling,
        preferred_captain_id=state.captain_id,
        preferred_vice_captain_id=state.vice_captain_id,
    )
    return replace(
        state,
        squad=team_state.squad,
        starting_xi=team_state.starting_xi,
        bench_order=team_state.bench_order,
        captain_id=team_state.captain_id,
        vice_captain_id=team_state.vice_captain_id,
    )


def _require_team_state() -> tuple[SquadState, MyTeamState]:
    state = get_squad_state()
    if len(state.squad) != SQUAD_SIZE or state.captain_id is None or state.vice_captain_id is None:
        raise HTTPException(400, "no complete 15-player squad yet")
    team_state = MyTeamState(
        squad=state.squad,
        starting_xi=state.starting_xi,
        bench_order=state.bench_order,
        captain_id=state.captain_id,
        vice_captain_id=state.vice_captain_id,
        mini_league_ids=state.mini_league_ids,
    )
    return state, team_state


def _save_team_state(state: SquadState, team_state: MyTeamState) -> None:
    set_squad_state(
        replace(
            state,
            squad=team_state.squad,
            starting_xi=team_state.starting_xi,
            bench_order=team_state.bench_order,
            captain_id=team_state.captain_id,
            vice_captain_id=team_state.vice_captain_id,
        )
    )


@app.get("/squad", response_model=schemas.SquadOut)
def get_squad() -> schemas.SquadOut:
    return _squad_out()


@app.post("/squad/players", response_model=schemas.SquadOut)
def add_squad_player(body: schemas.AddPlayerIn) -> schemas.SquadOut:
    state = get_squad_state()
    app_state = get_app_state()
    player = SquadPlayer(body.player_id, body.position, body.price)
    new_squad = add_player(state.squad, player, app_state.team_id_by_player, state.budget_ceiling)
    set_squad_state(_reconcile_after_squad_change(state, new_squad))
    return _squad_out()


@app.delete("/squad/players/{player_id}", response_model=schemas.SquadOut)
def remove_squad_player(player_id: int) -> schemas.SquadOut:
    state = get_squad_state()
    new_squad = remove_player(state.squad, player_id)
    set_squad_state(_reconcile_after_squad_change(state, new_squad))
    return _squad_out()


@app.delete("/squad/players", response_model=schemas.SquadOut)
def clear_squad() -> schemas.SquadOut:
    """Sandbox reset: empties the squad and resets the personal budget ceiling back to the
    classic £100m (:data:`~features.squad_rules.INITIAL_BUDGET`) — the team-selection page is a
    sandbox for exploring squad ideas, constrained only by the classic legality rules, not by what
    the current squad happens to be worth."""
    mini_league_ids = get_squad_state().mini_league_ids
    set_squad_state(SquadState(mini_league_ids=mini_league_ids))
    return _squad_out()


@app.post("/squad/import", response_model=schemas.SquadOut)
def import_squad(payload: schemas.ImportSquadIn) -> schemas.SquadOut:
    """Import a real FPL manager's current squad by their team (entry) ID.

    Fetches live from the official FPL API at request time -- a team ID is per-request user input
    with no precomputable form, unlike projections (the API never fetches or computes those on
    request; it doesn't cover reading a manager's own squad by their own ID). Overwrites whatever
    squad currently exists and recomputes the personal budget ceiling from this squad's total
    current value, floored at the classic £100m, since re-importing is meant to be usable any time
    as a re-sync, not just once at onboarding.

    Picks are fetched for ``entry["current_event"]``, not ``app_state.gameweek``: FPL only has a
    ``picks`` record for a gameweek once its deadline has passed, or the manager has explicitly
    saved a transfer for it -- before that, the upcoming gameweek's picks endpoint 404s and
    ``current_event`` is the most recent gameweek that does have one (the manager's real current
    squad either way, since nothing has changed since).
    """
    app_state = get_app_state()
    with FPLClient() as client:
        try:
            entry = client.get_entry(payload.team_id)
            picks = client.get_entry_picks(payload.team_id, entry["current_event"])
            elements = pd.DataFrame(client.get_bootstrap_static()["elements"])
        except FPLClientError as exc:
            raise ValueError(f"could not import FPL team {payload.team_id}: {exc}") from exc

    team_state = build_my_team_state(picks, elements, app_state.team_id_by_player)
    budget_ceiling = max(sum(player.price for player in team_state.squad), INITIAL_BUDGET)
    mini_league_ids = get_squad_state().mini_league_ids
    set_squad_state(
        SquadState(
            squad=team_state.squad,
            starting_xi=team_state.starting_xi,
            bench_order=team_state.bench_order,
            captain_id=team_state.captain_id,
            vice_captain_id=team_state.vice_captain_id,
            mini_league_ids=mini_league_ids,
            budget_ceiling=budget_ceiling,
        )
    )
    return _squad_out()


@app.post("/squad/captain", response_model=schemas.SquadOut)
def set_squad_captain(body: schemas.CaptainIn) -> schemas.SquadOut:
    state, team_state = _require_team_state()
    if body.role == "captain":
        new_team_state = set_captain(team_state, body.player_id)
    elif body.role == "vice":
        new_team_state = set_vice_captain(team_state, body.player_id)
    else:
        raise HTTPException(400, "role must be 'captain' or 'vice'")
    _save_team_state(state, new_team_state)
    return _squad_out()


@app.post("/squad/bench-order", response_model=schemas.SquadOut)
def set_squad_bench_order(body: schemas.BenchOrderIn) -> schemas.SquadOut:
    """Set the starting XI/bench partition directly — covers moving a player between XI and
    bench, and reordering the bench, in one call."""
    state, team_state = _require_team_state()
    position_by_player = {player.player_id: player.position for player in state.squad}
    violations = validate_xi(body.starting_xi, position_by_player)
    if violations:
        raise SquadRuleError(violations[0])
    squad_ids = set(team_state.player_ids)
    new_xi, new_bench = set(body.starting_xi), set(body.bench_order)
    if new_xi | new_bench != squad_ids or new_xi & new_bench:
        raise SquadRuleError(
            RuleViolation("xi_shape", "starting_xi + bench_order must exactly partition the squad")
        )
    new_team_state = replace(
        team_state, starting_xi=tuple(body.starting_xi), bench_order=tuple(body.bench_order)
    )
    _save_team_state(state, new_team_state)
    return _squad_out()


@app.post("/squad/optimise-xi", response_model=schemas.SquadOut)
def optimise_lineup() -> schemas.SquadOut:
    """Re-derive the best legal XI/bench from the current 15 — applies immediately, since it only
    ever rearranges players already owned."""
    state, team_state = _require_team_state()
    app_state = get_app_state()
    new_team_state = optimise_xi(team_state, app_state.expected_points())
    _save_team_state(state, new_team_state)
    return _squad_out()


@app.post("/squad/optimise", response_model=schemas.SquadOut)
def auto_build_squad(body: schemas.OptimiseIn) -> schemas.SquadOut:
    """The best-possible-squad solver: keeps whatever's currently in the squad and fills any
    remaining slots with the legal combination that maximizes projected points (an empty squad and
    a squad missing a few players are the same call — nothing already picked is ever locked
    differently). ``objective="full_squad"`` should be passed when Bench Boost is active, since
    bench points count then and are worth spending budget on.
    """
    state = get_squad_state()
    app_state = get_app_state()
    candidates = [
        PlayerCandidate(
            player_id=player_id,
            position=app_state.position_by_player[player_id],
            team_id=app_state.team_id_by_player[player_id],
            price=app_state.buy_prices[player_id],
            expected_points=horizon.horizon_total_points,
        )
        for player_id, horizon in app_state.projections.items()
        if player_id in app_state.buy_prices
    ]
    locked_player_ids = frozenset(player.player_id for player in state.squad)
    result = optimise_squad(
        candidates,
        locked_player_ids=locked_player_ids,
        objective=body.objective,
        captain_multiplier=body.captain_multiplier,
        budget=state.budget_ceiling,
    )
    set_squad_state(
        replace(
            state,
            squad=result.squad,
            starting_xi=result.starting_xi,
            bench_order=result.bench_order,
            captain_id=result.captain_id,
            vice_captain_id=result.vice_captain_id,
        )
    )
    return _squad_out()


@app.get("/squad/points", response_model=schemas.SquadPointsOut)
def get_squad_points(
    chip: str | None = None, horizon: int = 1, gameweek: int | None = None
) -> schemas.SquadPointsOut:
    """Pass ``chip`` (``"bench_boost"`` or ``"triple_captain"``) to preview points under that
    toggle — it's stateless, nothing is "spent" or remembered between calls. Pass ``gameweek`` to
    rescore the saved starting XI as of a single future gameweek within the current horizon,
    instead of summing ``horizon`` gameweeks from now."""
    _, team_state = _require_team_state()
    app_state = get_app_state()
    if gameweek is not None:
        if gameweek not in app_state.horizon_gameweeks:
            raise ValueError(
                f"gameweek {gameweek} is outside the current horizon "
                f"{app_state.horizon_gameweeks}"
            )
        gameweeks = [gameweek]
    else:
        gameweeks = app_state.horizon_gameweeks[: max(horizon, 1)] or [app_state.gameweek]
    result = projected_points(team_state, app_state.projections, gameweeks, chip=chip)
    return schemas.SquadPointsOut(
        total=result.total,
        starting_xi_points=result.starting_xi_points,
        bench_points=result.bench_points,
        captain_bonus=result.captain_bonus,
        per_player={str(player_id): points for player_id, points in result.per_player.items()},
        per_gameweek={str(gw): points for gw, points in result.per_gameweek.items()},
        missing_player_ids=list(result.missing_player_ids),
    )


# --- player selection panel (Phase 6) ------------------------------------------------------------


@app.get("/players", response_model=list[schemas.PlayerPanelRowOut])
def list_players(
    position: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    search: str | None = None,
) -> list[schemas.PlayerPanelRowOut]:
    app_state = get_app_state()
    player_names = {player_id: data["web_name"] for player_id, data in app_state.players.items()}
    low_confidence_ids = {
        player_id for player_id, data in app_state.players.items() if data.get("low_confidence")
    }
    fixture_map = build_team_fixture_map(app_state.fixtures)
    rows = build_panel_rows(
        app_state.projections,
        player_names,
        app_state.buy_prices,
        app_state.team_id_by_player,
        low_confidence_ids,
        fixture_map,
        app_state.horizon_gameweeks,
        search=search,
        position=position,
        max_price=max_price,
    )
    if min_price is not None:
        rows = [row for row in rows if row.price is not None and row.price >= min_price]
    return [
        schemas.PlayerPanelRowOut(
            player_id=row.player_id,
            name=row.name,
            team_id=row.team_id,
            position=row.position,
            price=row.price,
            low_confidence=row.low_confidence,
            fixtures=[
                schemas.FixtureCellOut(
                    gameweek=cell.gameweek,
                    opponent_id=cell.opponent_id,
                    is_home=cell.is_home,
                    expected_points=cell.expected_points,
                )
                for cell in row.fixtures
            ],
        )
        for row in rows
    ]


@app.get("/players/stats", response_model=list[schemas.PlayerStatsRowOut])
def list_player_stats(gameweek_from: int, gameweek_to: int) -> list[schemas.PlayerStatsRowOut]:
    """Every player with recorded actual stats in ``[gameweek_from, gameweek_to]``, plus their
    predicted expected points for each of the app's 3-gameweek horizon. Search, team, position,
    and price filtering (D14) all happen client-side over this one bulk response.

    Registered *before* ``/players/{player_id}`` below -- Starlette matches routes in
    registration order, so this literal path must come first or ``{player_id}`` would swallow
    the request and fail trying to parse "stats" as an int.
    """
    if gameweek_from > gameweek_to:
        raise HTTPException(400, "gameweek_from must be less than or equal to gameweek_to")

    app_state = get_app_state()
    player_names = {pid: data["web_name"] for pid, data in app_state.players.items()}
    low_confidence_ids = {
        pid for pid, data in app_state.players.items() if data.get("low_confidence")
    }
    ownership_by_player = {
        pid: data.get("selected_by_percent") for pid, data in app_state.players.items()
    }
    fixture_map = build_team_fixture_map(app_state.fixtures)

    actual_stats_by_player = build_actual_stats_by_player(
        app_state.player_history,
        app_state.position_by_player,
        gameweek_from,
        gameweek_to,
        ownership_by_player,
    )
    rows = build_player_stats_rows(
        actual_stats_by_player,
        app_state.projections,
        player_names,
        app_state.buy_prices,
        app_state.team_id_by_player,
        app_state.position_by_player,
        low_confidence_ids,
        fixture_map,
        app_state.horizon_gameweeks,
    )
    return [
        schemas.PlayerStatsRowOut(
            player_id=row.player_id,
            name=row.name,
            team_id=row.team_id,
            position=row.position,
            price=row.price,
            low_confidence=row.low_confidence,
            actuals=schemas.ActualStatsOut(
                gameweek_from=row.actuals.gameweek_from,
                gameweek_to=row.actuals.gameweek_to,
                apps=row.actuals.apps,
                minutes=row.actuals.minutes,
                goals_scored=row.actuals.goals_scored,
                assists=row.actuals.assists,
                clean_sheets=row.actuals.clean_sheets,
                goals_conceded=row.actuals.goals_conceded,
                own_goals=row.actuals.own_goals,
                penalties_missed=row.actuals.penalties_missed,
                penalties_saved=row.actuals.penalties_saved,
                saves=row.actuals.saves,
                bonus=row.actuals.bonus,
                yellow_cards=row.actuals.yellow_cards,
                red_cards=row.actuals.red_cards,
                total_points=row.actuals.total_points,
                expected_goals=row.actuals.expected_goals,
                expected_assists=row.actuals.expected_assists,
                expected_goal_involvements=row.actuals.expected_goal_involvements,
                expected_goals_conceded=row.actuals.expected_goals_conceded,
                points_breakdown=schemas.ComponentBreakdownOut(
                    appearance=row.actuals.points_breakdown.appearance,
                    goals=row.actuals.points_breakdown.goals,
                    assists=row.actuals.points_breakdown.assists,
                    clean_sheet=row.actuals.points_breakdown.clean_sheet,
                    goals_conceded=row.actuals.points_breakdown.goals_conceded,
                    defensive_contribution=row.actuals.points_breakdown.defensive_contribution,
                    saves=row.actuals.points_breakdown.saves,
                    bonus=row.actuals.points_breakdown.bonus,
                    cards=row.actuals.points_breakdown.cards,
                    penalty_misses=row.actuals.points_breakdown.penalty_misses,
                    own_goals=row.actuals.points_breakdown.own_goals,
                    total=row.actuals.points_breakdown.total,
                ),
                selected_by_percent=row.actuals.selected_by_percent,
                small_sample=row.actuals.small_sample,
            ),
            fixtures=[
                schemas.FixtureCellOut(
                    gameweek=cell.gameweek,
                    opponent_id=cell.opponent_id,
                    is_home=cell.is_home,
                    expected_points=cell.expected_points,
                )
                for cell in row.fixtures
            ],
        )
        for row in rows
    ]


@app.get("/players/differentials", response_model=schemas.DifferentialsResponseOut)
def list_differentials(
    window: int = DEFAULT_WINDOW_GAMEWEEKS,
    max_ownership: float | None = None,
    hide_owned: bool = True,
) -> schemas.DifferentialsResponseOut:
    """DIFFERENTIALS_PLAN Phase 3: players sustainedly outperforming their own position/price
    bracket among low-owned players, ranked (client-side, D10) over a set of independent,
    unblended columns rather than one composite score.

    ``window`` is the requested gameweek count, clamped to whatever has actually been played this
    season (D5/D6) -- the response's own ``window`` field reports the resolved range, since it
    frequently differs from what was requested early in a season. ``max_ownership`` and
    ``hide_owned`` are the two differentiating filters (D1): a player already in the committed
    squad is excluded by default rather than shown as a "differential" against yourself.

    Registered *before* ``/players/{player_id}`` below, for the same route-order reason
    ``/players/stats`` above already documents -- a literal path registered after the
    parameterised one gets swallowed by it.
    """
    app_state = get_app_state()
    player_names = {pid: data["web_name"] for pid, data in app_state.players.items()}
    ownership_by_player = {
        pid: data.get("selected_by_percent") for pid, data in app_state.players.items()
    }

    resolved_window, differentials = build_differentials(
        app_state.player_history,
        app_state.position_by_player,
        app_state.buy_prices,
        latest_played_gameweek=app_state.gameweek - 1,
        window_gameweeks=window,
        current_ownership_by_player=ownership_by_player,
        max_ownership_percent=max_ownership,
    )

    if hide_owned:
        squad_state = get_squad_state()
        owned_ids = {player.player_id for player in squad_state.squad}
        differentials = [d for d in differentials if d.player_id not in owned_ids]

    fixture_map = build_team_fixture_map(app_state.fixtures)
    rows = build_differential_rows(
        differentials,
        player_names,
        app_state.team_id_by_player,
        app_state.projections,
        fixture_map,
        app_state.horizon_gameweeks,
    )
    return schemas.DifferentialsResponseOut(
        window=schemas.DifferentialsWindowOut(
            gameweek_from=resolved_window.gameweek_from,
            gameweek_to=resolved_window.gameweek_to,
            requested_gameweeks=resolved_window.requested_gameweeks,
        ),
        rows=[
            schemas.DifferentialRowOut(
                player_id=row.differential.player_id,
                name=row.name,
                team_id=row.team_id,
                position=row.differential.position,
                price=row.differential.price,
                minutes=row.differential.minutes,
                apps_in_window=row.differential.apps_in_window,
                starts_in_window=row.differential.starts_in_window,
                points_per_90=row.differential.points_per_90,
                shrunk_points_per_90=row.differential.shrunk_points_per_90,
                bracket_median_points_per_90=row.differential.bracket_median_points_per_90,
                surplus_vs_bracket=row.differential.surplus_vs_bracket,
                confidence=row.differential.confidence.value,
                xgi_per_90=row.differential.xgi_per_90,
                goals_assists_per_90=row.differential.goals_assists_per_90,
                luck_gap=row.differential.luck_gap,
                defensive_contribution_per_90=row.differential.defensive_contribution_per_90,
                bps_per_90=row.differential.bps_per_90,
                return_frequency=row.differential.return_frequency,
                points_variance=row.differential.points_variance,
                recent_vs_earlier_points_per_90=row.differential.recent_vs_earlier_points_per_90,
                minutes_trend=row.differential.minutes_trend,
                current_ownership_percent=row.differential.current_ownership_percent,
                ownership_trend_pct_per_gw=row.differential.ownership_trend_pct_per_gw,
                net_transfers_per_gw=row.differential.net_transfers_per_gw,
                archetype=row.differential.archetype.value,
                fixtures=[
                    schemas.FixtureCellOut(
                        gameweek=cell.gameweek,
                        opponent_id=cell.opponent_id,
                        is_home=cell.is_home,
                        expected_points=cell.expected_points,
                    )
                    for cell in row.fixtures
                ],
            )
            for row in rows
        ],
    )


@app.get("/players/{player_id}", response_model=schemas.PlayerDetailOut)
def get_player(player_id: int, gameweek: int | None = None) -> schemas.PlayerDetailOut:
    app_state = get_app_state()
    player_names = {pid: data["web_name"] for pid, data in app_state.players.items()}
    try:
        detail = get_player_detail(
            player_id, app_state.projections, player_names, app_state.buy_prices, gameweek
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc

    player_meta = app_state.players.get(player_id, {})
    return schemas.PlayerDetailOut(
        player_id=detail.player_id,
        name=detail.name,
        position=detail.position,
        price=detail.price,
        team_id=player_meta.get("team_id"),
        low_confidence=player_meta.get("low_confidence", False),
        gameweek=detail.gameweek,
        expected_points=detail.expected_points,
        breakdown=schemas.ComponentBreakdownOut(
            appearance=detail.breakdown.appearance,
            goals=detail.breakdown.goals,
            assists=detail.breakdown.assists,
            clean_sheet=detail.breakdown.clean_sheet,
            goals_conceded=detail.breakdown.goals_conceded,
            defensive_contribution=detail.breakdown.defensive_contribution,
            saves=detail.breakdown.saves,
            bonus=detail.breakdown.bonus,
            cards=detail.breakdown.cards,
            penalty_misses=detail.breakdown.penalty_misses,
            own_goals=detail.breakdown.own_goals,
            total=detail.breakdown.total,
        ),
        floor=detail.floor,
        ceiling=detail.ceiling,
        prob_big_haul=detail.prob_big_haul,
    )
