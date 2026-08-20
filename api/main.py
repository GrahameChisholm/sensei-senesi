"""FastAPI backend (Phase 4) — thin endpoints over ``features/``, no logic reimplemented. Every
mutation delegates to ``features.squad_rules``/``features.squad_draft``; a
:class:`~features.squad_rules.SquadRuleError` (or any other ``ValueError``) becomes a 400 with a
``RuleViolationOut`` body, never a 500 — these are all caller-input problems, not server faults.
"""

from __future__ import annotations

import os
from dataclasses import replace

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import schemas
from api.fixtures_view import DEFAULT_FIXTURE_TICKER_HORIZON, build_fixture_ticker_rows
from api.panel import build_panel_rows, build_team_fixture_map
from api.player_stats_panel import build_player_stats_rows
from api.state import (
    get_app_state,
    get_build_picks,
    get_squad_state,
    set_build_picks,
    set_squad_state,
)
from features.chip_calendar import available_chips_this_gameweek
from features.player_stats import build_actual_stats_by_player
from features.players import get_player_detail
from features.squad_draft import (
    apply_optimise_xi_to_draft,
    apply_reorder_bench_to_draft,
    apply_set_captain_to_draft,
    apply_set_vice_captain_to_draft,
    apply_substitute_to_draft,
    apply_transfer_to_draft,
    confirm_draft,
    confirm_initial_squad,
    open_draft,
    set_draft_chip,
)
from features.squad_points import projected_points
from features.squad_rules import (
    MAX_PER_CLUB,
    RuleViolation,
    SquadRuleError,
    optimise_xi,
    set_captain,
    set_vice_captain,
    transfer,
)
from features.team_state import SquadPlayer
from features.transfers import TransferCandidate, find_transfer_candidates

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
    )


def _squad_player_out(player: SquadPlayer) -> schemas.SquadPlayerOut:
    return schemas.SquadPlayerOut(
        player_id=player.player_id,
        position=player.position,
        purchase_price=player.purchase_price,
        current_price=player.current_price,
        sell_price=player.sell_price,
    )


def _team_state_out(team_state) -> schemas.TeamStateOut:
    return schemas.TeamStateOut(
        squad=[_squad_player_out(player) for player in team_state.squad],
        starting_xi=list(team_state.starting_xi),
        bench_order=list(team_state.bench_order),
        captain_id=team_state.captain_id,
        vice_captain_id=team_state.vice_captain_id,
        bank=team_state.bank,
        free_transfers=team_state.free_transfers,
        chips_remaining=sorted(team_state.chips_remaining),
    )


def _squad_out(last_hit_cost: int | None = None) -> schemas.SquadOut:
    app_state = get_app_state()
    committed, pending = get_squad_state()

    build_picks_out = None
    if committed.team_state is None:
        build_picks_out = [_squad_player_out(player) for player in get_build_picks()]

    draft_out = None
    if pending is not None:
        draft_out = schemas.DraftOut(
            base_gameweek=pending.base_gameweek,
            working_state=_team_state_out(pending.working_state),
            transfers_made=pending.transfers_made,
            chip=pending.chip,
        )

    chips_available = sorted(
        available_chips_this_gameweek(committed.chip_usage, app_state.gameweek)
    )

    return schemas.SquadOut(
        is_complete=committed.team_state is not None,
        committed=(
            _team_state_out(committed.team_state) if committed.team_state is not None else None
        ),
        build_picks=build_picks_out,
        active_chip=committed.active_chip,
        active_chip_gameweek=committed.active_chip_gameweek,
        chips_available=chips_available,
        draft=draft_out,
        last_hit_cost=last_hit_cost,
    )


@app.get("/squad", response_model=schemas.SquadOut)
def get_squad() -> schemas.SquadOut:
    return _squad_out()


# --- build mode (D6/D23): assembling the very first squad from an empty £100m ------------------


@app.post("/squad/build/players", response_model=schemas.SquadOut)
def add_build_player(body: schemas.AddPlayerIn) -> schemas.SquadOut:
    committed, _ = get_squad_state()
    if committed.team_state is not None:
        raise HTTPException(400, "a squad already exists -- use /squad/draft to edit it instead")
    picks = get_build_picks()
    if any(player.player_id == body.player_id for player in picks):
        raise SquadRuleError(
            RuleViolation(
                "duplicate", f"player {body.player_id} is already picked", (body.player_id,)
            )
        )
    set_build_picks([*picks, SquadPlayer(body.player_id, body.position, body.price, body.price)])
    return _squad_out()


@app.delete("/squad/build/players/{player_id}", response_model=schemas.SquadOut)
def remove_build_player(player_id: int) -> schemas.SquadOut:
    committed, _ = get_squad_state()
    if committed.team_state is not None:
        raise HTTPException(400, "a squad already exists")
    picks = get_build_picks()
    set_build_picks([player for player in picks if player.player_id != player_id])
    return _squad_out()


@app.post("/squad/build/confirm", response_model=schemas.SquadOut)
def confirm_build(body: schemas.ConfirmSquadIn) -> schemas.SquadOut:
    """The very first commit (D23) — requires an explicit action once the picks satisfy every
    rule; it never happens automatically."""
    committed, _ = get_squad_state()
    if committed.team_state is not None:
        raise HTTPException(400, "a squad already exists")
    picks_by_id = {player.player_id: player for player in get_build_picks()}
    if set(body.player_ids) != set(picks_by_id):
        raise SquadRuleError(
            RuleViolation(
                "incomplete_squad",
                "confirmed player_ids must exactly match the current build picks",
            )
        )
    squad = tuple(picks_by_id[player_id] for player_id in body.player_ids)

    app_state = get_app_state()
    new_committed = confirm_initial_squad(
        squad=squad,
        starting_xi=tuple(body.starting_xi),
        bench_order=tuple(body.bench_order),
        captain_id=body.captain_id,
        vice_captain_id=body.vice_captain_id,
        team_id_by_player=app_state.team_id_by_player,
        gameweek=app_state.gameweek,
    )
    set_squad_state(new_committed, None)
    set_build_picks([])
    return _squad_out()


# --- editing an existing squad (D16): preview-then-confirm --------------------------------------


def _require_committed_squad():
    committed, _ = get_squad_state()
    if committed.team_state is None:
        raise HTTPException(400, "no committed squad yet -- build and confirm one first")
    return committed


def _require_open_draft():
    committed, pending = get_squad_state()
    if committed.team_state is None:
        raise HTTPException(400, "no committed squad yet -- build and confirm one first")
    if pending is None:
        raise SquadRuleError(
            RuleViolation("no_pending_draft", "no draft is open -- call POST /squad/draft first")
        )
    return committed, pending


@app.post("/squad/draft", response_model=schemas.SquadOut)
def open_edit_draft() -> schemas.SquadOut:
    committed = _require_committed_squad()
    app_state = get_app_state()
    draft = open_draft(committed, app_state.gameweek)
    set_squad_state(committed, draft)
    return _squad_out()


@app.delete("/squad/draft", response_model=schemas.SquadOut)
def discard_edit_draft() -> schemas.SquadOut:
    """D21 — "Reset team" discards only the pending draft, reverting to the last-confirmed squad."""
    committed, _ = get_squad_state()
    set_squad_state(committed, None)
    return _squad_out()


@app.post("/squad/draft/substitute", response_model=schemas.SquadOut)
def draft_substitute(body: schemas.SubstituteIn) -> schemas.SquadOut:
    committed, pending = _require_open_draft()
    app_state = get_app_state()
    new_draft = apply_substitute_to_draft(
        pending, body.out_id, body.in_id, app_state.position_by_player
    )
    set_squad_state(committed, new_draft)
    return _squad_out()


@app.post("/squad/draft/transfer", response_model=schemas.SquadOut)
def draft_transfer(body: schemas.TransferIn) -> schemas.SquadOut:
    committed, pending = _require_open_draft()
    app_state = get_app_state()
    new_draft = apply_transfer_to_draft(
        pending,
        body.out_id,
        body.in_id,
        body.in_price,
        body.in_position,
        app_state.team_id_by_player,
    )
    set_squad_state(committed, new_draft)
    return _squad_out()


def _require_live_committed():
    """Shared guard for the direct-mutation endpoints below: validates there's a committed squad
    already and no draft happens to be open."""
    app_state = get_app_state()
    committed, pending = get_squad_state()
    if committed.team_state is None:
        raise HTTPException(400, "no committed squad yet -- build and confirm one first")
    if pending is not None:
        raise HTTPException(400, "a draft is unexpectedly open -- confirm or discard it first")
    return app_state, committed


@app.post("/squad/live-transfer", response_model=schemas.SquadOut)
def live_transfer(body: schemas.TransferIn) -> schemas.SquadOut:
    """The live team page's only transfer path: swap a player straight in and out of the real
    committed squad, no draft/preview, no hit cost, no free-transfer accounting. This page is a
    planning tool for a season that hasn't been played yet, not a simulation of FPL's
    transfer-limit rules.
    """
    app_state, committed = _require_live_committed()
    new_team_state = transfer(
        committed.team_state,
        body.out_id,
        body.in_id,
        body.in_price,
        body.in_position,
        app_state.team_id_by_player,
    )
    set_squad_state(replace(committed, team_state=new_team_state), None)
    return _squad_out()


@app.post("/squad/live-captain", response_model=schemas.SquadOut)
def live_captain(body: schemas.CaptainIn) -> schemas.SquadOut:
    """The live team page's captain/vice-captain path -- same direct-mutation shape as
    :func:`live_transfer`, just simpler: captaincy never carries a hit cost, so there's nothing
    to bypass beyond the preview step itself."""
    _, committed = _require_live_committed()
    if body.role == "captain":
        new_team_state = set_captain(committed.team_state, body.player_id)
    elif body.role == "vice":
        new_team_state = set_vice_captain(committed.team_state, body.player_id)
    else:
        raise HTTPException(400, "role must be 'captain' or 'vice'")
    set_squad_state(replace(committed, team_state=new_team_state), None)
    return _squad_out()


@app.post("/squad/draft/captain", response_model=schemas.SquadOut)
def draft_captain(body: schemas.CaptainIn) -> schemas.SquadOut:
    committed, pending = _require_open_draft()
    if body.role == "captain":
        new_draft = apply_set_captain_to_draft(pending, body.player_id)
    elif body.role == "vice":
        new_draft = apply_set_vice_captain_to_draft(pending, body.player_id)
    else:
        raise HTTPException(400, "role must be 'captain' or 'vice'")
    set_squad_state(committed, new_draft)
    return _squad_out()


@app.post("/squad/draft/bench-order", response_model=schemas.SquadOut)
def draft_bench_order(body: schemas.BenchOrderIn) -> schemas.SquadOut:
    committed, pending = _require_open_draft()
    new_draft = apply_reorder_bench_to_draft(pending, body.bench_order)
    set_squad_state(committed, new_draft)
    return _squad_out()


@app.post("/squad/draft/chip", response_model=schemas.SquadOut)
def draft_chip(body: schemas.ChipIn) -> schemas.SquadOut:
    """D18: setting a chip on the draft is itself free — it is only actually spent when the draft
    is confirmed (:func:`confirm_edit_draft`)."""
    committed, pending = _require_open_draft()
    new_draft = set_draft_chip(pending, body.chip)
    set_squad_state(committed, new_draft)
    return _squad_out()


@app.post("/squad/draft/confirm", response_model=schemas.SquadOut)
def confirm_edit_draft() -> schemas.SquadOut:
    committed, pending = _require_open_draft()
    app_state = get_app_state()
    new_committed, hit_cost = confirm_draft(
        committed, pending, app_state.gameweek, app_state.deadline_passed
    )
    set_squad_state(new_committed, None)
    return _squad_out(last_hit_cost=hit_cost)


@app.post("/squad/optimise-xi", response_model=schemas.SquadOut)
def optimise_lineup() -> schemas.SquadOut:
    """D22 — applies immediately, no draft/confirm step: it only ever rearranges players already
    owned (the currently open draft's, if one exists, else the committed squad directly)."""
    committed, pending = get_squad_state()
    if committed.team_state is None:
        raise HTTPException(400, "no committed squad yet -- build and confirm one first")
    app_state = get_app_state()
    expected_points = app_state.expected_points()
    if pending is not None:
        new_draft = apply_optimise_xi_to_draft(pending, expected_points)
        set_squad_state(committed, new_draft)
    else:
        new_team_state = optimise_xi(committed.team_state, expected_points)
        set_squad_state(replace(committed, team_state=new_team_state), None)
    return _squad_out()


@app.get("/squad/points", response_model=schemas.SquadPointsOut)
def get_squad_points(
    chip: str | None = None, horizon: int = 1, source: str = "draft"
) -> schemas.SquadPointsOut:
    """The free chip-preview path (D18): pass ``chip`` to see what the total would be without
    spending anything. Scored against the open draft by default (``source="draft"``, falling back
    to the committed squad if none is open) — pass ``source="committed"`` to score the
    last-confirmed squad regardless, which is what D19's before/after rebuild comparison needs
    while a draft is open.
    """
    committed, pending = get_squad_state()
    if source == "committed":
        team_state = committed.team_state
    else:
        team_state = pending.working_state if pending is not None else committed.team_state
    if team_state is None:
        raise HTTPException(400, "no committed squad yet -- build and confirm one first")
    app_state = get_app_state()
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


def _respects_club_quota(
    team_state, candidate: TransferCandidate, team_id_by_player: dict[int, int]
) -> bool:
    """``find_transfer_candidates`` only checks position match and budget -- it has no concept of
    the max-3-per-club rule. Filtering here (rather than in ``features/transfers.py``) keeps that
    module's scope as a pure points comparator and guarantees the one recommendation this endpoint
    surfaces always applies cleanly."""
    counts: dict[int, int] = {}
    for player in team_state.squad:
        team_id = team_id_by_player.get(player.player_id)
        counts[team_id] = counts.get(team_id, 0) + 1
    counts[team_id_by_player[candidate.sell_player_id]] -= 1
    buy_team_id = team_id_by_player[candidate.buy_player_id]
    counts[buy_team_id] = counts.get(buy_team_id, 0) + 1
    return counts[buy_team_id] <= MAX_PER_CLUB


@app.get("/transfers/recommended", response_model=schemas.TransferRecommendationOut | None)
def get_recommended_transfer() -> schemas.TransferRecommendationOut | None:
    """The one headline transfer suggestion, evaluated against whichever squad is currently
    loaded (the open draft's working state if any, else the committed squad) over the full
    projection horizon, independent of this page's Next GW / Next 3 GWs toggle -- a transfer is a
    multi-gameweek commitment regardless of which window is on screen."""
    committed, pending = get_squad_state()
    team_state = pending.working_state if pending is not None else committed.team_state
    if team_state is None:
        raise HTTPException(400, "no committed squad yet -- build and confirm one first")
    app_state = get_app_state()
    plan = find_transfer_candidates(
        team_state, app_state.projections, app_state.projections, app_state.buy_prices
    )
    # affordable_candidates is already sorted by net_points_gain descending, so filtering
    # preserves that order.
    legal = [
        c
        for c in plan.affordable_candidates
        if _respects_club_quota(team_state, c, app_state.team_id_by_player)
    ]
    forced = [c for c in legal if c.is_forced]
    if forced:
        candidate = max(forced, key=lambda c: c.net_points_gain)
    else:
        candidate = legal[0] if legal and legal[0].net_points_gain > 0 else None
    if candidate is None:
        return None

    player_names = {pid: data["web_name"] for pid, data in app_state.players.items()}
    return schemas.TransferRecommendationOut(
        sell_player_id=candidate.sell_player_id,
        sell_player_name=player_names.get(candidate.sell_player_id, "?"),
        buy_player_id=candidate.buy_player_id,
        buy_player_name=player_names.get(candidate.buy_player_id, "?"),
        buy_price=candidate.buy_price,
        position=candidate.position,
        net_points_gain=candidate.net_points_gain,
        hit_cost=candidate.hit_cost,
        is_forced=candidate.is_forced,
        reasoning=candidate.reasoning,
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
