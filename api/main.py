"""FastAPI backend (Phase 4) — thin endpoints over ``features/``, no logic reimplemented. Every
mutation delegates to ``features.squad_rules``/``features.squad_draft``; a
:class:`~features.squad_rules.SquadRuleError` (or any other ``ValueError``) becomes a 400 with a
``RuleViolationOut`` body, never a 500 — these are all caller-input problems, not server faults.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import replace

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import schemas
from api.differentials_panel import DifferentialRow, build_differential_rows
from api.fixture_swing_panel import build_fixture_swing_rows
from api.fixtures_view import DEFAULT_FIXTURE_TICKER_HORIZON, build_fixture_ticker_rows
from api.mini_league_panel import (
    MiniLeaguePanel,
    build_mini_league_panel,
    get_cached_league_snapshot,
)
from api.panel import build_panel_rows, build_team_fixture_map
from api.player_stats_panel import PlayerAvailability, build_player_stats_rows
from api.squad_state import SquadState
from api.state import (
    AppState,
    get_app_settings,
    get_app_state,
    get_squad_state,
    set_app_settings,
    set_squad_state,
)
from api.transfer_panel import (
    MAX_TRANSFERS,
    LeagueContext,
    build_transfer_suggestion,
    marginal_gains,
    ownership_of,
)
from engine.data.fpl_client import FPLClient, FPLClientError
from engine.data.league_state_builder import DEFAULT_RIVAL_LIMIT
from engine.data.team_state_builder import build_my_team_state
from engine.rates import RateRatio
from features.differentials import (
    DEFAULT_WINDOW_GAMEWEEKS,
    GLOBAL_LENS,
    LEAGUE_LENS,
    OwnershipLens,
    build_differentials,
)
from features.fixture_swing import DEFAULT_FAR_GAMEWEEKS, DEFAULT_NEAR_GAMEWEEKS
from features.fixtures import HorizonDifficulty, league_average_rate
from features.mini_league import (
    PlayerOwnership,
    SwapCandidate,
    compute_exposures,
    compute_league_ownership,
    pair_with_weakest,
    prospective_swing,
)
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
    substitute,
    validate_xi,
)
from features.team_state import MyTeamState, SquadPlayer
from features.transfer_planner import TransferMove, TransferPlan

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
    gameweek_from: int | None = None,
    gameweek_to: int | None = None,
) -> list[schemas.FixtureTickerRowOut]:
    """Each bound defaults independently when omitted, chaining the same way
    ``/teams/fixture-swing``'s near/far bounds do: ``gameweek_from`` defaults to the app's
    decision gameweek, ``gameweek_to`` to
    ``gameweek_from + DEFAULT_FIXTURE_TICKER_HORIZON - 1``, so an arbitrary window like GW4-6 is
    just as valid as the locked-in 5-gameweek default."""
    app_state = get_app_state()
    if gameweek_from is None:
        gameweek_from = app_state.decision_gameweek
    if gameweek_to is None:
        gameweek_to = gameweek_from + DEFAULT_FIXTURE_TICKER_HORIZON - 1
    if gameweek_from < 1 or gameweek_to < gameweek_from:
        raise ValueError("gameweek_to must be >= gameweek_from, and both must be at least 1")
    gameweeks = list(range(gameweek_from, gameweek_to + 1))
    league_avg_xga = (
        league_average_rate(app_state.team_rates, "home_xga_per_90", "away_xga_per_90")
        if app_state.team_rates
        else None
    )
    rows = build_fixture_ticker_rows(
        app_state.fixtures,
        app_state.teams.keys(),
        gameweeks,
        team_rates=app_state.team_rates,
        league_avg_xga_per_90=league_avg_xga,
    )
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
                            expected_goals_for=entry.expected_goals_for,
                            expected_goals_against=entry.expected_goals_against,
                        )
                        for entry in cell.fixtures
                    ],
                )
                for cell in row.gameweeks
            ],
            average_difficulty=row.average_difficulty,
            total_expected_goals_for=row.total_expected_goals_for,
            total_expected_goals_against=row.total_expected_goals_against,
        )
        for row in rows
    ]


def _horizon_difficulty_out(
    difficulty: HorizonDifficulty | None,
) -> schemas.HorizonDifficultyOut | None:
    if difficulty is None:
        return None
    return schemas.HorizonDifficultyOut(
        attack_rating=difficulty.attack_rating,
        defense_rating=difficulty.defense_rating,
        mean_attack_factor=difficulty.mean_attack_factor,
        mean_defense_factor=difficulty.mean_defense_factor,
    )


@app.get("/teams/fixture-swing", response_model=schemas.FixtureSwingResponseOut)
def list_fixture_swing(
    near_from: int | None = None,
    near_to: int | None = None,
    far_from: int | None = None,
    far_to: int | None = None,
) -> schemas.FixtureSwingResponseOut:
    """Fixture swing detection plan Phase 3: per-team fixture-difficulty swing between an arbitrary
    near gameweek range and an arbitrary far one -- is this team's run getting easier or harder,
    distinct from any change in an individual player's own outlook.

    Each bound defaults independently when omitted, chaining onto whatever was resolved just
    before it so a caller can override only the window(s) they care about: ``near_from`` defaults
    to the app's decision gameweek, ``near_to`` to ``near_from + DEFAULT_NEAR_GAMEWEEKS - 1``,
    ``far_from`` to right after the (possibly customized) near window ends, and ``far_to`` to
    ``far_from + DEFAULT_FAR_GAMEWEEKS - 1`` -- reproducing the original locked-in 3-vs-5 default
    when all four are omitted.
    """
    app_state = get_app_state()
    if near_from is None:
        near_from = app_state.decision_gameweek
    if near_to is None:
        near_to = near_from + DEFAULT_NEAR_GAMEWEEKS - 1
    if far_from is None:
        far_from = near_to + 1
    if far_to is None:
        far_to = far_from + DEFAULT_FAR_GAMEWEEKS - 1
    if near_from < 1 or near_to < near_from or far_from < 1 or far_to < far_from:
        raise ValueError("each window's `to` must be >= its `from`, and both must be at least 1")
    near_gameweeks = list(range(near_from, near_to + 1))
    far_gameweeks = list(range(far_from, far_to + 1))

    squad_state = get_squad_state()
    owned_ids = {player.player_id for player in squad_state.squad}
    owned_team_ids = {app_state.team_id_by_player[pid] for pid in owned_ids}

    rows = build_fixture_swing_rows(app_state, near_gameweeks, far_gameweeks, owned_team_ids)
    return schemas.FixtureSwingResponseOut(
        near_gameweeks=near_gameweeks,
        far_gameweeks=far_gameweeks,
        rows=[
            schemas.TeamSwingRowOut(
                team_id=row.swing.team_id,
                near=_horizon_difficulty_out(row.swing.near),
                far=_horizon_difficulty_out(row.swing.far),
                attack_swing=row.swing.attack_swing,
                defense_swing=row.swing.defense_swing,
                has_owned_player=row.has_owned_player,
            )
            for row in rows
        ],
    )


@app.get("/gameweek", response_model=schemas.GameweekOut)
def get_gameweek() -> schemas.GameweekOut:
    state = get_app_state()
    decision_gameweek = state.decision_gameweek
    deadline_time = state.deadline_for(decision_gameweek) or state.deadline_time
    return schemas.GameweekOut(
        season=state.season,
        gameweek=decision_gameweek,
        projections_gameweek=state.gameweek,
        deadline_time=deadline_time.isoformat(),
        deadline_passed=state.is_deadline_passed(),
        generated_at=state.generated_at.isoformat(),
        model_version=state.model_version,
        horizon_gameweeks=state.remaining_horizon_gameweeks,
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

    Also records ``team_id`` as the app-wide ``fpl_team_id`` setting (MINI_LEAGUE_PLAN M14) --
    importing your own squad by ID is the one place this app already learns which FPL entry is
    "you", and the Mini League page needs exactly that to exclude your own entry from a league's
    effective-ownership field.
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
    settings = get_app_settings()
    set_app_settings(replace(settings, fpl_team_id=payload.team_id))
    return _squad_out()


# --- Mini League settings (MINI_LEAGUE_PLAN M14) -------------------------------------------------


@app.get("/mini-league/leagues", response_model=schemas.MiniLeagueSettingsOut)
def get_mini_league_settings() -> schemas.MiniLeagueSettingsOut:
    settings = get_app_settings()
    return schemas.MiniLeagueSettingsOut(
        fpl_team_id=settings.fpl_team_id, mini_league_ids=list(settings.mini_league_ids)
    )


@app.post("/mini-league/leagues", response_model=schemas.MiniLeagueSettingsOut)
def set_mini_league_settings(body: schemas.MiniLeagueSettingsIn) -> schemas.MiniLeagueSettingsOut:
    """Only ``fpl_team_id``/``mini_league_ids`` are settable here -- built on top of whatever
    settings already exist (via ``replace``) so this never resets ``planning_horizon_gameweeks``
    back to its default as a side effect of saving a league ID."""
    settings = replace(
        get_app_settings(),
        fpl_team_id=body.fpl_team_id,
        mini_league_ids=tuple(body.mini_league_ids),
    )
    set_app_settings(settings)
    return schemas.MiniLeagueSettingsOut(
        fpl_team_id=settings.fpl_team_id, mini_league_ids=list(settings.mini_league_ids)
    )


def _mini_league_panel_out(panel: MiniLeaguePanel) -> schemas.MiniLeaguePanelOut:
    return schemas.MiniLeaguePanelOut(
        league_id=panel.league_id,
        league_name=panel.league_name,
        picks_gameweek=panel.picks_gameweek,
        gameweek=panel.gameweek,
        my_rank=panel.my_rank,
        my_total_points=panel.my_total_points,
        coverage=panel.coverage,
        template_xi=list(panel.template_xi),
        exposures=[
            schemas.PlayerExposureOut(
                player_id=exposure.player_id,
                your_multiplier=exposure.your_multiplier,
                ownership=schemas.PlayerOwnershipOut(
                    player_id=exposure.ownership.player_id,
                    raw_ownership_percent=exposure.ownership.raw_ownership_percent,
                    eo_multiplier=exposure.ownership.eo_multiplier,
                    eo_percent=exposure.ownership.eo_percent,
                    captain_share_percent=exposure.ownership.captain_share_percent,
                    owner_names=list(exposure.ownership.owner_names),
                ),
                expected_points=exposure.expected_points,
                exposure=exposure.exposure,
                expected_swing=exposure.expected_swing,
            )
            for exposure in panel.exposures
        ],
        captain_options=[
            schemas.CaptainOptionOut(
                player_id=option.player_id,
                expected_points=option.expected_points,
                captain_share_percent=option.captain_share_percent,
                eo_multiplier=option.eo_multiplier,
                net_captain_ev=option.net_captain_ev,
                net_captain_std=option.net_captain_std,
            )
            for option in panel.captain_options
        ],
        insights=[
            schemas.LeagueInsightOut(
                kind=insight.kind,
                player_id=insight.player_id,
                reference_player_id=insight.reference_player_id,
                value=insight.value,
                owner_count=insight.owner_count,
                n_rivals=insight.n_rivals,
            )
            for insight in panel.insights
        ],
        rivals=[
            schemas.MiniLeagueRivalOut(
                entry_id=rival.entry_id,
                manager_name=rival.manager_name,
                team_name=rival.team_name,
                rank=rival.rank,
                total_points=rival.total_points,
                gameweek_points=rival.gameweek_points,
                chip_state=schemas.RivalChipStateOut(
                    entry_id=rival.chip_state.entry_id,
                    used_chip_names=list(rival.chip_state.used_chip_names),
                    remaining_chip_names=list(rival.chip_state.remaining_chip_names),
                ),
                posture=schemas.RivalPostureOut(
                    rival_entry_id=rival.posture.rival_entry_id,
                    projected_final_gap=rival.posture.projected_final_gap,
                    p_finish_ahead=rival.posture.p_finish_ahead,
                    variance_preference=rival.posture.variance_preference,
                    sensitivity=rival.posture.sensitivity,
                ),
                head_to_head=schemas.HeadToHeadOut(
                    rival_entry_id=rival.head_to_head.rival_entry_id,
                    shared_count=rival.head_to_head.shared_count,
                    differentials=[
                        schemas.DifferentialPickOut(
                            player_id=pick.player_id,
                            your_multiplier=pick.your_multiplier,
                            rival_multiplier=pick.rival_multiplier,
                            expected_points=pick.expected_points,
                            expected_gap_contribution=pick.expected_gap_contribution,
                        )
                        for pick in rival.head_to_head.differentials
                    ],
                    expected_gap=rival.head_to_head.expected_gap,
                    gap_std=rival.head_to_head.gap_std,
                    p_outscore=rival.head_to_head.p_outscore,
                ),
            )
            for rival in panel.rivals
        ],
    )


@app.get("/mini-league/{league_id}", response_model=schemas.MiniLeaguePanelOut)
def get_mini_league(
    league_id: int,
    limit: int = DEFAULT_RIVAL_LIMIT,
    refresh: bool = False,
    chip: str | None = None,
) -> schemas.MiniLeaguePanelOut:
    """The Mini League page's one bulk round trip (M17): standings, chip state, exposure, the
    captain grid, the league template, and a full head-to-head decomposition for every rival, all
    computed for the app's current gameweek against a live-fetched (and TTL-cached, M15) league
    snapshot.

    Requires a complete 15-player squad (``_require_team_state``, the same requirement every other
    squad-dependent endpoint on this page already has) and a saved ``fpl_team_id`` (M14) -- without
    knowing which entry is "you", there is no sensible "the field, excluding me" to compute.
    """
    settings = get_app_settings()
    if settings.fpl_team_id is None:
        raise ValueError(
            "no FPL team ID is configured -- import your squad via /squad/import, or save one "
            "directly via /mini-league/leagues, before requesting mini-league data"
        )
    app_state = get_app_state()
    _, team_state = _require_team_state()

    with FPLClient() as client:
        try:
            snapshot = get_cached_league_snapshot(client, league_id, limit=limit, refresh=refresh)
        except FPLClientError as exc:
            raise ValueError(f"could not fetch mini-league {league_id}: {exc}") from exc

    panel = build_mini_league_panel(
        app_state, team_state, snapshot, settings.fpl_team_id, chip=chip
    )
    return _mini_league_panel_out(panel)


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


@app.post("/squad/substitute", response_model=schemas.SquadOut)
def substitute_squad_player(body: schemas.SubstituteIn) -> schemas.SquadOut:
    """Swap one starting-XI player for one bench player."""
    state, team_state = _require_team_state()
    position_by_player = {player.player_id: player.position for player in state.squad}
    new_team_state = substitute(team_state, body.out_id, body.in_id, position_by_player)
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
    differently). ``objective="full_squad"`` values every one of the 15 picks, not just the XI, so
    the bench holds real depth rather than minimum-cost fodder; the web app always sends this for
    Auto Build. The starting XI itself is unaffected either way — it always comes out of
    :func:`~features.formation.select_starting_xi`'s own highest-EV-of-the-15 pick afterward, so
    only the 11 starters' own points ever decide who starts.
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


# --- Transfer banner (TRANSFER_BANNER) -----------------------------------------------------------


def _transfer_move_out(
    move: TransferMove,
    app_state: AppState,
    gameweek_points: Mapping[int, float],
    league: LeagueContext,
) -> schemas.TransferMoveOut:
    names = app_state.players
    return schemas.TransferMoveOut(
        out_player_id=move.out_player_id,
        in_player_id=move.in_player_id,
        out_name=names.get(move.out_player_id, {}).get("web_name", f"#{move.out_player_id}"),
        in_name=names.get(move.in_player_id, {}).get("web_name", f"#{move.in_player_id}"),
        position=move.position,
        price_delta=move.price_delta,
        out_expected_points=gameweek_points.get(move.out_player_id),
        in_expected_points=gameweek_points.get(move.in_player_id),
        in_eo_multiplier=ownership_of(move.in_player_id, league),
    )


def _transfer_plan_out(
    plan: TransferPlan,
    app_state: AppState,
    gameweek_points: Mapping[int, float],
    league: LeagueContext,
) -> schemas.TransferPlanOut:
    return schemas.TransferPlanOut(
        moves=[_transfer_move_out(move, app_state, gameweek_points, league) for move in plan.moves],
        out_player_ids=list(plan.out_player_ids),
        in_player_ids=list(plan.in_player_ids),
        n_transfers=plan.n_transfers,
        expected_points=plan.expected_points,
        expected_points_delta=plan.expected_points_delta,
        expected_gap=plan.expected_gap,
        expected_gap_delta=plan.expected_gap_delta,
        gap_std=plan.gap_std,
        gap_std_delta=plan.gap_std_delta,
        expected_final_rank=plan.expected_final_rank,
        expected_final_rank_delta=plan.expected_final_rank_delta,
        spend_delta=plan.spend_delta,
        budget_remaining=plan.budget_remaining,
    )


@app.get("/squad/transfers", response_model=schemas.TransferSuggestionOut)
def suggest_transfers(
    transfers: int = 1, horizon: int = 1, chip: str | None = None, league_id: int | None = None
) -> schemas.TransferSuggestionOut:
    """The Team page banner's suggestion: which players to sell and buy, ranked by projected
    finishing position in your mini-league rather than by expected points alone (see
    ``features.transfer_planner``'s module docstring for why those two are the same ranking once
    expectation is all you look at, and what variance adds).

    ``horizon`` sets how many gameweeks the expected points gain is summed over, matching
    ``/squad/points``' own argument, while the league math is always measured at the decision
    gameweek alone, since rival picks exist for exactly one gameweek at a time.

    Needs a complete 15-player squad, like every other squad-dependent endpoint here. A league is
    optional: without one the suggestion is still returned, ranked on expected points, with
    ``league_id`` null and ``n_rivals`` zero so the banner can say which it is.
    """
    state, team_state = _require_team_state()
    app_state = get_app_state()
    gameweeks = app_state.remaining_horizon_gameweeks[: max(horizon, 1)]

    suggestion, league = build_transfer_suggestion(
        app_state,
        team_state,
        get_app_settings(),
        budget=state.budget_ceiling,
        max_transfers=transfers,
        gameweeks=gameweeks,
        chip=chip,
        league_id=league_id,
    )

    gameweek_points = app_state.expected_points(suggestion.league_gameweek)
    return schemas.TransferSuggestionOut(
        plans=[
            _transfer_plan_out(plan, app_state, gameweek_points, league)
            for plan in suggestion.plans
        ],
        best_by_transfer_count=[
            _transfer_plan_out(plan, app_state, gameweek_points, league)
            for plan in suggestion.best_by_transfer_count
        ],
        marginal_points_gains=marginal_gains(suggestion),
        max_transfers=suggestion.max_transfers,
        max_transfers_allowed=MAX_TRANSFERS,
        current_expected_points=suggestion.current_expected_points,
        current_expected_gap=suggestion.current_expected_gap,
        current_gap_std=suggestion.current_gap_std,
        current_expected_final_rank=suggestion.current_expected_final_rank,
        variance_preference=suggestion.variance_preference,
        n_rivals=suggestion.n_rivals,
        league_id=league.league_id,
        league_name=league.league_name,
        picks_gameweek=league.picks_gameweek,
        gameweeks=list(suggestion.gameweeks),
        league_gameweek=suggestion.league_gameweek,
    )


@app.post("/squad/transfers/apply", response_model=schemas.SquadOut)
def apply_transfers(body: schemas.ApplyTransfersIn) -> schemas.SquadOut:
    """Apply a suggested plan in one call: drop every ``out_player_ids`` player, add every
    ``in_player_ids`` player at their current price, and re-derive the XI.

    The whole final 15 is validated once (``features.squad_rules.assemble_team_state``) rather than
    each swap being applied and checked in turn. Applying one at a time can fail on a transient
    illegality the finished squad does not have (buying before selling breaches the budget; two
    players from one club overlapping for a step breaches the club limit), and rejecting a legal
    destination because of the route taken to it would be wrong.

    The captain and vice are kept if they are still in the squad and still start, so a transfer
    elsewhere never casually moves the armband, matching ``_reconcile_after_squad_change``'s own
    behaviour on an add or remove.
    """
    state, team_state = _require_team_state()
    app_state = get_app_state()

    if len(body.out_player_ids) != len(body.in_player_ids):
        raise ValueError("out_player_ids and in_player_ids must be the same length")
    outgoing = set(body.out_player_ids)
    missing = outgoing - set(team_state.player_ids)
    if missing:
        raise SquadRuleError(
            RuleViolation(
                "unknown_player",
                f"player(s) {sorted(missing)} are not in the squad",
                tuple(sorted(missing)),
            )
        )
    already_owned = set(body.in_player_ids) & set(team_state.player_ids)
    if already_owned:
        raise SquadRuleError(
            RuleViolation(
                "duplicate",
                f"player(s) {sorted(already_owned)} are already in the squad",
                tuple(sorted(already_owned)),
            )
        )
    unpriced = [pid for pid in body.in_player_ids if pid not in app_state.buy_prices]
    if unpriced:
        raise ValueError(f"no current price is known for player(s) {sorted(unpriced)}")

    new_squad = tuple(
        player for player in team_state.squad if player.player_id not in outgoing
    ) + tuple(
        SquadPlayer(pid, app_state.position_by_player[pid], app_state.buy_prices[pid])
        for pid in body.in_player_ids
    )
    new_team_state = assemble_team_state(
        new_squad,
        app_state.expected_points(),
        app_state.team_id_by_player,
        budget=state.budget_ceiling,
        preferred_captain_id=state.captain_id,
        preferred_vice_captain_id=state.vice_captain_id,
    )
    _save_team_state(state, new_team_state)
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
        gameweeks = app_state.remaining_horizon_gameweeks[: max(horizon, 1)]
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


def _resolve_player_stats_ownership() -> tuple[str, Mapping[int, float]]:
    """Which mini-league ownership the Player Stats page can show, and why not when it cannot.

    Deliberately *not* :func:`_resolve_differentials_ownership`, whose whole contract is to fall
    back to the FPL-wide lens and never fail. This page's Own% column means "how many of my
    rivals own this" specifically, so silently substituting a population-wide percentage under
    the same heading would answer a different question than the one asked. Every status except
    ``"ok"`` returns an empty map, and the caller reports the reason to the UI.
    """
    settings = get_app_settings()
    if not settings.mini_league_ids or settings.fpl_team_id is None:
        return "not_configured", {}

    try:
        with FPLClient() as client:
            snapshot = get_cached_league_snapshot(client, settings.mini_league_ids[0])
    except FPLClientError:
        return "fetch_failed", {}

    ownership_by_player = compute_league_ownership(snapshot, exclude_entry_id=settings.fpl_team_id)
    n_rivals = sum(1 for entry in snapshot.entries if entry.entry_id != settings.fpl_team_id)
    if n_rivals == 0:
        return "no_rivals", {}

    # raw_ownership_percent, the plain "percent of rivals who own this", not the captaincy-weighted
    # eo_percent Differentials uses for haul exposure -- this column is a headcount question.
    return "ok", {pid: o.raw_ownership_percent for pid, o in ownership_by_player.items()}


def _rate_ratio_out(ratio: RateRatio | None) -> schemas.RateRatioOut | None:
    if ratio is None:
        return None
    return schemas.RateRatioOut(
        ratio=ratio.ratio, low=ratio.low, high=ratio.high, exposure=ratio.exposure
    )


@app.get("/players/stats", response_model=schemas.PlayerStatsResponseOut)
def list_player_stats(gameweek_from: int, gameweek_to: int) -> schemas.PlayerStatsResponseOut:
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
    availability_by_player = {
        pid: PlayerAvailability(
            status=data.get("status", "a"),
            chance_of_playing_next_round=data.get("chance_of_playing_next_round", 100.0),
            news=data.get("news"),
        )
        for pid, data in app_state.players.items()
    }
    ownership_status, ownership_by_player = _resolve_player_stats_ownership()
    penalty_takers = frozenset(
        pid for pid, data in app_state.players.items() if data.get("penalties_order") == 1
    )
    fixture_map = build_team_fixture_map(app_state.fixtures)

    actual_stats_by_player = build_actual_stats_by_player(
        app_state.player_history,
        app_state.position_by_player,
        gameweek_from,
        gameweek_to,
        ownership_by_player,
        # Priors are fitted over the whole season, never the selected range: k is a population
        # parameter, and a one-gameweek view would estimate it least reliably exactly when the
        # shrinkage it drives matters most.
        full_season_history=app_state.player_history,
        penalty_takers=penalty_takers,
    )
    rows = build_player_stats_rows(
        actual_stats_by_player,
        app_state.projections,
        player_names,
        app_state.buy_prices,
        app_state.team_id_by_player,
        app_state.position_by_player,
        low_confidence_ids,
        availability_by_player,
        fixture_map,
        app_state.horizon_gameweeks,
    )
    return schemas.PlayerStatsResponseOut(
        ownership_status=ownership_status,
        rows=[
            schemas.PlayerStatsRowOut(
                player_id=row.player_id,
                name=row.name,
                team_id=row.team_id,
                position=row.position,
                price=row.price,
                low_confidence=row.low_confidence,
                availability=(
                    schemas.AvailabilityOut(
                        status=row.availability.status,
                        chance_of_playing_next_round=row.availability.chance_of_playing_next_round,
                        news=row.availability.news,
                    )
                    if row.availability is not None
                    else None
                ),
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
                    defensive_contribution=row.actuals.defensive_contribution,
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
                        defensive_contribution=(
                            row.actuals.points_breakdown.defensive_contribution
                        ),
                        saves=row.actuals.points_breakdown.saves,
                        bonus=row.actuals.points_breakdown.bonus,
                        cards=row.actuals.points_breakdown.cards,
                        penalty_misses=row.actuals.points_breakdown.penalty_misses,
                        own_goals=row.actuals.points_breakdown.own_goals,
                        total=row.actuals.points_breakdown.total,
                    ),
                    ownership_percent=row.actuals.ownership_percent,
                    small_sample=row.actuals.small_sample,
                    attacking_ratio=_rate_ratio_out(row.actuals.attacking_ratio),
                    defensive_ratio=_rate_ratio_out(row.actuals.defensive_ratio),
                    is_penalty_taker=row.actuals.is_penalty_taker,
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
        ],
    )


def _global_ownership_lens() -> OwnershipLens:
    app_state = get_app_state()
    percent = {pid: data.get("selected_by_percent") for pid, data in app_state.players.items()}
    return OwnershipLens(
        source=GLOBAL_LENS,
        n_rivals=None,
        percent=percent,
        owner_count={},
        eo_multiplier={},
        owner_names={},
    )


def _resolve_differentials_ownership(
    league_id: int | None,
) -> tuple[OwnershipLens, int | None, Mapping[int, PlayerOwnership]]:
    """Resolve which ownership lens the Differentials page uses (MINI_LEAGUE_PLAN M24): the
    requested (or first tracked) league's effective ownership when a league is configured, its
    live fetch succeeds, and it actually has rivals to measure against; the FPL-wide percentage
    otherwise. Falls back silently to the global lens rather than raising -- Differentials has
    never had a live-fetch dependency before, and a manager who hasn't set up a mini-league (or
    whose league fetch hits a transient FPL problem) should still get a fully working page, just
    on the lens that's always been available. The caller reports which lens actually won via the
    response's own ``ownership_lens`` field, never left for the frontend to guess. Returns
    ``(lens, picks_gameweek, ownership_by_player)`` -- ``picks_gameweek`` is ``None`` under the
    global lens, where the concept (M1) doesn't apply; ``ownership_by_player`` is the raw
    per-player :class:`~features.mini_league.PlayerOwnership` map the league lens was built from
    (empty under the global lens), which the "Replaces" swap suggestion needs and the flattened
    ``OwnershipLens`` dicts don't carry.
    """
    settings = get_app_settings()
    resolved_league_id = (
        league_id
        if league_id is not None
        else (settings.mini_league_ids[0] if settings.mini_league_ids else None)
    )
    if resolved_league_id is None or settings.fpl_team_id is None:
        return _global_ownership_lens(), None, {}

    try:
        with FPLClient() as client:
            snapshot = get_cached_league_snapshot(client, resolved_league_id)
    except FPLClientError:
        return _global_ownership_lens(), None, {}

    ownership_by_player = compute_league_ownership(snapshot, exclude_entry_id=settings.fpl_team_id)
    n_rivals = sum(1 for entry in snapshot.entries if entry.entry_id != settings.fpl_team_id)
    if n_rivals == 0:
        return _global_ownership_lens(), None, {}

    lens = OwnershipLens(
        source=LEAGUE_LENS,
        n_rivals=n_rivals,
        percent={pid: o.eo_percent for pid, o in ownership_by_player.items()},
        owner_count={pid: o.owner_count for pid, o in ownership_by_player.items()},
        eo_multiplier={pid: o.eo_multiplier for pid, o in ownership_by_player.items()},
        owner_names={pid: o.owner_names for pid, o in ownership_by_player.items()},
    )
    return lens, snapshot.picks_gameweek, ownership_by_player


def _prospective_swing(row: DifferentialRow, current_gameweek: int) -> float | None:
    if row.differential.league_eo_multiplier is None:
        return None
    cell = next((c for c in row.fixtures if c.gameweek == current_gameweek), None)
    if cell is None or cell.expected_points is None:
        return None
    return prospective_swing(row.differential.league_eo_multiplier, cell.expected_points)


def _differentials_replacements(
    app_state: AppState,
    ownership_by_player: Mapping[int, PlayerOwnership],
    swing_by_player: Mapping[int, float],
    current_gameweek: int,
) -> dict[int, SwapCandidate]:
    """Pairs each differential with the weakest same-position starting-XI player by expected
    swing, for the league lens's "Replaces" column. Returns an empty dict, never raises, when
    there is no complete 15-player squad to compare against -- Differentials has always worked
    without a squad and must keep doing so, unlike the Mini League page's own
    ``_require_team_state`` which is allowed to demand one.
    """
    squad_state = get_squad_state()
    if (
        len(squad_state.squad) != SQUAD_SIZE
        or squad_state.captain_id is None
        or squad_state.vice_captain_id is None
    ):
        return {}

    team_state = MyTeamState(
        squad=squad_state.squad,
        starting_xi=squad_state.starting_xi,
        bench_order=squad_state.bench_order,
        captain_id=squad_state.captain_id,
        vice_captain_id=squad_state.vice_captain_id,
        mini_league_ids=squad_state.mini_league_ids,
    )
    projections = {
        player_id: horizon.gameweeks[current_gameweek]
        for player_id, horizon in app_state.projections.items()
        if current_gameweek in horizon.gameweeks
    }
    starting_xi_exposures = compute_exposures(
        team_state.starting_xi, team_state, ownership_by_player, projections
    )
    return pair_with_weakest(
        swing_by_player.keys(),
        starting_xi_exposures,
        app_state.position_by_player,
        app_state.buy_prices,
        swing_by_player,
    )


@app.get("/players/differentials", response_model=schemas.DifferentialsResponseOut)
def list_differentials(
    window: int = DEFAULT_WINDOW_GAMEWEEKS,
    max_ownership: float | None = None,
    max_league_owners: int | None = None,
    league_id: int | None = None,
    hide_owned: bool = True,
) -> schemas.DifferentialsResponseOut:
    """DIFFERENTIALS_PLAN Phase 3: players sustainedly outperforming their own position/price
    bracket among low-owned players, ranked (client-side, D10) over a set of independent,
    unblended columns rather than one composite score.

    ``window`` is the requested gameweek count, clamped to whatever has actually been played this
    season (D5/D6) -- the response's own ``window`` field reports the resolved range, since it
    frequently differs from what was requested early in a season. ``hide_owned`` excludes a
    player already in the committed squad by default, rather than showing them as a "differential"
    against yourself.

    Ownership is lens-dependent (MINI_LEAGUE_PLAN M24, supersedes D1's original global-only
    framing): ``max_ownership`` filters against the FPL-wide percentage under the global lens,
    ``max_league_owners`` against a plain rival count under the league lens -- see
    :func:`_resolve_differentials_ownership` for the fallback chain between them, and
    ``features.differentials``'s own module docstring for why a percentage ceiling stops making
    sense once "the league" means 11 rivals rather than the whole FPL player base.

    Registered *before* ``/players/{player_id}`` below, for the same route-order reason
    ``/players/stats`` above already documents -- a literal path registered after the
    parameterised one gets swallowed by it.
    """
    app_state = get_app_state()
    player_names = {pid: data["web_name"] for pid, data in app_state.players.items()}

    ownership_lens, picks_gameweek, ownership_by_player = _resolve_differentials_ownership(
        league_id
    )

    resolved_window, differentials = build_differentials(
        app_state.player_history,
        app_state.position_by_player,
        app_state.buy_prices,
        # The cache's own gameweek, not `decision_gameweek`: this reads actuals out of
        # `player_history`, which was captured with the cache and so knows nothing about any
        # gameweek played since.
        latest_played_gameweek=app_state.gameweek - 1,
        window_gameweeks=window,
        ownership=ownership_lens,
        max_ownership_percent=max_ownership,
        max_league_owners=max_league_owners,
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

    swing_by_player = {
        row.differential.player_id: swing
        for row in rows
        if (swing := _prospective_swing(row, app_state.gameweek)) is not None
    }
    replacements = (
        _differentials_replacements(
            app_state, ownership_by_player, swing_by_player, app_state.gameweek
        )
        if ownership_lens.source == LEAGUE_LENS
        else {}
    )

    return schemas.DifferentialsResponseOut(
        window=schemas.DifferentialsWindowOut(
            gameweek_from=resolved_window.gameweek_from,
            gameweek_to=resolved_window.gameweek_to,
            requested_gameweeks=resolved_window.requested_gameweeks,
        ),
        ownership_lens=ownership_lens.source,
        picks_gameweek=picks_gameweek,
        n_rivals=ownership_lens.n_rivals,
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
                league_owner_count=row.differential.league_owner_count,
                league_eo_multiplier=row.differential.league_eo_multiplier,
                league_owner_names=list(row.differential.league_owner_names),
                expected_swing=swing_by_player.get(row.differential.player_id),
                replaces=(
                    schemas.SwapCandidateOut(
                        incoming_player_id=swap.incoming_player_id,
                        outgoing_player_id=swap.outgoing_player_id,
                        incoming_swing=swap.incoming_swing,
                        outgoing_swing=swap.outgoing_swing,
                        net_swing_delta=swap.net_swing_delta,
                        price_delta=swap.price_delta,
                    )
                    if (swap := replacements.get(row.differential.player_id)) is not None
                    else None
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
