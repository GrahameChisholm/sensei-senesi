"""The season loop (``planning/SEASON_SIMULATOR.md`` components 5-6): replay a real historical
season gameweek by gameweek, deciding chips, transfers, formation, and captaincy purely from
pre-deadline horizon projections, then reveal that gameweek's real recorded ``total_points`` and
score it — captain doubling (or tripling under Triple Captain, with the standard FPL vice-captain
fallback when the captain records 0 minutes), Bench Boost's bench-scores-too effect, and FPL's
real autosub rule all applied. Runs a second "hold the GW1 squad, never transfer or use a chip,
captain the highest-EV eligible player" baseline through the identical loop alongside it, so the
final report shows the delta actually attributable to the engine's decisions, not just to holding
a reasonable squad (``planning/SEASON_SIMULATOR.md``'s own reporting requirement).

**Why this doesn't replay one specific real manager's actual 2025/26 season.** See
``simulator/initial_squad.py``'s docstring: FPL's entry endpoints only serve the *current*
season's picks, so no specific real manager's actual historical GW1 squad/transfers/chips are
retrievable any more (confirmed live, 2026-07-31, well after the 2025/26 season had rolled over to
2026/27). The GW1 squad here is a real-priced, real-constrained *template* squad instead — the
same hindsight-informed device ``backtest/run_season.py``'s own ``build_stand_in_squad_starting_
xi`` already uses for captaincy backtesting, for the same underlying reason. If ``entry_id`` and
``fpl_client`` are supplied, that specific manager's real *lifetime season total* (still
retrievable — only the per-gameweek picks are gone) is attached to the report purely as an
external, differently-seeded benchmark, clearly not an apples-to-apples decision replay.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

from backtest.run_season import DEFAULT_CACHE_DIR, fit_fn, season_label
from engine.data.fpl_client import FPLClient
from features.captaincy import rank_captaincy_pool
from features.chips import (
    FREE_HIT_BLOCKED_GAMEWEEKS,
    evaluate_bench_boost,
    evaluate_free_hit,
    evaluate_triple_captain,
    evaluate_wildcard,
)
from features.fixtures import TeamFixture
from features.team_state import MyTeamState, SquadPlayer
from features.transfers import find_transfer_candidates
from simulator.chip_calendar import ChipUsage, available_chips_this_gameweek, record_chip_played
from simulator.data_prep import prepare_season_data
from simulator.formation import apply_autosubs, select_starting_xi
from simulator.horizon import build_horizon_predictions, build_horizon_projections
from simulator.initial_squad import DEFAULT_BUDGET, build_squad

__all__ = [
    "DEFAULT_HORIZON_LENGTH",
    "DEFAULT_MIN_TRAINING_GAMEWEEKS",
    "SimulatorSquad",
    "GameweekRecord",
    "SimulationReport",
    "run_season_simulation",
]

DEFAULT_HORIZON_LENGTH = 5
DEFAULT_MIN_TRAINING_GAMEWEEKS = 3
_CAPTAIN_MULTIPLIER = 2
_TRIPLE_CAPTAIN_MULTIPLIER = 3


@dataclass(frozen=True)
class SimulatorSquad:
    """The simulator's own evolving squad bookkeeping between gameweeks — deliberately separate
    from ``features.team_state.MyTeamState`` (frozen, and missing the "pre-Free-Hit squad to
    restore next week" concept this loop needs). ``to_my_team_state`` builds the shared object
    fresh each week for ``features.captaincy``/``transfers``/``chips`` to read."""

    squad: tuple[SquadPlayer, ...]
    bank: int
    free_transfers: int
    chip_usage: ChipUsage
    free_hit_reserve: tuple[SquadPlayer, ...] | None = None

    @property
    def total_budget(self) -> int:
        """Bank plus every squad player's sell price — the full amount available for a Wildcard
        or Free Hit's complete squad rebuild."""
        return self.bank + sum(p.sell_price for p in self.squad)


@dataclass(frozen=True)
class GameweekRecord:
    """One gameweek's decisions and outcome, for the per-gameweek decision log (``planning/
    SEASON_SIMULATOR.md`` component 6)."""

    gameweek: int
    chip_played: str | None
    transfer_in: int | None
    transfer_out: int | None
    hit_cost: int
    captain_id: int
    vice_captain_id: int
    starting_xi: tuple[int, ...]
    points_scored: float
    baseline_points_scored: float
    running_total: float
    baseline_running_total: float


@dataclass(frozen=True)
class SimulationReport:
    season_start_year: int
    engine_total_points: float
    baseline_total_points: float
    gameweek_log: tuple[GameweekRecord, ...]
    skipped_gameweeks: tuple[int, ...]
    real_recorded_total: float | None = None


def _to_my_team_state(
    simulator_squad: SimulatorSquad,
    starting_xi: tuple[int, ...],
    bench_order: tuple[int, ...],
    captain_id: int,
    vice_captain_id: int,
    gameweek: int,
) -> MyTeamState:
    return MyTeamState(
        squad=simulator_squad.squad,
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        bank=simulator_squad.bank,
        free_transfers=simulator_squad.free_transfers,
        chips_remaining=available_chips_this_gameweek(simulator_squad.chip_usage, gameweek),
    )


def _select_captain_and_vice(my_team: MyTeamState, gw_pool: list) -> tuple[int, int]:
    """Top-two eligible EV picks via ``features.captaincy.rank_captaincy_pool``; falls back to
    the first two starting-XI ids if fewer than two eligible players have a projection this
    gameweek (always structurally possible, since ``starting_xi`` is always 11 long)."""
    eligible_ranked: list[int] = []
    if gw_pool:
        recommendation = rank_captaincy_pool(my_team, gw_pool)
        eligible_ranked = [
            option.player_id for option in recommendation.ranked_pool if option.is_eligible
        ]
    ordered = eligible_ranked + [pid for pid in my_team.starting_xi if pid not in eligible_ranked]
    return ordered[0], ordered[1]


def _rebuild_squad_via_chip(
    gw_frame: pd.DataFrame,
    value_by_player: dict[int, float],
    budget: int,
) -> tuple[SquadPlayer, ...]:
    pool = gw_frame[["player_id", "position", "price", "team"]].copy()
    pool["player_id"] = pool["player_id"].astype(int)
    pool["price"] = pool["price"].round().astype(int)
    pool["value_score"] = pool["player_id"].map(value_by_player).fillna(0.0)
    return build_squad(pool, budget=budget, value_col="value_score")


def _effective_captain(
    captain_id: int, vice_captain_id: int, minutes_by_player: dict[int, int]
) -> int:
    """Real FPL rule: the vice-captain gets the multiplier instead if the captain recorded 0
    minutes (this simulator goes no further than that one level of fallback, matching every other
    decision policy's documented v1 depth)."""
    if minutes_by_player.get(captain_id, 0) > 0:
        return captain_id
    return vice_captain_id


def _score_squad(
    xi_after_subs: tuple[int, ...],
    bench_order: tuple[int, ...],
    captain_id: int,
    vice_captain_id: int,
    chip_played: str | None,
    points_by_player: dict[int, float],
    minutes_by_player: dict[int, int],
) -> float:
    score = sum(points_by_player.get(pid, 0.0) for pid in xi_after_subs)
    effective_captain = _effective_captain(captain_id, vice_captain_id, minutes_by_player)
    multiplier = (
        _TRIPLE_CAPTAIN_MULTIPLIER if chip_played == "triple_captain" else _CAPTAIN_MULTIPLIER
    )
    score += (multiplier - 1) * points_by_player.get(effective_captain, 0.0)
    if chip_played == "bench_boost":
        score += sum(points_by_player.get(pid, 0.0) for pid in bench_order)
    return score


def _decide_chip(
    my_team: MyTeamState,
    gameweek: int,
    horizon_gameweeks: list[int],
    horizon_projection_map: dict,
    fixtures: list[TeamFixture],
    team_id_by_player: dict[int, int],
    current_projections: dict,
    candidate_pool: dict,
    buy_prices: dict[int, int],
) -> str | None:
    """Simple, single-chip-per-week trigger (``planning/SEASON_SIMULATOR.md``'s own "simple chip
    triggers" v1 scope): plays whichever available chip's evaluator says ``play_now`` for *this*
    gameweek, preferring Bench Boost/Triple Captain/Free Hit (all gameweek-specific) over Wildcard
    (a standing drift signal, so it's checked last -- there's no "wrong week" to play it in)."""
    available = my_team.chips_remaining
    if "bench_boost" in available:
        verdict = evaluate_bench_boost(my_team, horizon_projection_map, gameweek)
        if verdict.recommendation == "play_now":
            return "bench_boost"
    if "triple_captain" in available:
        verdict = evaluate_triple_captain(my_team, horizon_projection_map, gameweek)
        if verdict.recommendation == "play_now":
            return "triple_captain"
    if "free_hit" in available and gameweek not in FREE_HIT_BLOCKED_GAMEWEEKS:
        verdict = evaluate_free_hit(
            my_team, team_id_by_player, fixtures, horizon_gameweeks, gameweek
        )
        if verdict.recommendation == "play_now":
            return "free_hit"
    if "wildcard" in available:
        verdict = evaluate_wildcard(my_team, current_projections, candidate_pool, buy_prices)
        if verdict.recommendation == "play_now":
            return "wildcard"
    return None


def run_season_simulation(
    season_start_year: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    refresh: bool = False,
    horizon_length: int = DEFAULT_HORIZON_LENGTH,
    min_training_gameweeks: int = DEFAULT_MIN_TRAINING_GAMEWEEKS,
    n_simulation_runs: int = 200,
    seed: int | None = None,
    initial_budget: int = DEFAULT_BUDGET,
    entry_id_for_benchmark: int | None = None,
    fpl_client: FPLClient | None = None,
    engineered: pd.DataFrame | None = None,
) -> SimulationReport:
    """Replay ``season_start_year`` end to end. ``engineered`` can be supplied directly (tests use
    small synthetic frames); the default fetches and engineers the real season via
    ``simulator.data_prep.prepare_season_data``.
    """
    if engineered is None:
        engineered = prepare_season_data(season_start_year, cache_dir, refresh)

    all_gameweeks = sorted(engineered["gameweek"].unique())
    season_totals = engineered.groupby("player_id")["total_points"].sum()

    # Seed the squad from the earliest gameweek whose player pool is actually rich enough to build
    # a legal 15 from -- real 2025/26 data found the true GW1 row count can be a near-empty
    # handful (most players' crosswalk-matched Understat history is thinnest right at a season's
    # own first gameweek, even with multi-season carry-forward), which alone can't fill every
    # position/club-constrained slot. Falling forward to the next gameweek with enough rows is a
    # generic fix for that whole class of early-season thinness, not a hardcoded "skip GW1" rule.
    initial_squad: tuple[SquadPlayer, ...] | None = None
    gw1 = all_gameweeks[0]
    for candidate_gw in all_gameweeks:
        candidate_frame = engineered[engineered["gameweek"] == candidate_gw].copy()
        candidate_frame["player_id"] = candidate_frame["player_id"].astype(int)
        candidate_frame["price"] = candidate_frame["price"].round().astype(int)
        candidate_frame["season_total_points"] = (
            candidate_frame["player_id"].map(season_totals).fillna(0.0)
        )
        try:
            initial_squad = build_squad(
                candidate_frame, budget=initial_budget, value_col="season_total_points"
            )
        except ValueError:
            continue
        gw1 = candidate_gw
        break
    if initial_squad is None:
        raise ValueError(
            "could not build a legal initial squad from any gameweek's player pool this season"
        )

    state = SimulatorSquad(
        squad=initial_squad,
        bank=initial_budget - sum(p.purchase_price for p in initial_squad),
        free_transfers=1,
        chip_usage=ChipUsage(),
    )
    baseline_squad = initial_squad  # never mutated -- the "hold GW1, never act" comparison

    fixture_rows = engineered.drop_duplicates(subset=["team_id", "gameweek", "opponent_team"])
    fixtures = [
        TeamFixture(
            team_id=int(row.team_id),
            opponent_id=int(row.opponent_team),
            gameweek=int(row.gameweek),
            is_home=bool(row.was_home),
        )
        for row in fixture_rows.itertuples()
    ]

    decision_gameweeks = [gw for gw in all_gameweeks if gw >= gw1]
    skipped: list[int] = []
    log: list[GameweekRecord] = []
    engine_total = 0.0
    baseline_total = 0.0

    for gameweek in decision_gameweeks:
        training_history = engineered[engineered["gameweek"] < gameweek]
        if training_history["gameweek"].nunique() < min_training_gameweeks:
            skipped.append(gameweek)
            continue

        # A Free Hit's squad is temporary by design -- revert to the real squad first thing, so
        # every decision this gameweek (including a fresh chip check) sees the real squad, never
        # last week's one-off Free Hit lineup.
        if state.free_hit_reserve is not None:
            state = replace(state, squad=state.free_hit_reserve, free_hit_reserve=None)

        fitted_state = fit_fn(training_history)
        horizon_gameweeks = [
            gw for gw in range(gameweek, gameweek + horizon_length) if gw in all_gameweeks
        ]
        horizon_predictions = build_horizon_predictions(
            engineered, fitted_state, horizon_gameweeks, n_simulation_runs, seed
        )
        horizon_projection_map = build_horizon_projections(horizon_predictions)

        gw_frame = engineered[engineered["gameweek"] == gameweek]
        team_id_by_player = {int(row.player_id): int(row.team_id) for row in gw_frame.itertuples()}
        buy_prices = {int(row.player_id): int(round(row.price)) for row in gw_frame.itertuples()}
        gw_points = {
            pid: proj.per_gameweek_points.get(gameweek, 0.0)
            for pid, proj in horizon_projection_map.items()
        }
        gw_pool = [
            proj.gameweeks[gameweek]
            for proj in horizon_projection_map.values()
            if gameweek in proj.gameweeks
        ]

        # --- chip decision, on the pre-transfer squad ---
        provisional_xi, provisional_bench = select_starting_xi(state.squad, gw_points)
        provisional_my_team = _to_my_team_state(
            state,
            provisional_xi,
            provisional_bench,
            provisional_xi[0],
            provisional_xi[1],
            gameweek,
        )
        owned_ids = {p.player_id for p in state.squad}
        current_projections = {
            pid: proj for pid, proj in horizon_projection_map.items() if pid in owned_ids
        }
        candidate_pool = {
            pid: proj for pid, proj in horizon_projection_map.items() if pid not in owned_ids
        }

        chip_played = _decide_chip(
            provisional_my_team,
            gameweek,
            horizon_gameweeks,
            horizon_projection_map,
            fixtures,
            team_id_by_player,
            current_projections,
            candidate_pool,
            buy_prices,
        )

        transfer_in: int | None = None
        transfer_out: int | None = None
        hit_cost = 0

        if chip_played == "wildcard":
            horizon_totals = {
                pid: proj.horizon_total_points for pid, proj in horizon_projection_map.items()
            }
            budget = state.total_budget
            try:
                new_squad = _rebuild_squad_via_chip(gw_frame, horizon_totals, budget)
            except ValueError:
                # The greedy rebuild heuristic couldn't find a legal 15 for this budget/pool this
                # week (a real limit of a v1 greedy squad-builder, not a real-world impossibility)
                # -- decline the chip this week rather than crash the whole replay.
                chip_played = None
            else:
                state = replace(
                    state,
                    squad=new_squad,
                    bank=budget - sum(p.purchase_price for p in new_squad),
                    chip_usage=record_chip_played(state.chip_usage, "wildcard"),
                )
        elif chip_played == "free_hit":
            budget = state.total_budget
            try:
                temp_squad = _rebuild_squad_via_chip(gw_frame, gw_points, budget)
            except ValueError:
                chip_played = None
            else:
                state = replace(
                    state,
                    squad=temp_squad,
                    bank=budget - sum(p.purchase_price for p in temp_squad),
                    free_hit_reserve=state.squad,
                    chip_usage=record_chip_played(state.chip_usage, "free_hit"),
                )

        if chip_played not in ("wildcard", "free_hit"):
            plan = find_transfer_candidates(
                provisional_my_team, current_projections, candidate_pool, buy_prices
            )
            if plan.recommended is not None:
                candidate = plan.recommended
                transfer_in = candidate.buy_player_id
                transfer_out = candidate.sell_player_id
                hit_cost = candidate.hit_cost
                bought_position = current_projections[transfer_out].position
                new_squad = tuple(
                    (
                        SquadPlayer(
                            player_id=candidate.buy_player_id,
                            position=bought_position,
                            purchase_price=candidate.buy_price,
                            current_price=candidate.buy_price,
                        )
                        if p.player_id == transfer_out
                        else p
                    )
                    for p in state.squad
                )
                new_free_transfers = (
                    state.free_transfers - 1 if hit_cost == 0 else state.free_transfers
                )
                state = replace(
                    state,
                    squad=new_squad,
                    bank=state.bank + candidate.sell_price - candidate.buy_price,
                    free_transfers=new_free_transfers,
                )
            if chip_played in ("bench_boost", "triple_captain"):
                state = replace(state, chip_usage=record_chip_played(state.chip_usage, chip_played))

        # --- finalize XI/captain on the (possibly just-changed) squad ---
        final_xi, bench_order = select_starting_xi(state.squad, gw_points)
        final_my_team = _to_my_team_state(
            state, final_xi, bench_order, final_xi[0], final_xi[1], gameweek
        )
        captain_id, vice_captain_id = _select_captain_and_vice(final_my_team, gw_pool)

        baseline_xi, baseline_bench = select_starting_xi(baseline_squad, gw_points)
        baseline_my_team = _to_my_team_state(
            SimulatorSquad(baseline_squad, 0, 1, ChipUsage()),
            baseline_xi,
            baseline_bench,
            baseline_xi[0],
            baseline_xi[1],
            gameweek,
        )
        baseline_captain, baseline_vice = _select_captain_and_vice(baseline_my_team, gw_pool)

        # --- reveal real results and score ---
        points_by_player = {
            int(row.player_id): float(row.total_points) for row in gw_frame.itertuples()
        }
        minutes_by_player = {int(row.player_id): int(row.minutes) for row in gw_frame.itertuples()}

        effective_xi = apply_autosubs(final_xi, bench_order, state.squad, minutes_by_player)
        gw_score = _score_squad(
            effective_xi,
            bench_order,
            captain_id,
            vice_captain_id,
            chip_played,
            points_by_player,
            minutes_by_player,
        )

        baseline_effective_xi = apply_autosubs(
            baseline_xi, baseline_bench, baseline_squad, minutes_by_player
        )
        baseline_score = _score_squad(
            baseline_effective_xi,
            baseline_bench,
            baseline_captain,
            baseline_vice,
            None,
            points_by_player,
            minutes_by_player,
        )

        engine_total += gw_score
        baseline_total += baseline_score

        # A banked free transfer accrues every gameweek regardless of chip/transfer activity
        # (real FPL rule) -- whether one was *consumed* this week is already reflected in
        # `state.free_transfers` above.
        state = replace(state, free_transfers=min(state.free_transfers + 1, 5))

        log.append(
            GameweekRecord(
                gameweek=gameweek,
                chip_played=chip_played,
                transfer_in=transfer_in,
                transfer_out=transfer_out,
                hit_cost=hit_cost,
                captain_id=captain_id,
                vice_captain_id=vice_captain_id,
                starting_xi=final_xi,
                points_scored=gw_score,
                baseline_points_scored=baseline_score,
                running_total=engine_total,
                baseline_running_total=baseline_total,
            )
        )

    real_recorded_total = None
    if entry_id_for_benchmark is not None and fpl_client is not None:
        history = fpl_client.get_entry_history(entry_id_for_benchmark)
        label = season_label(season_start_year)
        for entry in history.get("past", []):
            if entry.get("season_name") == label:
                real_recorded_total = float(entry["total_points"])
                break

    return SimulationReport(
        season_start_year=season_start_year,
        engine_total_points=engine_total,
        baseline_total_points=baseline_total,
        gameweek_log=tuple(log),
        skipped_gameweeks=tuple(skipped),
        real_recorded_total=real_recorded_total,
    )
