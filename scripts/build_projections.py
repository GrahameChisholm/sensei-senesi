"""The batch job (team-page plan Phase 1.3 / D7): live FPL + Understat snapshot -> cross-season-
augmented feature inputs -> fitted horizon projections -> cold-start baselines for every remaining
live player -> one JSON cache file the API only ever reads.

Reference for the snapshot/logging wiring style: ``git show d51af0e^:scripts/weekly_refresh.py``
(deleted in the web-app-removal commit). This job differs from it in three ways that module's own
docstring already named as out of scope: it closes the true GW1 cold start
(``engine.data.cross_season``/``engine.data.live_horizon.augment_feature_inputs_with_prior_season``),
it gives every remaining live player a flagged baseline rather than dropping them
(``engine.data.cold_start``), and it builds a real multi-gameweek horizon in one pass
(``engine.data.live_horizon.build_live_horizon_from_feature_inputs``) rather than one gameweek at
a time.

Deliberately split into pure, disk/network-free assembly functions
(:func:`merge_cold_start_projections`, :func:`assemble_projection_cache`,
:func:`build_fixture_list`, :func:`write_projection_cache`) and a thin orchestrating
:func:`build_projections` — the same split ``engine/data/live_horizon.py``
already uses between its disk-free core and its disk-touching wrapper, for the same reason: the
logic that actually needs checking carefully (does every live player end up projected exactly
once, does the cache validate) is testable without a real network pull.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from backtest.prediction_log import current_model_version, log_predictions
from backtest.run_season import (
    DEFAULT_CACHE_DIR,
    build_season_crosswalk,
    fetch_understat_league_data_raw,
    fetch_understat_player_histories,
    fetch_vaastav_merged_gw,
    fetch_vaastav_teams,
)
from engine.data.availability_log import append_observations, build_availability_observations
from engine.data.cold_start import ColdStartPriors, baseline_projection, fit_cold_start_priors
from engine.data.cross_season import (
    fetch_vaastav_players_raw,
    player_code_map,
    prior_season_merged_gw,
    remap_player_histories,
    synthetic_team_rows,
    team_id_map,
)
from engine.data.fpl_client import FPLClient
from engine.data.ingest import capture_current_gameweek
from engine.data.live_adapter import (
    DEFAULT_TOTAL_MANAGERS,
    build_live_availability,
    snapshot_to_feature_inputs,
)
from engine.data.live_horizon import (
    augment_feature_inputs_with_prior_season,
    build_live_horizon_from_feature_inputs,
)
from engine.data.player_history import PlayerGameweekActual, load_live_player_history
from engine.data.snapshots import DEFAULT_BASE_DIR, load_snapshot_tables
from engine.data.team_rates import TeamRateSnapshot, build_current_team_rates
from engine.data.understat_client import UnderstatClient
from engine.projections import (
    PlayerGameweekProjection,
    PlayerHorizonProjection,
    project_player_horizon,
)
from engine.scoring import ELEMENT_TYPE_TO_POSITION

__all__ = [
    "DEFAULT_HORIZON",
    "DEFAULT_OUTPUT_DIR",
    "build_fixture_list",
    "merge_cold_start_projections",
    "assemble_projection_cache",
    "write_projection_cache",
    "build_projections",
    "main",
]

# Current gameweek + next 2 -- matches engine.data.live_horizon.DEFAULT_HORIZON_LENGTH and the
# team-page plan's D11 (the pitch's own "Next 3 GWs" view never needs more than this).
DEFAULT_HORIZON = 3
DEFAULT_OUTPUT_DIR = Path("data_store/projections")


# =================================================================================================
# Pure assembly -- no network, no disk beyond the final atomic write
# =================================================================================================


def build_fixture_list(fixtures: pd.DataFrame) -> list[dict]:
    """One row per team-perspective fixture (both the home and away sides of every match) --
    exactly the cache's own ``fixtures`` shape (§6.1 of the team-page plan): ``team_id``,
    ``opponent_id``, ``gameweek``, ``is_home``, ``kickoff_time``, ``difficulty``. A fixture with no
    assigned gameweek yet (``event`` is null -- not yet scheduled) is skipped, matching how a real
    blank gameweek already has no fixture row for the affected teams.

    ``difficulty`` is FPL's own ``team_h_difficulty``/``team_a_difficulty`` (1 easiest to 5
    hardest, from this team's own perspective for this specific fixture), carried straight through
    for the fixtures ticker rather than derived. Read defensively via ``getattr`` like ``event``
    above, since a hand built test frame may not carry these columns."""
    rows: list[dict] = []
    for row in fixtures.itertuples():
        event = getattr(row, "event", None)
        if event is None or pd.isna(event):
            continue
        gameweek = int(event)
        kickoff = row.kickoff_time
        rows.append(
            {
                "team_id": int(row.team_h),
                "opponent_id": int(row.team_a),
                "gameweek": gameweek,
                "is_home": True,
                "kickoff_time": kickoff,
                "difficulty": getattr(row, "team_h_difficulty", None),
            }
        )
        rows.append(
            {
                "team_id": int(row.team_a),
                "opponent_id": int(row.team_h),
                "gameweek": gameweek,
                "is_home": False,
                "kickoff_time": kickoff,
                "difficulty": getattr(row, "team_a_difficulty", None),
            }
        )
    return rows


def merge_cold_start_projections(
    live_elements: pd.DataFrame,
    engine_projections: dict[int, PlayerHorizonProjection],
    priors: ColdStartPriors,
    team_id_by_player: dict[int, int],
    team_gameweeks_with_fixture: set[tuple[int, int]],
    target_gameweeks: Sequence[int],
) -> tuple[dict[int, PlayerHorizonProjection], set[int]]:
    """Every live element ends up projected exactly once -- the real engine output if one exists,
    a flagged :func:`~engine.data.cold_start.baseline_projection` otherwise (D5) -- rather than the
    engine's own dropna silently vanishing the ~107 historyless players from the page entirely.

    A cold-start player still only gets a projection for a gameweek their own team actually has a
    fixture in (``team_gameweeks_with_fixture``), matching the "blank gameweek gets no entry, not
    a zero" rule every other projection already follows; a player whose team has no fixture at all
    across the whole horizon is left out entirely (nothing to show).

    Returns ``(full_projections, cold_start_player_ids)`` -- the id set drives both the cache's own
    ``low_confidence``/``source`` fields and the build summary's own reporting.

    Each player's price rank among their own club's *entire current squad* at the same position
    (1 = the most expensive, so presumptively most nailed) is passed to
    :func:`~engine.data.cold_start.baseline_projection` as ``within_club_position_rank``, so two
    cold-start players who share a price bucket, such as a newly-promoted club's first- and
    third-choice striker, no longer collapse onto the same projection. Ranked against the whole
    squad rather than only its other cold-start players, since an established (non-cold-start)
    incumbent at that position still correctly pushes a cold-start backup down a rank tier even
    though the incumbent itself never calls into this function.
    """
    result = dict(engine_projections)
    cold_start_ids: set[int] = set()
    position_by_player = {
        int(row.id): ELEMENT_TYPE_TO_POSITION[int(row.element_type)]
        for row in live_elements.itertuples()
    }
    price_rank_by_player: dict[int, int] = {}
    for (_team_id, _position), group in live_elements.assign(
        _position=live_elements["id"].map(position_by_player)
    ).groupby([live_elements["team"], "_position"]):
        ranks = group["now_cost"].rank(ascending=False, method="min").astype(int)
        price_rank_by_player.update(zip(group["id"].astype(int), ranks, strict=True))
    for row in live_elements.itertuples():
        player_id = int(row.id)
        if player_id in result:
            continue
        position = position_by_player[player_id]
        price = int(row.now_cost)
        team_id = team_id_by_player.get(player_id)
        within_club_position_rank = price_rank_by_player.get(player_id)
        gameweeks: dict[int, PlayerGameweekProjection] = {}
        for gameweek in target_gameweeks:
            if team_id is None or (team_id, gameweek) not in team_gameweeks_with_fixture:
                continue
            gameweeks[gameweek] = baseline_projection(
                player_id, position, price, gameweek, priors, within_club_position_rank
            )
        if not gameweeks:
            continue
        result[player_id] = project_player_horizon(player_id, position, gameweeks)
        cold_start_ids.add(player_id)
    return result, cold_start_ids


def _serialize_breakdown(breakdown) -> dict:
    return {
        "appearance": breakdown.appearance,
        "goals": breakdown.goals,
        "assists": breakdown.assists,
        "clean_sheet": breakdown.clean_sheet,
        "goals_conceded": breakdown.goals_conceded,
        "defensive_contribution": breakdown.defensive_contribution,
        "saves": breakdown.saves,
        "bonus": breakdown.bonus,
        "cards": breakdown.cards,
        "penalty_misses": breakdown.penalty_misses,
        "own_goals": breakdown.own_goals,
    }


def _serialize_minutes(minutes) -> dict:
    return {
        "p_zero": minutes.p_zero,
        "p_1_to_59": minutes.p_1_to_59,
        "p_60_plus": minutes.p_60_plus,
        "expected_minutes_given_1_to_59": minutes.expected_minutes_given_1_to_59,
        "expected_minutes_given_60_plus": minutes.expected_minutes_given_60_plus,
    }


def _serialize_simulation(simulation) -> dict | None:
    if simulation is None:
        return None
    return {
        "mean": simulation.mean,
        "median": simulation.median,
        "floor": simulation.floor,
        "ceiling": simulation.ceiling,
        "prob_big_haul": simulation.prob_big_haul,
    }


def _serialize_gameweek_projection(projection: PlayerGameweekProjection) -> dict:
    return {
        "gameweek": projection.gameweek,
        "breakdown": _serialize_breakdown(projection.breakdown),
        "minutes": _serialize_minutes(projection.minutes),
        "simulation": _serialize_simulation(projection.simulation),
        # ENGINE_IMPROVEMENTS_5.md Tier 2.1: E[points | plays 60+]. The breakdown above already
        # sums to the availability-weighted expected points, so this is the other half a manager
        # needs and cannot derive from it (dividing by p_60_plus over-inflates badly, see
        # engine.pipeline._plays_60_counterfactual). Null for a cold-start baseline, which has no
        # component chain to re-run.
        "conditional_expected_points": projection.conditional_expected_points,
    }


def _serialize_horizon_projection(horizon: PlayerHorizonProjection) -> dict:
    return {
        "position": horizon.position,
        "gameweeks": [
            _serialize_gameweek_projection(horizon.gameweeks[gw])
            for gw in sorted(horizon.gameweeks)
        ],
    }


def _isoformat(value) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return pd.Timestamp(value).isoformat()


def _serialize_player_history(history: list[PlayerGameweekActual]) -> list[dict]:
    return [
        {
            "gameweek": actual.gameweek,
            "minutes": actual.minutes,
            "goals_scored": actual.goals_scored,
            "assists": actual.assists,
            "clean_sheets": actual.clean_sheets,
            "goals_conceded": actual.goals_conceded,
            "own_goals": actual.own_goals,
            "penalties_saved": actual.penalties_saved,
            "penalties_missed": actual.penalties_missed,
            "saves": actual.saves,
            "yellow_cards": actual.yellow_cards,
            "red_cards": actual.red_cards,
            "bonus": actual.bonus,
            "defensive_contribution": actual.defensive_contribution,
            "total_points": actual.total_points,
            "expected_goals": actual.expected_goals,
            "expected_assists": actual.expected_assists,
            "expected_goal_involvements": actual.expected_goal_involvements,
            "expected_goals_conceded": actual.expected_goals_conceded,
            "selected": actual.selected,
            "starts": actual.starts,
            "value": actual.value,
            "transfers_in": actual.transfers_in,
            "transfers_out": actual.transfers_out,
            "bps": actual.bps,
        }
        for actual in history
    ]


def assemble_projection_cache(
    season: str,
    gameweek: int,
    horizon_gameweeks: Sequence[int],
    projections: dict[int, PlayerHorizonProjection],
    cold_start_ids: set[int],
    live_elements: pd.DataFrame,
    live_teams: pd.DataFrame,
    fixture_rows: list[dict],
    generated_at: datetime,
    deadline_time: datetime,
    deadline_passed: bool,
    model_version: str,
    diagnostics: dict,
    player_history: dict[int, list[PlayerGameweekActual]] | None = None,
    team_rates: dict[int, TeamRateSnapshot] | None = None,
) -> dict:
    """Build the exact JSON-serialisable cache dict §6.1 of the team-page plan specifies -- pure
    assembly from already-computed pieces, no I/O of its own.

    ``player_history`` (PLAYER_STATS_PLAN Phase 2) is optional so any caller/fixture built before
    the Player Stats page still assembles a valid cache; it defaults to no history for every
    player rather than requiring every call site to be updated.

    ``team_rates`` (fixture-swing plan Phase 1) is likewise optional so a cache built before this
    feature still assembles; it defaults to no rate for any team, which ``api/state.py`` treats the
    same way as a team missing from a real, current pull (no snapshot yet, e.g. true GW1).
    """
    team_rates = team_rates or {}
    player_history = player_history or {}
    team_name_by_id = {int(row.id): row.name for row in live_teams.itertuples()}
    team_short_name_by_id = {int(row.id): row.short_name for row in live_teams.itertuples()}

    players: dict[str, dict] = {}
    for row in live_elements.itertuples():
        player_id = int(row.id)
        chance = getattr(row, "chance_of_playing_next_round", None)
        ownership = getattr(row, "selected_by_percent", None)
        players[str(player_id)] = {
            "web_name": row.web_name,
            "full_name": f"{row.first_name} {row.second_name}",
            "team_id": int(row.team),
            "position": ELEMENT_TYPE_TO_POSITION[int(row.element_type)],
            "price": int(row.now_cost),
            "status": row.status,
            "chance_of_playing_next_round": float(chance) if pd.notna(chance) else 100.0,
            "low_confidence": player_id in cold_start_ids,
            "source": "cold_start" if player_id in cold_start_ids else "engine",
            "selected_by_percent": float(ownership) if pd.notna(ownership) else None,
        }

    teams = {
        str(team_id): {
            "name": team_name_by_id[team_id],
            "short_name": team_short_name_by_id[team_id],
        }
        for team_id in team_name_by_id
    }

    fixtures_out = [
        {
            "team_id": row["team_id"],
            "opponent_id": row["opponent_id"],
            "gameweek": row["gameweek"],
            "is_home": row["is_home"],
            "kickoff_time": _isoformat(row["kickoff_time"]),
            "difficulty": row["difficulty"],
        }
        for row in fixture_rows
    ]

    return {
        "season": season,
        "gameweek": gameweek,
        "horizon_gameweeks": list(horizon_gameweeks),
        "deadline_passed": deadline_passed,
        "generated_at": _isoformat(generated_at),
        "deadline_time": _isoformat(deadline_time),
        "model_version": model_version,
        "projections": {
            str(player_id): _serialize_horizon_projection(horizon)
            for player_id, horizon in projections.items()
        },
        "players": players,
        "teams": teams,
        "fixtures": fixtures_out,
        "diagnostics": diagnostics,
        "player_history": {
            str(player_id): _serialize_player_history(history)
            for player_id, history in player_history.items()
        },
        "team_rates": {
            str(team_id): {
                "home_xg_per_90": snapshot.home_xg_per_90,
                "away_xg_per_90": snapshot.away_xg_per_90,
                "home_xga_per_90": snapshot.home_xga_per_90,
                "away_xga_per_90": snapshot.away_xga_per_90,
            }
            for team_id, snapshot in team_rates.items()
        },
    }


def write_projection_cache(cache: dict, output_dir: Path, season: str, gameweek: int) -> Path:
    """Atomic write (temp file + rename) so a crash mid-write never leaves a partial cache file for
    the API to read."""
    target_dir = output_dir / season
    target_dir.mkdir(parents=True, exist_ok=True)
    final_path = target_dir / f"gw{gameweek:02d}.json"
    tmp_path = final_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(cache, indent=2))
    tmp_path.replace(final_path)
    return final_path


def _deadline_time_for_gameweek(events: pd.DataFrame, gameweek: int) -> datetime:
    matches = events[events["id"] == gameweek]
    if matches.empty:
        raise ValueError(f"no event found for gameweek {gameweek}")
    return pd.to_datetime(matches.iloc[0]["deadline_time"], utc=True).to_pydatetime()


# =================================================================================================
# Orchestration -- real network/disk I/O
# =================================================================================================


def build_projections(
    season: str,
    gameweek: int,
    understat_season_start_year: int,
    prior_season_start_year: int,
    horizon: int = DEFAULT_HORIZON,
    base_dir: Path = DEFAULT_BASE_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    n_simulation_runs: int = 200,
    seed: int | None = None,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
    reuse_snapshot: datetime | None = None,
    n_prior_seasons_for_team_rates: int = 3,
) -> Path:
    """The real end-to-end build: capture (or reuse) a live snapshot, close the GW1 cold start with
    cross-season history, fit once and project ``horizon`` gameweeks, fill every remaining live
    player with a flagged baseline, log predictions immutably, and write the cache atomically.

    Raises whatever :func:`~engine.data.live_horizon.build_live_horizon_from_feature_inputs` raises
    if even the cross-season-augmented training history is too thin -- a real, loud failure rather
    than a silent bad cache, exactly as that function's own docstring intends.
    """
    target_gameweeks = list(range(gameweek, gameweek + horizon))

    with (
        httpx.Client(timeout=30.0) as http_client,
        FPLClient() as fpl_client,
        UnderstatClient() as understat_client,
    ):
        if reuse_snapshot is not None:
            captured_at = reuse_snapshot
        else:
            manifest = capture_current_gameweek(
                fpl_client,
                understat_client,
                season,
                understat_season_start_year,
                gameweek,
                base_dir,
            )
            captured_at = manifest.captured_at

        fpl_tables = load_snapshot_tables(base_dir, season, gameweek, captured_at, "fpl")
        live_elements = fpl_tables["elements"]
        live_teams = fpl_tables["teams"]
        events = fpl_tables["events"]
        fixtures_df = fpl_tables["fixtures"]

        # --- cross-season history (engine.data.cross_season) --------------------------------
        prior_teams = fetch_vaastav_teams(prior_season_start_year, cache_dir, http_client)
        prior_players_raw = fetch_vaastav_players_raw(
            prior_season_start_year, cache_dir, http_client
        )
        prior_merged_gw_raw = fetch_vaastav_merged_gw(
            prior_season_start_year, cache_dir, http_client
        )

        code_map = player_code_map(prior_players_raw, live_elements)
        team_map = team_id_map(prior_teams, live_teams)
        synthetic_teams = synthetic_team_rows(prior_teams, live_teams)
        relegated_team_ids = dict(
            zip(
                prior_teams.loc[~prior_teams["code"].isin(live_teams["code"]), "id"],
                synthetic_teams["id"],
                strict=True,
            )
        )
        prior_merged_gw = prior_season_merged_gw(
            prior_merged_gw_raw, code_map, team_map, relegated_team_ids
        )

        prior_league_data = fetch_understat_league_data_raw(
            prior_season_start_year, cache_dir, understat_client
        )
        prior_crosswalk = build_season_crosswalk(
            prior_season_start_year, prior_league_data, cache_dir, http_client
        )
        prior_player_histories_by_prior_id = fetch_understat_player_histories(
            prior_crosswalk, prior_season_start_year, cache_dir, understat_client
        )
        prior_player_histories = remap_player_histories(
            prior_player_histories_by_prior_id, code_map
        )

        # --- live feature inputs (team-rate prior-season pooling already built in) -----------
        feature_inputs = snapshot_to_feature_inputs(
            season,
            gameweek,
            captured_at,
            understat_season_start_year,
            base_dir,
            total_managers,
            understat_client,
            cache_dir / "understat_prior_seasons",
            n_prior_seasons_for_team_rates,
            target_gameweeks=target_gameweeks,
        )
        augmented = augment_feature_inputs_with_prior_season(
            feature_inputs, prior_merged_gw, synthetic_teams, prior_player_histories
        )

        # --- fixture-swing plan Phase 1: every team's current xG/xGA rate, live for the first
        # time (previously only ever computed inside backtest's per-gameweek walk-forward replay).
        team_id_by_name = {row.name: int(row.id) for row in live_teams.itertuples()}
        team_rate_snapshots = build_current_team_rates(augmented.team_histories, captured_at)
        team_rates = {
            team_id_by_name[name]: snapshot
            for name, snapshot in team_rate_snapshots.items()
            if name in team_id_by_name
        }

        # --- fit once, project every horizon gameweek ----------------------------------------
        # T-F: live_elements is already loaded above (deadline lookup, cold-start fallback), so
        # building the real chance_of_playing_next_round/status override here is free -- without
        # it every player would keep engineer_features' "fully fit" default and injured/suspended
        # players would rank as if fully available (ENGINE_AUDIT_FIXES-implementation.md T-F).
        live_availability = build_live_availability(live_elements)
        # ENGINE_IMPROVEMENTS_5.md Tier 1.1: record this snapshot's availability signals before the
        # deadline, so a dataset the minutes model can actually learn from accumulates week by week.
        # `chance_of_playing_next_round`/`status_score` are in that model's FEATURE_COLUMNS but have
        # zero variance across every historical training row, so it currently learns no weight for
        # either -- the binding constraint on the engine's largest available lever (perfect minutes
        # knowledge would move pooled Spearman from 0.638 to 0.864). Writing this costs almost
        # nothing per run and cannot be backfilled later, so it starts now rather than when the
        # modelling work that consumes it begins.
        append_observations(
            build_availability_observations(live_elements, gameweek, captured_at),
            gameweek,
            captured_at,
        )
        horizon_result = build_live_horizon_from_feature_inputs(
            augmented,
            gameweek,
            target_gameweeks,
            n_simulation_runs=n_simulation_runs,
            seed=seed,
            live_availability=live_availability,
        )

        # --- cold-start fallback for every player the engine still couldn't cover -----------
        priors = fit_cold_start_priors(prior_merged_gw_raw)
        team_id_by_player = {int(row.id): int(row.team) for row in live_elements.itertuples()}
        fixture_rows = build_fixture_list(fixtures_df)

        # --- this season's actual per-gameweek performance (Player Stats page, D1/D4/D8) -----
        player_history = load_live_player_history(
            fpl_client, [int(row.id) for row in live_elements.itertuples()]
        )
        team_gameweeks_with_fixture = {(row["team_id"], row["gameweek"]) for row in fixture_rows}
        full_projections, cold_start_ids = merge_cold_start_projections(
            live_elements,
            horizon_result.projections,
            priors,
            team_id_by_player,
            team_gameweeks_with_fixture,
            target_gameweeks,
        )

        # --- immutable prediction log ------------------------------------------------------
        # ENGINE_IMPROVEMENTS_5.md Tier 0.2: this used to swallow FileExistsError, which quietly
        # broke the one guarantee the log exists to provide. Re-running against an already-logged
        # snapshot still rewrote the projection cache the API serves, while leaving the log holding
        # the *older* engine's numbers, so the "immutable accuracy record" no longer described the
        # predictions anyone acted on. That is not hypothetical: 2026-27 gw01.json was rebuilt on
        # 2026-08-24 from the 2026-08-20 snapshot and disagreed with the 2026-08-20 log on 1,191 of
        # 1,215 rows (max 3.31 points), both tagged `e82629b-dirty`.
        #
        # A repeat run now writes a *new* log keyed by its own wall-clock time rather than the
        # snapshot's capture time, so both runs are preserved and attributable, and the divergence
        # is visible instead of silent. The snapshot's `captured_at` is still what the first log is
        # stamped with, keeping the common single-run case unchanged.
        model_version = current_model_version()
        try:
            log_predictions(
                horizon_result.predictions, gameweek, model_version, logged_at=captured_at
            )
        except FileExistsError:
            log_predictions(
                horizon_result.predictions,
                gameweek,
                model_version,
                logged_at=datetime.now(UTC),
            )

        deadline_time = _deadline_time_for_gameweek(events, gameweek)
        deadline_passed = captured_at >= deadline_time

        diagnostics = {
            "training_rows": int(len(augmented.merged_gw)),
            "engine_projected_players": len(horizon_result.projections),
            "cold_start_players": len(cold_start_ids),
            "used_prior_season_history": True,
            "used_cold_start_pooling": True,
        }

        cache = assemble_projection_cache(
            season,
            gameweek,
            target_gameweeks,
            full_projections,
            cold_start_ids,
            live_elements,
            live_teams,
            fixture_rows,
            captured_at,
            deadline_time,
            deadline_passed,
            model_version,
            diagnostics,
            player_history,
            team_rates,
        )
        path = write_projection_cache(cache, output_dir, season, gameweek)
        _print_summary(cache, diagnostics, path)
        return path


def _print_summary(cache: dict, diagnostics: dict, path: Path) -> None:
    total_players = len(cache["players"])
    print(f"Wrote {path}")
    print(
        f"  players: {total_players} total, {diagnostics['engine_projected_players']} engine, "
        f"{diagnostics['cold_start_players']} cold-start"
    )
    print(f"  training rows: {diagnostics['training_rows']}")
    print(f"  deadline_passed: {cache['deadline_passed']}")
    significant_dropped = [
        p for pid, p in cache["players"].items() if p["source"] == "cold_start" and p["price"] >= 60
    ]
    if significant_dropped:
        names = ", ".join(f"{p['web_name']} (£{p['price']/10:.1f}m)" for p in significant_dropped)
        print(f"  significant cold-start players (>= £6.0m): {names}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Build the team-selection page's projection cache from live data."
    )
    parser.add_argument("--season", required=True, help='e.g. "2026-27"')
    parser.add_argument("--gameweek", type=int, required=True)
    parser.add_argument("--understat-season-start-year", type=int, required=True, help="e.g. 2026")
    parser.add_argument("--prior-season-start-year", type=int, required=True, help="e.g. 2025")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--n-simulation-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--reuse-snapshot",
        type=str,
        default=None,
        help="ISO timestamp of an already-captured snapshot to reuse instead of pulling live",
    )
    args = parser.parse_args(argv)

    reuse_snapshot = (
        datetime.fromisoformat(args.reuse_snapshot).astimezone(UTC) if args.reuse_snapshot else None
    )

    build_projections(
        season=args.season,
        gameweek=args.gameweek,
        understat_season_start_year=args.understat_season_start_year,
        prior_season_start_year=args.prior_season_start_year,
        horizon=args.horizon,
        n_simulation_runs=args.n_simulation_runs,
        seed=args.seed,
        reuse_snapshot=reuse_snapshot,
    )


if __name__ == "__main__":
    main()
