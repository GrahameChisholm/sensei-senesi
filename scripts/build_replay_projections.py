"""Season Replay batch job: turn one already-finished real season (vaastav's historical
``merged_gw``/``teams`` data, already cached locally under ``data_store/season_cache/``) into the
exact same per-gameweek projection-cache JSON shape the live Team Selection page already reads
(``scripts/build_projections.py``'s own cache format) -- so the whole existing app (squad builder,
pitch view, transfers, chips) can be pointed at a real historical season with zero frontend changes.

Deliberately reuses, rather than reimplements:
- :func:`prepare_replay_season_data` (below) for one point-in-time-correct ``engineered`` frame for
  the whole season, closing the *true* GW1 problem the same way the live path's
  ``scripts.build_projections.build_projections`` already does: :mod:`engine.data.cross_season`
  re-keys the **prior** season's real per-player history onto this season's element ids (via FPL's
  season-stable ``code`` field) and prepends it at negative gameweek numbers, so
  ``training_history = engineered[engineered.gameweek < gameweek]`` has real signal to fit on even
  at gameweek 1 -- not just for a genuinely new-to-the-league player, but for any player who simply
  hasn't played *this* season yet. (An earlier version of this script skipped this and fell back to
  the generic position x price cold-start prior for *every* player at GW1-3 -- correct only for
  gameweek 1's true blank slate, wrong for gameweeks 2-3 once one gameweek of real prior-season-
  augmented history already exists, and needlessly low-confidence for GW1 itself given the same
  season is fully known in hindsight.)
- :func:`simulator.horizon.build_horizon_predictions` / :func:`build_horizon_projections` to fit
  the engine on strictly-prior gameweeks and project a 3-gameweek horizon, exactly like the live
  batch job does.
- :func:`engine.data.cold_start.fit_cold_start_priors` / :func:`baseline_projection` for the
  narrower remaining gap: a player with no prior-season match at all (new to the Premier League
  entirely -- a promoted club's own signing, or a debutant from a league Understat doesn't cover).
- ``scripts/build_projections.py``'s pure assembly helpers (:func:`build_fixture_list`,
  :func:`merge_cold_start_projections`, :func:`assemble_projection_cache`,
  :func:`write_projection_cache`) unchanged -- they are already network-free and season-agnostic.

The one thing the live job gets from a live FPL snapshot that this job cannot -- a per-gameweek
``elements``/``teams`` frame with real names/prices/positions -- is instead built directly from
vaastav's own raw ``merged_gw`` per-gameweek slice (see :func:`vaastav_elements_for_gameweek`),
which (unlike the *engineered*, dropna'd frame) has full player coverage every single gameweek.
That same raw frame also supplies :func:`build_ground_truth` -- the real ``total_points``/
``minutes`` per player per gameweek that ``POST /squad/advance`` scores a committed squad against.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pandas as pd

from backtest.run_season import (
    DEFAULT_CACHE_DIR,
    build_season_crosswalk,
    build_team_rate_histories,
    engineer_features,
    fetch_understat_league_data_raw,
    fetch_understat_multi_season_league_data,
    fetch_understat_player_histories,
    fetch_vaastav_merged_gw,
    fetch_vaastav_teams,
    fit_fn,
)
from engine.data.cold_start import fit_cold_start_priors
from engine.data.cross_season import (
    fetch_vaastav_players_raw,
    merge_player_histories,
    player_code_map,
    prior_season_merged_gw,
    remap_player_histories,
    synthetic_team_rows,
    team_id_map,
)
from engine.data.understat_client import UnderstatClient, league_data_to_dataframes
from engine.scoring import DEF, FWD, GK, MID
from scripts.build_projections import (
    assemble_projection_cache,
    merge_cold_start_projections,
    write_projection_cache,
)
from simulator.horizon import build_horizon_predictions, build_horizon_projections

__all__ = [
    "DEFAULT_REPLAY_HORIZON",
    "DEFAULT_MIN_TRAINING_GAMEWEEKS",
    "POSITION_TO_ELEMENT_TYPE",
    "prepare_replay_season_data",
    "vaastav_elements_for_gameweek",
    "build_replay_fixtures",
    "build_ground_truth",
    "build_replay_season",
]

DEFAULT_REPLAY_HORIZON = 3
DEFAULT_MIN_TRAINING_GAMEWEEKS = 3
DEFAULT_OUTPUT_DIR = Path("data_store/projections")
DEFAULT_RESULTS_DIR = Path("data_store/replay")

# The inverse of engine.scoring.ELEMENT_TYPE_TO_POSITION -- vaastav's own `position` column already
# uses these exact GK/DEF/MID/FWD strings, so this is a pure lookup, not a guess.
POSITION_TO_ELEMENT_TYPE = {GK: 1, DEF: 2, MID: 3, FWD: 4}


def vaastav_elements_for_gameweek(
    merged_gw: pd.DataFrame, teams: pd.DataFrame, gameweek: int
) -> pd.DataFrame:
    """One row per player who has a ``merged_gw`` row at ``gameweek`` (i.e. every player vaastav's
    export tracks that week, whether they played or not) -- reshaped into exactly the column names
    ``scripts.build_projections.assemble_projection_cache``/``merge_cold_start_projections`` expect
    from a live FPL ``elements`` frame (``id``, ``web_name``, ``first_name``, ``second_name``,
    ``team``, ``element_type``, ``now_cost``, ``status``, ``chance_of_playing_next_round``).

    Historical vaastav data has no live-only fields (injury status, minutes-chance) -- every player
    is marked fully available (``status="a"``, ``chance_of_playing_next_round=100.0``), a documented
    simplification: this is about replaying real results, not modelling historical fitness news.

    ``web_name`` is everything after the first token (not just the last token): vaastav's ``name``
    is a plain "First Last" or "First Middle Last" string with no structured surname field, and for
    a player with two surnames (e.g. "David Raya Martín") the true FPL web_name ("Raya") isn't
    reliably derivable -- taking only the last token instead produces "Martín", which is both a
    display bug (nobody recognises it) and a real search bug, since ``/players?search=`` only
    matches ``web_name``. "Raya Martín" is longer than ideal but is at least the name a manager
    would actually search for.
    """
    team_name_to_id = dict(zip(teams["name"], teams["id"], strict=True))
    rows = merged_gw[merged_gw["GW"] == gameweek]
    if rows.empty:
        raise ValueError(f"no merged_gw rows for gameweek {gameweek}")

    def _surname(full_name: str) -> str:
        tokens = full_name.split(" ")
        return " ".join(tokens[1:]) if len(tokens) > 1 else tokens[0]

    out = pd.DataFrame(
        {
            "id": rows["element"].astype(int),
            "web_name": rows["name"].astype(str).map(_surname),
            "first_name": rows["name"].astype(str).map(lambda n: n.split(" ")[0]),
            "second_name": rows["name"].astype(str).map(_surname),
            "team": rows["team"].map(team_name_to_id).astype(int),
            "element_type": rows["position"].map(POSITION_TO_ELEMENT_TYPE),
            "now_cost": rows["value"].astype(int),
            "status": "a",
            "chance_of_playing_next_round": 100.0,
        }
    )
    # A player transferred mid-season can appear more than once for the same gameweek in rare
    # vaastav export quirks (loan moves) -- keep the last (most complete) row per player id.
    return out.drop_duplicates(subset="id", keep="last").reset_index(drop=True)


def build_replay_fixtures(merged_gw: pd.DataFrame, teams: pd.DataFrame) -> list[dict]:
    """Reconstruct the cache's own ``fixtures`` shape (one row per team-perspective per gameweek)
    directly from ``merged_gw`` -- it already carries each player's own team, the opponent's numeric
    id, home/away, and kickoff time, so no separate fixtures export is needed."""
    team_name_to_id = dict(zip(teams["name"], teams["id"], strict=True))
    dedup = merged_gw.drop_duplicates(subset=["team", "GW", "opponent_team"])
    rows: list[dict] = []
    for row in dedup.itertuples():
        rows.append(
            {
                "team_id": team_name_to_id[row.team],
                "opponent_id": int(row.opponent_team),
                "gameweek": int(row.GW),
                "is_home": bool(row.was_home),
                "kickoff_time": row.kickoff_time,
            }
        )
    return rows


def build_ground_truth(merged_gw: pd.DataFrame) -> dict[int, dict[int, dict]]:
    """Real recorded ``{gameweek: {player_id: {"minutes", "total_points"}}}`` -- read straight off
    vaastav's raw (never dropna'd) frame, so every player who actually has a gameweek row is
    covered, unlike the engineered/feature frame which drops rows lacking prior-gameweek history."""
    results: dict[int, dict[int, dict]] = {}
    for gameweek, rows in merged_gw.groupby("GW"):
        results[int(gameweek)] = {
            int(row.element): {"minutes": int(row.minutes), "total_points": float(row.total_points)}
            for row in rows.itertuples()
        }
    return results


def prepare_replay_season_data(
    season_start_year: int,
    prior_season_start_year: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> pd.DataFrame:
    """Like ``simulator.data_prep.prepare_season_data``, but with the prior season's own real
    per-player history re-keyed onto this season's element ids and prepended at negative gameweek
    numbers (``engine.data.cross_season``) before ``engineer_features`` runs -- closing the true
    GW1 problem the same way the live path's own batch job already does. See this module's
    docstring for why that matters for a replay specifically (gameweeks 1-3 would otherwise be
    100% cold-start baselines with no real per-player signal at all).
    """
    with httpx.Client(timeout=30.0) as http_client, UnderstatClient() as understat:
        merged_gw = fetch_vaastav_merged_gw(season_start_year, cache_dir, http_client)
        teams = fetch_vaastav_teams(season_start_year, cache_dir, http_client)
        league_data = fetch_understat_league_data_raw(season_start_year, cache_dir, understat)
        multi_season_league_data = fetch_understat_multi_season_league_data(
            season_start_year, cache_dir, understat
        )
        multi_season_teams_history = pd.concat(
            [
                league_data_to_dataframes(data)["teams_history"]
                for data in multi_season_league_data.values()
            ],
            ignore_index=True,
        )
        team_histories = build_team_rate_histories(multi_season_teams_history)
        crosswalk = build_season_crosswalk(season_start_year, league_data, cache_dir, http_client)
        player_histories = fetch_understat_player_histories(
            crosswalk, season_start_year, cache_dir, understat
        )

        # --- cross-season augmentation: real prior-season history, re-keyed onto this season ----
        prior_merged_gw_raw = fetch_vaastav_merged_gw(
            prior_season_start_year, cache_dir, http_client
        )
        prior_teams = fetch_vaastav_teams(prior_season_start_year, cache_dir, http_client)
        prior_players_raw = fetch_vaastav_players_raw(
            prior_season_start_year, cache_dir, http_client
        )
        current_players_raw = fetch_vaastav_players_raw(season_start_year, cache_dir, http_client)

        # `defensive_contribution` is a 2025/26-rules-only stat -- a prior season that predates it
        # genuinely scored zero from it for every player, a real historical fact, not a hack.
        if "defensive_contribution" not in prior_merged_gw_raw.columns:
            prior_merged_gw_raw = prior_merged_gw_raw.assign(defensive_contribution=0)

        code_map = player_code_map(prior_players_raw, current_players_raw)
        team_map = team_id_map(prior_teams, teams)
        synthetic_teams = synthetic_team_rows(prior_teams, teams)
        relegated_team_ids = dict(
            zip(
                prior_teams.loc[~prior_teams["code"].isin(teams["code"]), "id"],
                synthetic_teams["id"],
                strict=True,
            )
        )
        prior_merged_gw = prior_season_merged_gw(
            prior_merged_gw_raw, code_map, team_map, relegated_team_ids
        )

        prior_league_data = fetch_understat_league_data_raw(
            prior_season_start_year, cache_dir, understat
        )
        prior_crosswalk = build_season_crosswalk(
            prior_season_start_year, prior_league_data, cache_dir, http_client
        )
        prior_player_histories_by_prior_id = fetch_understat_player_histories(
            prior_crosswalk, prior_season_start_year, cache_dir, understat
        )
        prior_player_histories = remap_player_histories(
            prior_player_histories_by_prior_id, code_map
        )

    # prior_season_merged_gw's own output is already renamed to `player_id` (MERGED_GW_COLUMNS'
    # shape) -- but `merged_gw` (the current season's raw frame) still uses vaastav's own `element`
    # naming, since engineer_features does that rename itself, internally, on its actual input.
    # Renaming back to `element` here (rather than pre-renaming `merged_gw` to match) keeps both
    # sides using vaastav's own raw column name before concatenating, so engineer_features's own
    # rename runs exactly once, on a single unambiguous `element` column -- renaming one side but
    # not the other before concatenating would instead leave both an `element` and a `player_id`
    # column, and engineer_features's rename would then collide into two same-named columns (the
    # same class of bug this script's own cross_season.py fix just corrected for `GW`/`round`).
    prior_merged_gw = prior_merged_gw.rename(columns={"player_id": "element"})
    augmented_merged_gw = pd.concat([prior_merged_gw, merged_gw], ignore_index=True)
    augmented_teams = pd.concat([teams, synthetic_teams], ignore_index=True)
    augmented_player_histories = merge_player_histories(prior_player_histories, player_histories)

    engineered = engineer_features(
        augmented_merged_gw, augmented_teams, team_histories, augmented_player_histories
    )
    # Deliberately only *this* season's own teams for the target rows' team_id (matching
    # simulator.data_prep.prepare_season_data's own convention) -- a prior-season row with a
    # relegated club's name has no entry here and gets NaN, which is fine: this script's own
    # per-gameweek team_id_by_player always comes from vaastav_elements_for_gameweek instead, never
    # from this column.
    team_name_to_id = dict(zip(teams["name"], teams["id"], strict=True))
    engineered["team_id"] = engineered["team"].map(team_name_to_id)
    return engineered


def build_replay_season(
    season_start_year: int,
    prior_season_start_year: int,
    horizon: int = DEFAULT_REPLAY_HORIZON,
    min_training_gameweeks: int = DEFAULT_MIN_TRAINING_GAMEWEEKS,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    n_simulation_runs: int = 200,
    seed: int | None = 42,
) -> None:
    """Write one projection cache per gameweek (1..38) for ``season_start_year`` under
    ``output_dir/{season}/gw{NN}.json``, plus one combined ground-truth file at
    ``results_dir/{season}/results.json`` -- the two files ``api/state.py``'s replay support reads.
    """
    season = f"{season_start_year}-{str(season_start_year + 1)[-2:]}"
    print(f"Preparing point-in-time engineered frame for {season}...")
    engineered = prepare_replay_season_data(season_start_year, prior_season_start_year, cache_dir)
    # Real this-season gameweeks only -- prepare_replay_season_data's cross-season augmentation
    # prepends prior-season rows at negative gameweek numbers, which must never be treated as a
    # real target/horizon gameweek.
    all_gameweeks = sorted(gw for gw in engineered["gameweek"].unique().tolist() if gw >= 1)

    with httpx.Client(timeout=30.0) as http_client:
        merged_gw = fetch_vaastav_merged_gw(season_start_year, cache_dir, http_client)
        teams = fetch_vaastav_teams(season_start_year, cache_dir, http_client)
        prior_merged_gw = fetch_vaastav_merged_gw(prior_season_start_year, cache_dir, http_client)

    # `defensive_contribution` is a 2025/26-rules-only stat (engine.scoring's own "BPS reworked"
    # note) -- a prior season that predates it genuinely scored zero from it for every player, so
    # backfilling the column with 0 is a real historical fact, not a hack.
    if "defensive_contribution" not in prior_merged_gw.columns:
        prior_merged_gw = prior_merged_gw.assign(defensive_contribution=0)
    # 2024/25 introduced draftable "Assistant Manager" (position "AM") rows -- not one of this
    # engine's four scored positions (GK/DEF/MID/FWD), so irrelevant to a player cold-start prior.
    prior_merged_gw = prior_merged_gw[prior_merged_gw["position"].isin([GK, DEF, MID, FWD])]

    priors = fit_cold_start_priors(prior_merged_gw)
    fixture_rows = build_replay_fixtures(merged_gw, teams)
    team_gameweeks_with_fixture = {(row["team_id"], row["gameweek"]) for row in fixture_rows}

    results = build_ground_truth(merged_gw)
    results_path = results_dir / season
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "results.json").write_text(json.dumps(results, indent=2))
    print(f"Wrote {results_path / 'results.json'} ({len(results)} gameweeks)")

    generated_at = datetime.now(UTC)

    for gameweek in range(1, 39):
        target_gameweeks = [gw for gw in range(gameweek, gameweek + horizon) if gw in all_gameweeks]
        if not target_gameweeks:
            target_gameweeks = [gameweek]

        training_history = engineered[engineered["gameweek"] < gameweek]
        if training_history["gameweek"].nunique() < min_training_gameweeks:
            # Too little (or zero, at GW1) in-season history to fit anything yet -- every player
            # this gameweek comes from the cold-start prior, same as the live GW1 problem.
            engine_projections: dict = {}
        else:
            fitted = fit_fn(training_history)
            predictions = build_horizon_predictions(
                engineered, fitted, target_gameweeks, n_simulation_runs=n_simulation_runs, seed=seed
            )
            engine_projections = build_horizon_projections(predictions)

        live_elements = vaastav_elements_for_gameweek(merged_gw, teams, gameweek)
        team_id_by_player = {int(row.id): int(row.team) for row in live_elements.itertuples()}

        full_projections, cold_start_ids = merge_cold_start_projections(
            live_elements,
            engine_projections,
            priors,
            team_id_by_player,
            team_gameweeks_with_fixture,
            target_gameweeks,
        )

        deadline_time = merged_gw.loc[merged_gw["GW"] == gameweek, "kickoff_time"].min()
        diagnostics = {
            "training_rows": int(len(training_history)),
            "engine_projected_players": len(engine_projections),
            "cold_start_players": len(cold_start_ids),
            "note": "Season Replay batch job -- real historical vaastav data, not live FPL data",
        }

        cache = assemble_projection_cache(
            season=season,
            gameweek=gameweek,
            horizon_gameweeks=target_gameweeks,
            projections=full_projections,
            cold_start_ids=cold_start_ids,
            live_elements=live_elements,
            live_teams=teams,
            fixture_rows=fixture_rows,
            generated_at=generated_at,
            deadline_time=pd.to_datetime(deadline_time, utc=True).to_pydatetime(),
            deadline_passed=False,
            model_version=f"replay-{season}",
            diagnostics=diagnostics,
        )
        path = write_projection_cache(cache, output_dir, season, gameweek)
        print(
            f"GW{gameweek:02d}: {len(full_projections)} players projected "
            f"({diagnostics['engine_projected_players']} engine, "
            f"{diagnostics['cold_start_players']} cold-start) -> {path}"
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build a Season Replay projection cache set.")
    parser.add_argument("--season-start-year", type=int, default=2025, help="e.g. 2025 for 2025/26")
    parser.add_argument("--prior-season-start-year", type=int, default=2024)
    parser.add_argument("--horizon", type=int, default=DEFAULT_REPLAY_HORIZON)
    parser.add_argument("--n-simulation-runs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_replay_season(
        season_start_year=args.season_start_year,
        prior_season_start_year=args.prior_season_start_year,
        horizon=args.horizon,
        n_simulation_runs=args.n_simulation_runs,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
