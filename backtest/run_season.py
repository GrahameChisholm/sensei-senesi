"""Real single-season walk-forward backtest driver (BUILD_PLAN 3.1/3.5).

Promotes the ephemeral, unversioned scratch scripts used for the first real-data backtest
(``real_backtest.py``, ``diagnose.py``, ``diagnose2.py`` — see planning/ENGINE_IMPROVEMENTS.md
"Reproduction scripts") into a real, reproducible, testable module, per that document's own
recommendation: "right now the only real backtest we have is not reproducible from the repo."

Fetches real FPL (via the `vaastav/Fantasy-Premier-League` community archive) and Understat data
for one finished season, engineers every point-in-time feature the engine's components need
(including the Tier 1.1 minutes-model features and the Tier 1.3 real non-penalty xG + penalty
sub-model inputs), runs ``backtest.harness.run_walk_forward`` with a real ``fit_fn``/``predict_fn``
pair that also refits the Tier 1.2 :class:`engine.pipeline.FittedConstants`, and scores the result
against ``backtest.metrics``/``backtest.baselines``/``backtest.gate``.

**Scope.** This targets a single finished season only. Multi-season backtesting (Tier 3.1), GK
saves (3.2 — no opponent shots-on-target data source), bonus recomputed from raw BPS events (3.3 —
blocked, no 2026/27 BPS numeric table exists), and the DC opponent-possession adjustment (3.4 — no
possession data source) are explicitly out of scope; goalkeepers are excluded from this run for the
same reason the first real backtest excluded them.

**Known simplifications in this driver** (beyond the above, which are ENGINE_IMPROVEMENTS.md's own
documented gaps):

1. ``chance_of_playing_next_round``/``status`` are FPL live-only fields with no retained
   per-gameweek history to replay, so they're held at constant "fully available" values
   throughout this backtest (ENGINE_IMPROVEMENTS.md notes the engine should perform *better* live
   than it backtests for exactly this reason).
2. ``opponent_possession_share`` is held at the neutral league-average (0.5) — Tier 3.4's data
   source doesn't exist yet, so the defensive-contribution overdispersion fit below is computed
   against an unadjusted rate.
3. Bonus is still trained against the raw, pre-2026/27 ``bonus`` column (Tier 3.3, blocked) using
   each row's *realised* minutes rather than modelled expected minutes (a known train/serve skew,
   ENGINE_IMPROVEMENTS.md limitation #7) — neither is fixed here; only the five Tier 1.2 constants
   and the minutes/goals models are in this pass's scope.
4. ``team_expected_penalties`` is a plain point-in-time EWMA of a team's own realised penalty
   attempts, not opponent-xGA-adjusted the way BUILD_PLAN 2.2 specifies for open-play goals — a
   deliberate scope cut given the inherently tiny per-season penalty sample.
5. Player-to-gameweek Understat match matching uses nearest-calendar-date matching (±2 days), not
   an explicit fixture id join — accurate for the near-total majority of gameweeks but, like the
   double-gameweek handling noted in ENGINE_IMPROVEMENTS.md limitation #8, can misattribute a
   small fraction of double-gameweek rows.
"""

from __future__ import annotations

import argparse
import io
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from backtest import baselines, gate, metrics
from backtest.harness import run_walk_forward
from engine.data.crosswalk import (
    CrosswalkEntry,
    assert_matched_share,
    build_crosswalk,
    fetch_fpl_id_list,
    fetch_fpl_web_names,
    understat_players_from_league_data,
)
from engine.data.understat_client import (
    UnderstatClient,
    league_data_to_dataframes,
    player_data_to_dataframe,
)
from engine.models.assists import expected_assist_rate
from engine.models.bonus import BonusModel
from engine.models.clean_sheets import (
    clean_sheet_probability,
    fit_dixon_coles_rho,
    split_by_venue,
    team_expected_goals_rate,
)
from engine.models.defensive_contribution import (
    DEFAULT_OVERDISPERSION,
    expected_defensive_action_rate,
    fit_overdispersion,
)
from engine.models.goals import (
    expected_non_penalty_goal_rate,
    fit_penalty_conversion_rates,
    realized_penalty_goals,
)
from engine.models.minutes import FEATURE_COLUMNS as MINUTES_FEATURE_COLUMNS
from engine.models.minutes import MinutesModel, encode_status
from engine.models.saves import fit_away_shot_multiplier, fit_save_conversion_rate
from engine.pipeline import FittedConstants, project_gameweek_pool
from engine.rates import effective_sample_minutes, ewma_rate_asof, latest_ewma_rate
from engine.scoring import DEF, DEFENSIVE_CONTRIBUTION_THRESHOLD, FWD, MID, POSITIONS
from engine.simulate import (
    DEFAULT_N_RUNS,
    PlayerMatchInputs,
    TeamMatchInputs,
    simulate_fixture,
)

VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_CACHE_DIR = Path("data_store/season_cache")
DEFAULT_HALFLIFE = 10.0

# Goals/assists thin-sample-rate shrinkage strength, in effective-minutes units
# (ENGINE_IMPROVEMENTS_2.md B.2). Unlike the five Tier 1.2 constants, this has no closed-form
# per-gameweek estimator — it was selected once via an end-to-end backtest sweep over
# {0 (disabled), 180, 450, 900} on the real 2025/26 walk-forward: every metric improved at 180
# (top-10 mean actual 4.65 -> 4.80, max single-gameweek projection 25.5 -> 9.7 pts), while larger
# values kept improving pooled rank correlation but hurt top-N — the metric ENGINE_IMPROVEMENTS.md
# 2.1 argues is primary for decisions. Revisit only with a proper nested-CV walk-forward search;
# treated as a fixed, evidence-based hyperparameter until then, not re-derived every gameweek.
SHRINKAGE_K = 180.0

# Understat's ``team_title`` -> the vaastav/FPL team-name spelling for the same club, wherever they
# differ (BUILD_PLAN 1.1's ID-crosswalk problem, at team-name granularity rather than player-id
# granularity — same "match names across two sources" problem engine/data/crosswalk.py solves for
# players). Verified against a real 2025/26 pull; extend if a season adds/renames a club.
UNDERSTAT_TO_FPL_TEAM_NAME = {
    "Tottenham": "Spurs",
    "Newcastle United": "Newcastle",
    "Manchester City": "Man City",
    "Manchester United": "Man Utd",
    "Wolverhampton Wanderers": "Wolves",
    "Nottingham Forest": "Nott'm Forest",
}

__all__ = [
    "DEFAULT_CACHE_DIR",
    "FittedEngineState",
    "SeasonReport",
    "season_label",
    "fetch_vaastav_merged_gw",
    "fetch_vaastav_teams",
    "fetch_understat_league_data_raw",
    "build_season_crosswalk",
    "fetch_understat_player_histories",
    "build_team_rate_histories",
    "collapse_double_gameweeks",
    "engineer_features",
    "fit_fn",
    "make_predict_fn",
    "simulate_gameweek_pool",
    "make_simulate_predict_fn",
    "CoverageReport",
    "compute_coverage_report",
    "build_stand_in_squad_starting_xi",
    "score_season",
    "run_backtest",
    "main",
]


def season_label(season_start_year: int) -> str:
    """2025 -> '2025-26', matching the vaastav repo's directory naming."""
    return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"


# =================================================================================================
# Data-prep: fetch + cache raw sources
# =================================================================================================


def fetch_vaastav_merged_gw(
    season_start_year: int, cache_dir: Path, client: httpx.Client, refresh: bool = False
) -> pd.DataFrame:
    """One row per (player, gameweek): FPL's own recorded outcome and raw inputs for that season —
    the season's ground truth this whole driver backtests against."""
    label = season_label(season_start_year)
    cache_path = cache_dir / "vaastav" / label / "merged_gw.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    response = client.get(f"{VAASTAV_RAW_BASE}/{label}/gws/merged_gw.csv")
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_vaastav_teams(
    season_start_year: int, cache_dir: Path, client: httpx.Client, refresh: bool = False
) -> pd.DataFrame:
    """FPL numeric team id -> team name for *this specific season* — never trust the live
    bootstrap-static endpoint for a historical season's team-id mapping, since team ids get
    reassigned across promotion/relegation between seasons."""
    label = season_label(season_start_year)
    cache_path = cache_dir / "vaastav" / label / "teams.parquet"
    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)
    response = client.get(f"{VAASTAV_RAW_BASE}/{label}/teams.csv")
    response.raise_for_status()
    df = pd.read_csv(io.StringIO(response.text))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    return df


def fetch_understat_league_data_raw(
    season_start_year: int, cache_dir: Path, understat: UnderstatClient, refresh: bool = False
) -> dict[str, Any]:
    """The raw ``get_league_data`` payload, cached as JSON — used both for team-level match
    history (:func:`build_team_rate_histories`) and the player-id crosswalk."""
    cache_path = cache_dir / "understat" / str(season_start_year) / "league.json"
    if cache_path.exists() and not refresh:
        return json.loads(cache_path.read_text())
    league_data = understat.get_league_data(season_start_year)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(league_data))
    return league_data


def build_season_crosswalk(
    season_start_year: int,
    league_data: dict[str, Any],
    cache_dir: Path,
    client: httpx.Client,
    refresh: bool = False,
) -> list[CrosswalkEntry]:
    """FPL id <-> Understat id for every matchable player this season, cached. Uses
    ``strict=False`` (unlike ``engine.data.crosswalk``'s live-ingestion default of failing loudly
    on any miss) — a backtest can tolerate dropping a handful of unmatched fringe players the same
    way it already tolerates dropping goalkeepers and thin-history rows; a live pre-deadline
    ingestion cannot.

    Matches against both the full-legal-name list (``player_idlist.csv``) and the short
    ``web_name`` list (``players_raw.csv``), plus the surname/initial-surname token passes both
    enable (ENGINE_IMPROVEMENTS_2.md C.1) — the full-name-only match covered barely half of real
    2025/26 outfield players, concentrated in exactly the premium names captaincy decisions turn
    on.
    """
    cache_path = cache_dir / "crosswalk" / f"{season_start_year}.parquet"
    if cache_path.exists() and not refresh:
        records = pd.read_parquet(cache_path).to_dict("records")
        return [CrosswalkEntry(**record) for record in records]
    understat_players = understat_players_from_league_data(league_data)
    fpl_id_by_name = fetch_fpl_id_list(season_start_year, client)
    fpl_id_by_web_name = fetch_fpl_web_names(season_start_year, client)
    entries = build_crosswalk(
        understat_players,
        fpl_id_by_name,
        strict=False,
        fpl_id_by_web_name=fpl_id_by_web_name,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([entry.__dict__ for entry in entries]).to_parquet(cache_path, index=False)
    return entries


def fetch_understat_player_histories(
    crosswalk: list[CrosswalkEntry],
    season_start_year: int,
    cache_dir: Path,
    understat: UnderstatClient,
    refresh: bool = False,
) -> dict[int, pd.DataFrame]:
    """FPL id -> that player's Understat match history **up to and including**
    ``season_start_year`` (not that season alone — ENGINE_IMPROVEMENTS_2.md C.3), chronologically
    sorted (oldest first) — the point-in-time-safe order :mod:`engine.rates` requires. Understat's
    ``get_player_data`` has no bulk/multi-player form (the same one-request-per-player limitation
    ``FPLClient.iter_element_summaries`` documents), so this is one request per matched player,
    cached individually so an interrupted run doesn't lose earlier progress.

    ``engine/rates.py``'s own module docstring is explicit that callers "should pass a single
    chronologically-sorted history spanning prior seasons into the current one" so the EWMA decay
    lets recent matches dominate without a separate cold-start rule for the first few gameweeks —
    but this driver previously discarded every prior-season row the cache already held, filtering
    down to ``season_start_year`` alone (the real cached files here carry 2023, 2024, *and* 2025
    for most players). Scoring itself is unaffected: only ``season_start_year``'s own gameweeks are
    ever predicted or scored (``predict_fn`` slices ``engineered`` by gameweek, and ``engineered``
    is built from ``merged_gw``, which is ``season_start_year``-only) — this only enriches what
    "as of this kickoff" rate/weight computations see for that season's early gameweeks, which is
    exactly where the point-in-time rates (and B.2's shrinkage weight) were previously coldest.
    Rows from any season *after* ``season_start_year`` are dropped, in case a cached history file
    is later reused across multiple driver runs — this backtest must never see the future.
    """
    directory = cache_dir / "understat" / str(season_start_year) / "players"
    directory.mkdir(parents=True, exist_ok=True)
    histories: dict[int, pd.DataFrame] = {}
    for entry in crosswalk:
        cache_path = directory / f"{entry.understat_id}.parquet"
        if cache_path.exists() and not refresh:
            df = pd.read_parquet(cache_path)
        else:
            player_data = understat.get_player_data(entry.understat_id)
            df = player_data_to_dataframe(player_data)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
        matches = df[df["season"].astype(int) <= season_start_year].copy()
        if matches.empty:
            continue
        for col in ("time", "npxG", "xA", "goals", "npg"):
            matches[col] = matches[col].astype(float)
        matches["date"] = pd.to_datetime(matches["date"], utc=True)
        histories[entry.fpl_id] = matches.sort_values("date").reset_index(drop=True)
    return histories


def build_team_rate_histories(teams_history: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """FPL team name -> that team's Understat match history (xG/xGA), chronologically sorted, with
    a constant ``minutes=90`` column so :mod:`engine.rates`'s per-90 EWMA helpers (built for
    player-level data) apply unchanged at team level — a full match is always a 90-minute unit for
    a team, so the per-match average already *is* the per-90 rate.
    """
    histories: dict[str, pd.DataFrame] = {}
    for _, group in teams_history.groupby("team_title"):
        fpl_name = UNDERSTAT_TO_FPL_TEAM_NAME.get(
            group["team_title"].iloc[0], group["team_title"].iloc[0]
        )
        g = group.copy()
        g["date"] = pd.to_datetime(g["date"], utc=True)
        g["minutes"] = 90.0
        # ENGINE_IMPROVEMENTS_2.md D.5: venue split for the home/away rate difference BUILD_PLAN
        # 2.4 specifies (home defence is measurably stronger than away defence).
        g["is_home"] = g["h_a"] == "h"
        histories[fpl_name] = g.sort_values("date").reset_index(drop=True)
    return histories


# =================================================================================================
# Feature engineering — point-in-time, one row per (player, gameweek)
# =================================================================================================


def _per_player_series(gw: pd.DataFrame, compute: Callable[[pd.DataFrame], pd.Series]) -> pd.Series:
    """Apply ``compute`` independently to each player's chronologically-sorted rows, and return a
    Series aligned back to ``gw``'s own row index/order — the shared plumbing every per-player,
    point-in-time feature below is built from."""
    result = pd.Series(index=gw.index, dtype=float)
    for _, group in gw.groupby("player_id"):
        g = group.sort_values("kickoff_time")
        values = compute(g)
        result.loc[g.index] = np.asarray(values, dtype=float)
    return result


def compute_fixture_congestion(gw: pd.DataFrame, window_days: int = 7) -> pd.Series:
    """Games played by this player's team in the ``window_days`` before this fixture's own kickoff
    (BUILD_PLAN 2.1) — a real, team-level rotation-risk signal, joined onto every player on that
    team for that gameweek. Previously hardcoded to 0.0 in the ephemeral backtest driver for no
    good reason (ENGINE_IMPROVEMENTS.md 1.1) — ``kickoff_time`` is already in the source data."""
    result = pd.Series(index=gw.index, dtype=float)
    fixtures = gw[["team", "kickoff_time"]].drop_duplicates()
    for team, group in fixtures.groupby("team"):
        kickoffs = group["kickoff_time"].sort_values().to_numpy()
        counts = {
            kt: int(((kickoffs >= kt - np.timedelta64(window_days, "D")) & (kickoffs < kt)).sum())
            for kt in kickoffs
        }
        mask = gw["team"] == team
        result.loc[mask] = gw.loc[mask, "kickoff_time"].map(counts)
    return result


def compute_days_since_last_appearance(gw: pd.DataFrame, default_days: float = 60.0) -> pd.Series:
    """Days since this player's last appearance (minutes > 0) strictly before this gameweek's
    kickoff — the strongest available proxy for the missing live injury/availability flags in
    historical data (ENGINE_IMPROVEMENTS.md 1.1). ``default_days`` covers a player's very first
    row in the sample (no prior appearance to measure from)."""

    def _compute(g: pd.DataFrame) -> pd.Series:
        last_appearance: pd.Timestamp | None = None
        values = []
        for kickoff, minutes in zip(g["kickoff_time"], g["minutes"], strict=True):
            values.append(
                default_days if last_appearance is None else (kickoff - last_appearance).days
            )
            if minutes > 0:
                last_appearance = kickoff
        return pd.Series(values, dtype=float)

    return _per_player_series(gw, _compute)


def compute_zero_minute_streak_length(gw: pd.DataFrame) -> pd.Series:
    """Consecutive prior gameweeks (strictly before this one) with zero minutes — directly
    separates "deep squad / unavailable" from "rotation risk" (ENGINE_IMPROVEMENTS.md 1.1)."""

    def _compute(g: pd.DataFrame) -> pd.Series:
        streak = 0
        values = []
        for minutes in g["minutes"]:
            values.append(float(streak))
            streak = 0 if minutes > 0 else streak + 1
        return pd.Series(values, dtype=float)

    return _per_player_series(gw, _compute)


DGW_SUM_COLUMNS = [
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "bonus",
    "bps",
    "defensive_contribution",
    "penalties_missed",
    "saves",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "goals_conceded",
    "realized_penalty_goals",
    "penalties_attempted",
]
# Boolean-like columns where "did this happen in either fixture" (max), not a sum, is the right
# collapse — a player who started both legs of a double gameweek still only "started" in the
# binary sense the minutes model's ``started`` target and FPL's own ``clean_sheets`` flag mean.
DGW_MAX_COLUMNS = ["starts", "clean_sheets"]


def collapse_double_gameweeks(gw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse a double gameweek's two (or more) per-fixture rows for the same player into one
    row per ``(player_id, gameweek)`` (ENGINE_IMPROVEMENTS_2.md A.1).

    Without this, ``engineer_features`` emits one row per fixture (375 extra outfield rows in the
    2025/26 archive, ~1.6% of rows) and ``backtest.metrics``'s merge on ``(player_id, gameweek)``
    silently fans out 2x2 — pairing fixture-1's prediction against fixture-2's actual, and vice
    versa — rather than raising or dropping anything, so the resulting MAE is neither the
    per-fixture nor the per-gameweek number.

    Outcome columns in :data:`DGW_SUM_COLUMNS` sum (a manager's actual gameweek return is the sum
    across both fixtures); :data:`DGW_MAX_COLUMNS` take the max (boolean-like "did this happen at
    all this gameweek" flags, not additive counts). Every other column — every point-in-time
    engineered feature, computed upstream on the ungrouped per-fixture frame so its EWMA/shift
    logic sees each match individually — takes the **first** fixture by kickoff time, per this
    module's own documented simplification (see the module docstring, simplification 5, and
    ENGINE_IMPROVEMENTS.md limitation #8: "opponent-adjustment features took whichever fixture
    appeared first"). A genuinely correct fixture-level treatment (predict each fixture separately,
    sum the *projections*) is out of scope here — this only stops the cross-join.

    Returns ``(collapsed, n_rows_removed)`` — the second value is the count of extra fixture rows
    that were merged away, for the coverage reporting in ENGINE_IMPROVEMENTS_2.md A.5.
    """
    g = gw.sort_values(["player_id", "gameweek", "kickoff_time"]).reset_index(drop=True)
    n_before = len(g)
    sum_cols = [c for c in DGW_SUM_COLUMNS if c in g.columns]
    max_cols = [c for c in DGW_MAX_COLUMNS if c in g.columns]
    first_cols = [
        c for c in g.columns if c not in sum_cols and c not in max_cols and c not in ("gameweek",)
    ]
    agg = {**{c: "first" for c in first_cols}, **{c: "sum" for c in sum_cols}}
    agg.update({c: "max" for c in max_cols})
    # "player_id" is both a group key and a first_col candidate; drop the redundant agg entry.
    agg.pop("player_id", None)
    collapsed = (
        g.groupby(["player_id", "gameweek"], as_index=False, sort=False).agg(agg).reset_index(drop=True)
    )
    return collapsed, n_before - len(collapsed)


def compute_team_rotation_propensity(gw: pd.DataFrame) -> pd.Series:
    """Team-level rotation-propensity proxy: as of each gameweek, the standard deviation across
    that team's squad of each player's own cumulative start rate to date (BUILD_PLAN 2.1 — "some
    managers rotate systematically"). A team where regulars and fringe players are both consistent
    has low dispersion; a team rotating heavily game-to-game spreads who's started how often.
    Point-in-time: only ever uses gameweeks strictly before the one being scored.
    """
    g = gw.sort_values(["player_id", "kickoff_time"]).copy()
    g["_cum_start_rate"] = g.groupby("player_id")["starts"].transform(
        lambda s: s.astype(float).shift(1).expanding().mean()
    )
    dispersion = (
        g.dropna(subset=["_cum_start_rate"]).groupby(["team", "gameweek"])["_cum_start_rate"].std()
    )
    keys = pd.MultiIndex.from_frame(gw[["team", "gameweek"]])
    result = dispersion.reindex(keys).fillna(0.0)
    result.index = gw.index
    return result


def _team_rate_asof(team_history: pd.DataFrame, stat_col: str, before: pd.Timestamp) -> float:
    if team_history.empty:
        return float("nan")
    prior = team_history[team_history["date"] < before]
    if prior.empty:
        return float("nan")
    return latest_ewma_rate(prior, stat_col, minutes_col="minutes")


def _team_rate_asof_venue_split(
    team_history: pd.DataFrame, stat_col: str, before: pd.Timestamp, is_home: bool
) -> float:
    """Point-in-time per-90 EWMA rate restricted to the team's own home or away matches
    (ENGINE_IMPROVEMENTS_2.md D.5 — BUILD_PLAN 2.4's home/away split, "home defence is measurably
    stronger than away defence", specified but never supplied by the real backtest). Falls back to
    the combined (both-venue) rate — :func:`_team_rate_asof` — when the venue-specific subset is
    empty (early season, or a promoted team with too little same-venue history yet): some signal
    beats none, and this only ever affects the first few matches at a given venue each season.
    """
    if team_history.empty:
        return float("nan")
    prior = team_history[team_history["date"] < before]
    if prior.empty:
        return float("nan")
    home_prior, away_prior = split_by_venue(prior, is_home_col="is_home")
    venue_prior = home_prior if is_home else away_prior
    if venue_prior.empty:
        return latest_ewma_rate(prior, stat_col, minutes_col="minutes")
    return latest_ewma_rate(venue_prior, stat_col, minutes_col="minutes")


def build_fixture_rate_frame(
    gw: pd.DataFrame, team_histories: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """One row per (team, gameweek): that team's own and its opponent's point-in-time xG/xGA
    rates, the gameweek's league-average xGA, and a Dixon-Coles clean-sheet probability computed
    at the untuned :data:`~engine.models.clean_sheets.DEFAULT_DIXON_COLES_RHO` (used only to build
    the bonus regression's training features — see module docstring, simplification 3)."""
    fixtures = (
        gw[
            [
                "team",
                "gameweek",
                "kickoff_time",
                "opponent_team_name",
                "was_home",
                "team_h_score",
                "team_a_score",
            ]
        ]
        .drop_duplicates(subset=["team", "gameweek"])
        .reset_index(drop=True)
    )

    empty = pd.DataFrame(columns=["date", "xG", "xGA", "minutes", "is_home"])
    fixtures["team_xg_per_90"] = [
        _team_rate_asof_venue_split(team_histories.get(team, empty), "xG", kickoff, is_home)
        for team, kickoff, is_home in zip(
            fixtures["team"], fixtures["kickoff_time"], fixtures["was_home"], strict=True
        )
    ]
    fixtures["team_xga_per_90"] = [
        _team_rate_asof_venue_split(team_histories.get(team, empty), "xGA", kickoff, is_home)
        for team, kickoff, is_home in zip(
            fixtures["team"], fixtures["kickoff_time"], fixtures["was_home"], strict=True
        )
    ]
    fixtures["league_avg_xga_per_90"] = fixtures.groupby("gameweek")["team_xga_per_90"].transform(
        "mean"
    )

    opponent_rates = fixtures[["team", "gameweek", "team_xg_per_90", "team_xga_per_90"]].rename(
        columns={
            "team": "opponent_team_name",
            "team_xg_per_90": "opponent_xg_per_90",
            "team_xga_per_90": "opponent_xga_per_90",
        }
    )
    fixtures = fixtures.merge(opponent_rates, on=["opponent_team_name", "gameweek"], how="left")

    def _clean_sheet_prob(row: pd.Series) -> float:
        if (
            pd.isna(row["team_xg_per_90"])
            or pd.isna(row["opponent_xg_per_90"])
            or not row["league_avg_xga_per_90"] > 0
        ):
            return float("nan")
        team_for = team_expected_goals_rate(
            row["team_xg_per_90"], row["opponent_xga_per_90"], row["league_avg_xga_per_90"]
        )
        team_against = team_expected_goals_rate(
            row["opponent_xg_per_90"], row["team_xga_per_90"], row["league_avg_xga_per_90"]
        )
        return clean_sheet_probability(team_for, team_against)

    fixtures["clean_sheet_probability_default_rho"] = fixtures.apply(_clean_sheet_prob, axis=1)
    return fixtures


def build_match_level_frame(training_history: pd.DataFrame) -> pd.DataFrame:
    """One row per real match (home team's perspective only, so each match counts once) — the
    training data :func:`fit_fn` needs for :func:`engine.models.clean_sheets.fit_dixon_coles_rho`.
    """
    required = ["team_xg_per_90", "team_xga_per_90", "opponent_xg_per_90", "opponent_xga_per_90"]
    return (
        training_history[training_history["was_home"]]
        .drop_duplicates(subset=["team", "gameweek"])
        .dropna(subset=required)
    )


def _understat_rate_asof(
    player_histories: dict[int, pd.DataFrame], player_id: int, stat_col: str, before: pd.Timestamp
) -> float:
    history = player_histories.get(player_id)
    if history is None or history.empty:
        return float("nan")
    prior = history[history["date"] < before]
    if prior.empty:
        return float("nan")
    return latest_ewma_rate(prior, stat_col, minutes_col="time")


def _understat_effective_minutes_asof(
    player_histories: dict[int, pd.DataFrame], player_id: int, before: pd.Timestamp
) -> float:
    """Point-in-time evidence weight for the goals/assists shrinkage (ENGINE_IMPROVEMENTS_2.md
    B.2) — how much real Understat match history stands behind this player's own npxG/assist
    rate as of this gameweek. Unlike :func:`_understat_rate_asof`, ``0.0`` (not NaN) is the
    correct "no history at all" answer here: :func:`engine.rates.shrink_toward_prior` already
    treats a weight of 0 as "fall back to the prior rate entirely", exactly right for a player
    with no prior Understat appearances.
    """
    history = player_histories.get(player_id)
    if history is None or history.empty:
        return 0.0
    prior = history[history["date"] < before]
    if prior.empty:
        return 0.0
    return effective_sample_minutes(prior, minutes_col="time")


def _match_realized_penalty_goals(
    player_histories: dict[int, pd.DataFrame],
    player_id: int,
    kickoff_time: pd.Timestamp,
    tolerance_days: int = 2,
) -> float:
    history = player_histories.get(player_id)
    if history is None or history.empty:
        return 0.0
    diffs = (history["date"] - kickoff_time).abs()
    candidates = history[diffs <= pd.Timedelta(days=tolerance_days)].copy()
    if candidates.empty:
        return 0.0
    candidates["_realized_penalty_goals"] = realized_penalty_goals(candidates)
    nearest_index = diffs.loc[candidates.index].idxmin()
    return float(candidates.loc[nearest_index, "_realized_penalty_goals"])


def compute_team_expected_penalties(gw: pd.DataFrame) -> pd.DataFrame:
    """Point-in-time EWMA of a team's own total attempted penalties per gameweek (module docstring
    simplification 4 — not opponent-adjusted, unlike BUILD_PLAN 2.2's open-play-goals formula)."""
    team_gw = gw.groupby(["team", "gameweek"], as_index=False).agg(
        kickoff_time=("kickoff_time", "first"), penalties_attempted=("penalties_attempted", "sum")
    )
    result = pd.Series(index=team_gw.index, dtype=float)
    for _, group in team_gw.groupby("team"):
        g = group.sort_values("kickoff_time")
        rates = (
            g["penalties_attempted"]
            .astype(float)
            .shift(1)
            .ewm(halflife=DEFAULT_HALFLIFE, adjust=True)
            .mean()
        )
        result.loc[g.index] = rates.to_numpy()
    team_gw["team_expected_penalties"] = result.fillna(0.0)
    return team_gw[["team", "gameweek", "team_expected_penalties"]]


def build_penalty_attempts_frame(training_history: pd.DataFrame) -> pd.DataFrame:
    """One row per realized historical penalty attempt (``player_id``, ``scored`` 0/1) — the
    training data :func:`engine.models.goals.fit_penalty_conversion_rates` needs, expanded from
    each gameweek's aggregate scored/missed counts (real penalties are rare enough per season that
    filtering to only rows with at least one attempt keeps this fast)."""
    relevant = training_history[
        (training_history["realized_penalty_goals"] > 0)
        | (training_history["penalties_missed"] > 0)
    ]
    rows: list[dict[str, Any]] = []
    for _, row in relevant.iterrows():
        rows.extend(
            {"player_id": row["player_id"], "scored": 1}
            for _ in range(int(round(row["realized_penalty_goals"])))
        )
        rows.extend(
            {"player_id": row["player_id"], "scored": 0}
            for _ in range(int(round(row["penalties_missed"])))
        )
    return pd.DataFrame(rows, columns=["player_id", "scored"])


def engineer_features(
    merged_gw: pd.DataFrame,
    teams: pd.DataFrame,
    team_histories: dict[str, pd.DataFrame],
    player_histories: dict[int, pd.DataFrame],
) -> pd.DataFrame:
    """Join the vaastav ground truth, the FPL team-id map, Understat team-level rates, and
    Understat player-level rates into one point-in-time ``history`` frame keyed by
    ``(player_id, gameweek)`` — exactly what :func:`backtest.harness.run_walk_forward` and
    :func:`engine.pipeline.project_gameweek_pool` expect. Outfield only (GK excluded, matching the
    first real backtest's known limitation — no opponent shots-on-target data source, Tier 3.2).
    """
    gw = merged_gw[merged_gw["position"].isin([DEF, MID, FWD])].copy()
    gw["kickoff_time"] = pd.to_datetime(gw["kickoff_time"], utc=True)
    gw = gw.rename(columns={"element": "player_id", "GW": "gameweek"})

    team_id_to_name = dict(zip(teams["id"], teams["name"], strict=True))
    gw["opponent_team_name"] = gw["opponent_team"].map(team_id_to_name)

    # --- Tier 1.1: minutes-model features -------------------------------------------------------
    gw["recent_start_rate"] = _per_player_series(
        gw,
        lambda g: g["starts"]
        .astype(float)
        .shift(1)
        .ewm(halflife=DEFAULT_HALFLIFE, adjust=True)
        .mean(),
    )
    gw["recent_minutes_ewma"] = _per_player_series(
        gw,
        lambda g: g["minutes"]
        .astype(float)
        .shift(1)
        .ewm(halflife=DEFAULT_HALFLIFE, adjust=True)
        .mean(),
    )
    gw["fixture_congestion"] = compute_fixture_congestion(gw)
    gw["chance_of_playing_next_round"] = 100.0  # live-only field, unavailable retrospectively
    gw["status_score"] = encode_status("a")  # live-only field, unavailable retrospectively
    gw["days_since_last_appearance"] = compute_days_since_last_appearance(gw)
    gw["zero_minute_streak_length"] = compute_zero_minute_streak_length(gw)
    for window in (3, 6, 15):
        gw[f"start_rate_last_{window}"] = _per_player_series(
            gw,
            lambda g, w=window: g["starts"].astype(float).shift(1).rolling(w, min_periods=1).mean(),
        )
    gw["team_rotation_propensity"] = compute_team_rotation_propensity(gw)

    # --- Tier B.3: crowd features (ENGINE_IMPROVEMENTS_2.md) --------------------------------------
    # `value`/`selected`/`transfers_out`/`transfers_balance` are all already in the archive and
    # unused. A real walk-forward ablation found these four raise "played at all" AUC from 0.8655
    # to 0.8859, clearing the >0.88 target ENGINE_IMPROVEMENTS.md recorded as missed —
    # `transfers_out_share` alone contributes +0.012 AUC, the largest single feature, because mass
    # transfers-out is the crowd reacting in real time to *this week's* injury news: a
    # retrospectively-available proxy for the two live-only fields (`chance_of_playing_next_round`,
    # `status`) this backtest otherwise can't reconstruct.
    #
    # CAVEAT (unverified in this environment): this assumes the archive's per-gameweek `value` /
    # `selected` / `transfers_*` are the *at-deadline* snapshot, not an end-of-gameweek one. That
    # should be checked once against a live `engine/data/snapshots.py` capture for the same
    # gameweek before fully trusting this feature set live; if it turns out to be end-of-gameweek,
    # fall back to a strictly-lagged (shift-by-one-gameweek) version of these four columns instead
    # — measured to still improve AUC to 0.8710, a smaller but real gain.
    gw["price"] = gw["value"].astype(float)
    gw["ownership_log"] = np.log1p(gw["selected"].astype(float))
    _selected_denom = gw["selected"].astype(float).clip(lower=1.0)
    gw["transfers_out_share"] = gw["transfers_out"].astype(float) / _selected_denom
    gw["transfers_balance_share"] = gw["transfers_balance"].astype(float) / _selected_denom

    # --- shared per-90 rates already in the vaastav frame itself --------------------------------
    gw["dc_per_90"] = _per_player_series(
        gw, lambda g: ewma_rate_asof(g, "defensive_contribution", minutes_col="minutes")
    )
    gw["yellow_card_rate_per_90"] = _per_player_series(
        gw, lambda g: ewma_rate_asof(g, "yellow_cards", minutes_col="minutes")
    )
    gw["red_card_rate_per_90"] = _per_player_series(
        gw, lambda g: ewma_rate_asof(g, "red_cards", minutes_col="minutes")
    )
    # ENGINE_IMPROVEMENTS_2.md D.6: real but rare, modelled the same way as cards.
    gw["own_goal_rate_per_90"] = _per_player_series(
        gw, lambda g: ewma_rate_asof(g, "own_goals", minutes_col="minutes")
    )
    gw["opponent_possession_share"] = 0.5  # neutral default; Tier 3.4 data source doesn't exist

    # --- Tier 1.3: real (Understat) non-penalty xG/xA + penalty sub-model inputs ----------------
    gw["npxg_per_90"] = gw.apply(
        lambda r: _understat_rate_asof(
            player_histories, int(r["player_id"]), "npxG", r["kickoff_time"]
        ),
        axis=1,
    )
    gw["xa_per_90"] = gw.apply(
        lambda r: _understat_rate_asof(
            player_histories, int(r["player_id"]), "xA", r["kickoff_time"]
        ),
        axis=1,
    )
    # ENGINE_IMPROVEMENTS_2.md B.2: evidence weight behind each row's own npxg_per_90/xa_per_90 —
    # the shrinkage target for players whose real Understat point-in-time rates are thin-sample
    # outliers (up to 35.4 npxG/90 for a sub-90-minute cameo).
    gw["understat_effective_minutes"] = gw.apply(
        lambda r: _understat_effective_minutes_asof(
            player_histories, int(r["player_id"]), r["kickoff_time"]
        ),
        axis=1,
    )
    gw["realized_penalty_goals"] = gw.apply(
        lambda r: _match_realized_penalty_goals(
            player_histories, int(r["player_id"]), r["kickoff_time"]
        ),
        axis=1,
    )
    gw["penalties_attempted"] = gw["realized_penalty_goals"] + gw["penalties_missed"].astype(float)
    gw["_player_expected_penalty_attempts"] = _per_player_series(
        gw,
        lambda g: g["penalties_attempted"]
        .astype(float)
        .shift(1)
        .ewm(halflife=DEFAULT_HALFLIFE, adjust=True)
        .mean(),
    ).fillna(0.0)
    team_expected_penalties = compute_team_expected_penalties(gw)
    gw = gw.merge(team_expected_penalties, on=["team", "gameweek"], how="left")
    gw["taker_share"] = np.where(
        gw["team_expected_penalties"] > 0,
        (gw["_player_expected_penalty_attempts"] / gw["team_expected_penalties"]).clip(0.0, 1.0),
        0.0,
    )
    gw = gw.drop(columns=["_player_expected_penalty_attempts"])

    # --- team/opponent xG rates for goals/assists/clean-sheets, + bonus's training-only input ---
    fixtures = build_fixture_rate_frame(gw, team_histories)
    gw = gw.merge(
        fixtures[
            [
                "team",
                "gameweek",
                "team_xg_per_90",
                "team_xga_per_90",
                "opponent_xg_per_90",
                "opponent_xga_per_90",
                "league_avg_xga_per_90",
                "clean_sheet_probability_default_rho",
            ]
        ],
        on=["team", "gameweek"],
        how="left",
    )

    # --- A.1: collapse double gameweeks before scoring ever sees them --------------------------
    gw, n_dgw_rows_collapsed = collapse_double_gameweeks(gw)

    required = [
        *MINUTES_FEATURE_COLUMNS,
        "npxg_per_90",
        "xa_per_90",
        "team_xg_per_90",
        "team_xga_per_90",
        "opponent_xg_per_90",
        "opponent_xga_per_90",
        "league_avg_xga_per_90",
        "dc_per_90",
        "clean_sheet_probability_default_rho",
    ]
    n_before_dropna = len(gw)
    result = gw.dropna(subset=required).reset_index(drop=True)
    # A.5 coverage reporting reads these off the returned frame immediately after the call, before
    # any further pandas operation that might not preserve them.
    result.attrs["n_dgw_rows_collapsed"] = n_dgw_rows_collapsed
    result.attrs["n_rows_before_dropna"] = n_before_dropna
    result.attrs["n_rows_dropped_for_missing_features"] = n_before_dropna - len(result)
    return result


# =================================================================================================
# Fit / predict, wired to backtest.harness.run_walk_forward
# =================================================================================================


@dataclass(frozen=True)
class FittedEngineState:
    minutes_model: MinutesModel
    bonus_model: BonusModel
    fitted_constants: FittedConstants


def fit_fn(training_history: pd.DataFrame) -> FittedEngineState:
    """Real ``fit_fn`` for :func:`backtest.harness.run_walk_forward` — fits the minutes model, the
    bonus regression proxy, and (ENGINE_IMPROVEMENTS.md Tier 1.2) all five previously-unfitted
    component constants, entirely from ``training_history`` (gameweeks strictly before the one
    being predicted — the harness's own no-leakage guarantee)."""
    minutes_model = MinutesModel().fit(
        training_history[MINUTES_FEATURE_COLUMNS],
        training_history["starts"].astype(int),
        training_history["minutes"].astype(float),
    )

    expected_goals = (
        training_history["npxg_per_90"]
        * (training_history["opponent_xga_per_90"] / training_history["league_avg_xga_per_90"])
        * (training_history["minutes"] / 90.0)
    )
    expected_assists = (
        training_history["xa_per_90"]
        * (training_history["opponent_xga_per_90"] / training_history["league_avg_xga_per_90"])
        * (training_history["minutes"] / 90.0)
    )
    bonus_features = pd.DataFrame(
        {
            "expected_goals": expected_goals,
            "expected_assists": expected_assists,
            "clean_sheet_probability": training_history["clean_sheet_probability_default_rho"],
            "defensive_action_rate": training_history["dc_per_90"],
        },
        index=training_history.index,
    )
    for position in POSITIONS:
        bonus_features[f"position_{position}"] = (training_history["position"] == position).astype(
            float
        )
    bonus_model = BonusModel().fit(bonus_features, training_history["bonus"].astype(float))

    matches = build_match_level_frame(training_history)
    home_lambda = matches["team_xg_per_90"] * (
        matches["opponent_xga_per_90"] / matches["league_avg_xga_per_90"]
    )
    away_lambda = matches["opponent_xg_per_90"] * (
        matches["team_xga_per_90"] / matches["league_avg_xga_per_90"]
    )
    rho = fit_dixon_coles_rho(
        home_lambda, away_lambda, matches["team_h_score"], matches["team_a_score"]
    )

    save_conversion_rate = fit_save_conversion_rate(pd.Series(dtype=float), pd.Series(dtype=float))
    away_shot_multiplier = fit_away_shot_multiplier(pd.Series(dtype=float), pd.Series(dtype=float))

    dc_alpha_by_position = {}
    for position in (DEF, MID, FWD):
        subset = training_history[training_history["position"] == position]
        mu = subset["dc_per_90"] * (subset["minutes"] / 90.0)  # possession adjustment is neutral
        dc_alpha_by_position[position] = fit_overdispersion(subset["defensive_contribution"], mu)

    penalty_attempts = build_penalty_attempts_frame(training_history)
    penalty_rates_by_player, league_avg_penalty_rate = fit_penalty_conversion_rates(
        penalty_attempts
    )

    fitted_constants = FittedConstants(
        dixon_coles_rho=rho,
        save_conversion_rate=save_conversion_rate,
        away_shot_multiplier=away_shot_multiplier,
        dc_overdispersion_alpha=dc_alpha_by_position,
        penalty_conversion_rate_by_player=penalty_rates_by_player,
        league_avg_penalty_conversion_rate=league_avg_penalty_rate,
        shrinkage_k=SHRINKAGE_K,
    )
    return FittedEngineState(
        minutes_model=minutes_model, bonus_model=bonus_model, fitted_constants=fitted_constants
    )


def make_predict_fn(
    engineered: pd.DataFrame,
) -> Callable[[FittedEngineState, int], pd.DataFrame]:
    """Closure factory: ``backtest.harness.run_walk_forward``'s ``predict_fn`` signature is
    ``(fitted_state, gameweek) -> DataFrame`` with no access to the full engineered-features frame,
    so this captures it once and slices per gameweek on each call."""

    def predict_fn(fitted_state: FittedEngineState, gameweek: int) -> pd.DataFrame:
        players_gw = engineered[engineered["gameweek"] == gameweek]
        if players_gw.empty:
            return pd.DataFrame(columns=["player_id", "position", "gameweek", "expected_points"])
        return project_gameweek_pool(
            players_gw,
            gameweek,
            fitted_state.minutes_model,
            fitted_state.bonus_model,
            fitted_state.fitted_constants,
        )

    return predict_fn


# =================================================================================================
# D.3: simulation-based floor/ceiling/prob_big_haul (opt-in, run_backtest(run_simulation=True))
# =================================================================================================


def _build_team_match_inputs(
    team_players: pd.DataFrame, fitted_state: FittedEngineState
) -> TeamMatchInputs:
    """One team's roster for one fixture, translated into :mod:`engine.simulate`'s
    :class:`~engine.simulate.PlayerMatchInputs` (ENGINE_IMPROVEMENTS_2.md D.3) — the same
    already-fitted per-gameweek constants (``fitted_state.fitted_constants``) and adjusted-rate
    helper functions the point-estimate path (``engine.pipeline``) uses, called with
    ``expected_minutes=90.0`` to get the pure per-90 adjusted rate :mod:`engine.simulate` itself
    scales by each run's *drawn* minutes (not the point estimate's, since the whole point of
    simulating is that minutes genuinely vary run to run).
    """
    minutes_distributions = fitted_state.minutes_model.predict(team_players)
    players: list[PlayerMatchInputs] = []
    for (_, row), minutes_distribution in zip(
        team_players.iterrows(), minutes_distributions, strict=True
    ):
        player_id = int(row["player_id"])
        position = row["position"]
        adjusted_goal_rate = expected_non_penalty_goal_rate(
            row["npxg_per_90"], row["opponent_xga_per_90"], row["league_avg_xga_per_90"], 90.0
        )
        adjusted_assist_rate = expected_assist_rate(
            row["xa_per_90"], row["opponent_xga_per_90"], row["league_avg_xga_per_90"], 90.0
        )
        adjusted_dc_rate = (
            expected_defensive_action_rate(row["dc_per_90"], row["opponent_possession_share"], 90.0)
            if position in DEFENSIVE_CONTRIBUTION_THRESHOLD
            else 0.0
        )
        players.append(
            PlayerMatchInputs(
                player_id=player_id,
                position=position,
                minutes_distribution=minutes_distribution,
                adjusted_goal_rate_per_90=adjusted_goal_rate,
                adjusted_assist_rate_per_90=adjusted_assist_rate,
                is_penalty_taker=float(row.get("taker_share", 0.0)) > 0.0,
                penalty_conversion_rate=fitted_state.fitted_constants.penalty_conversion_rate_by_player.get(
                    player_id, fitted_state.fitted_constants.league_avg_penalty_conversion_rate
                ),
                adjusted_defensive_action_rate_per_90=adjusted_dc_rate,
                dc_overdispersion_alpha=fitted_state.fitted_constants.dc_overdispersion_alpha.get(
                    position, DEFAULT_OVERDISPERSION
                ),
                yellow_card_rate_per_90=float(row["yellow_card_rate_per_90"]),
                red_card_rate_per_90=float(row["red_card_rate_per_90"]),
            )
        )
    team_expected_penalties = (
        float(team_players["team_expected_penalties"].iloc[0])
        if "team_expected_penalties" in team_players.columns
        else 0.0
    )
    return TeamMatchInputs(players=players, team_expected_penalties=team_expected_penalties)


def simulate_gameweek_pool(
    players_gw: pd.DataFrame,
    fitted_state: FittedEngineState,
    n_runs: int = DEFAULT_N_RUNS,
    seed: int | None = None,
) -> pd.DataFrame:
    """Run :func:`engine.simulate.simulate_fixture` for every real fixture in one gameweek's pool
    (ENGINE_IMPROVEMENTS_2.md D.3) and return one row per player: ``floor``, ``ceiling``,
    ``prob_big_haul``, plus the simulation's own ``sim_mean``/``sim_median`` for comparison against
    the point-estimate ``expected_points``. Fixtures are found by grouping on
    ``(team, opponent_team_name)`` — each team processed exactly once — and a team whose opponent
    doesn't also appear in this gameweek's pool (a data gap, not expected in practice) is skipped
    rather than guessed at.
    """
    columns = ["player_id", "gameweek", "sim_mean", "sim_median", "floor", "ceiling", "prob_big_haul"]
    if players_gw.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, float]] = []
    processed_teams: set[str] = set()
    for team, group in players_gw.groupby("team"):
        if team in processed_teams:
            continue
        opponent = group["opponent_team_name"].iloc[0]
        opponent_group = players_gw[players_gw["team"] == opponent]
        if opponent_group.empty:
            continue
        processed_teams.add(team)
        processed_teams.add(opponent)

        is_home = bool(group["was_home"].iloc[0])
        home_group, away_group = (group, opponent_group) if is_home else (opponent_group, group)
        home_inputs = _build_team_match_inputs(home_group, fitted_state)
        away_inputs = _build_team_match_inputs(away_group, fitted_state)

        result = simulate_fixture(
            home_inputs,
            away_inputs,
            fitted_state.bonus_model,
            n_runs=n_runs,
            rho=fitted_state.fitted_constants.dixon_coles_rho,
            seed=seed,
        )
        gameweek = int(group["gameweek"].iloc[0])
        for player_id, summary in result.player_summaries.items():
            rows.append(
                {
                    "player_id": player_id,
                    "gameweek": gameweek,
                    "sim_mean": summary.mean,
                    "sim_median": summary.median,
                    "floor": summary.floor,
                    "ceiling": summary.ceiling,
                    "prob_big_haul": summary.prob_big_haul,
                }
            )
    return pd.DataFrame(rows, columns=columns)


def make_simulate_predict_fn(
    engineered: pd.DataFrame,
    n_runs: int = DEFAULT_N_RUNS,
    seed: int | None = None,
) -> Callable[[FittedEngineState, int], pd.DataFrame]:
    """Same closure-factory shape as :func:`make_predict_fn`, for the simulation path — lets
    ``backtest.harness.run_walk_forward`` drive :func:`simulate_gameweek_pool` gameweek by
    gameweek with the same no-leakage guarantee. This refits ``fitted_state`` independently of the
    point-estimate walk-forward run (duplicated work, since both call the same ``fit_fn``) — an
    accepted inefficiency given this path is opt-in only (``run_backtest(run_simulation=True)``),
    not the default.
    """

    def predict_fn(fitted_state: FittedEngineState, gameweek: int) -> pd.DataFrame:
        players_gw = engineered[engineered["gameweek"] == gameweek]
        return simulate_gameweek_pool(players_gw, fitted_state, n_runs=n_runs, seed=seed)

    return predict_fn


# =================================================================================================
# Scoring / gate
# =================================================================================================


@dataclass(frozen=True)
class CoverageReport:
    """Sample-coverage accounting for one backtest run (ENGINE_IMPROVEMENTS_2.md A.5) — Correction 3
    found a 19% scored-sample shrinkage across two passes that was invisible in the report; this
    makes that class of change visible on every run instead of requiring a fresh audit to catch it.
    """

    raw_outfield_rows: int
    distinct_player_gameweeks: int
    dgw_rows_collapsed: int
    engineered_rows: int
    rows_dropped_for_missing_features: int
    outfield_players: int
    crosswalk_matched_players: int
    crosswalk_match_rate: float
    points_excluded_share: float  # season points held by unmatched players with >450 minutes

    def summary(self) -> str:
        return "\n".join(
            [
                f"Raw outfield rows: {self.raw_outfield_rows} "
                f"({self.distinct_player_gameweeks} distinct player-gameweeks, "
                f"{self.dgw_rows_collapsed} double-gameweek rows collapsed)",
                f"Engineered rows after dropna: {self.engineered_rows} "
                f"({self.rows_dropped_for_missing_features} dropped for missing point-in-time "
                "features)",
                f"Crosswalk: {self.crosswalk_matched_players}/{self.outfield_players} outfield "
                f"players matched to Understat ({self.crosswalk_match_rate:.1%})",
                "Season points held by unmatched players with >450 minutes: "
                f"{self.points_excluded_share:.1%} of total outfield points",
            ]
        )


@dataclass(frozen=True)
class SeasonReport:
    accuracy: metrics.AccuracyReport
    bias_by_position: metrics.BiasReport
    top_n: metrics.TopNReport
    rank_correlation: metrics.RankCorrelationReport
    clean_sheet_calibration: metrics.CalibrationReport
    minutes_diagnostics: metrics.MinutesDiagnosticsReport
    minutes_played_calibration: metrics.CalibrationReport
    minutes_60_plus_calibration: metrics.CalibrationReport
    defensive_contribution_calibration: metrics.CalibrationReport
    mean_calibrations: dict[str, metrics.MeanCalibrationReport]
    baseline_results: dict[str, baselines.PairedBootstrapResult]
    definition_of_done: gate.DefinitionOfDoneReport
    coverage: CoverageReport | None = None
    captaincy: metrics.CaptaincyHitRateResult | None = None
    floor_ceiling_coverage: float | None = None
    big_haul_calibration: metrics.CalibrationReport | None = None

    def summary(self) -> str:
        lines = [
            f"Overall MAE: {self.accuracy.overall_mae:.4f}",
            f"Overall RMSE: {self.accuracy.overall_rmse:.4f}",
            "",
            "By position:",
            self.accuracy.by_position.to_string(index=False),
            "",
            "Top-N mean actual points:",
            self.top_n.by_n.to_string(index=False),
            "",
            f"Pooled Spearman (predicted vs actual, all rows): {self.rank_correlation.overall:.4f}",
            f"Starters-only Spearman (minutes > 0): "
            f"{self.rank_correlation.overall_starters_only:.4f}"
            if self.rank_correlation.overall_starters_only is not None
            else "Starters-only Spearman: not computed (no minutes_col given)",
            "",
            "Minutes-model diagnostics:",
            f"  zero-minute share: {self.minutes_diagnostics.zero_minute_share:.4f}",
            f"  mean expected_minutes on zero-minute rows: "
            f"{self.minutes_diagnostics.mean_expected_minutes_on_zero_rows:.2f}",
            f"  predicted-points mass on zero-minute rows: "
            f"{self.minutes_diagnostics.predicted_points_mass_on_zero_rows:.1f} "
            f"({self.minutes_diagnostics.predicted_points_mass_per_scored_row:.4f} / scored row)",
            f"  AUC 'played at all' (from 1 - p_zero): "
            f"{self.minutes_diagnostics.auc_played_at_all:.4f}",
            "",
            f"Clean-sheet MACE (gated, apples-to-apples): "
            f"{self.clean_sheet_calibration.mean_absolute_calibration_error:.4f}",
            f"Minutes 'played at all' MACE: "
            f"{self.minutes_played_calibration.mean_absolute_calibration_error:.4f}",
            f"Minutes 'played 60+' MACE: "
            f"{self.minutes_60_plus_calibration.mean_absolute_calibration_error:.4f}",
            f"Defensive-contribution threshold MACE: "
            f"{self.defensive_contribution_calibration.mean_absolute_calibration_error:.4f}",
            "",
            "Mean calibration (continuous components, informational — not gated):",
            *[
                f"  {name}: predicted={report.mean_predicted:.4f} actual={report.mean_actual:.4f} "
                f"relative_gap={report.relative_gap:.2%}"
                for name, report in self.mean_calibrations.items()
            ],
            "",
            "Baselines beaten (paired bootstrap, engine MAE vs baseline MAE):",
            *[
                f"  {name}: mean_diff={result.mean_diff:.4f} "
                f"CI=[{result.ci_low:.4f}, {result.ci_high:.4f}] beats={result.beats_baseline}"
                for name, result in self.baseline_results.items()
            ],
            "",
            self.definition_of_done.summary(),
        ]
        if self.captaincy is not None:
            lines = lines + [
                "",
                "Captaincy hit-rate (stand-in squad, see D.1 docstring — not the gate's own metric):",
                f"  raw hit-rate: {self.captaincy.raw_hit_rate:.4f} "
                f"(n={len(self.captaincy.per_gameweek)})",
                f"  'played as expected' hit-rate: {self.captaincy.played_as_expected_hit_rate:.4f}",
            ]
        if self.floor_ceiling_coverage is not None or self.big_haul_calibration is not None:
            lines = lines + [
                "",
                "Simulation layer (D.3 — floor/ceiling/prob_big_haul):",
            ]
            if self.floor_ceiling_coverage is not None:
                lines.append(
                    f"  floor/ceiling coverage (target ~80%, [P10, P90]): "
                    f"{self.floor_ceiling_coverage:.4f}"
                )
            if self.big_haul_calibration is not None:
                lines.append(
                    f"  prob_big_haul MACE: "
                    f"{self.big_haul_calibration.mean_absolute_calibration_error:.4f}"
                )
        if self.coverage is not None:
            lines = ["Sample coverage:", self.coverage.summary(), ""] + lines
        return "\n".join(lines)


def build_stand_in_squad_starting_xi(
    ground_truth: pd.DataFrame, squad_size: int = 13, starts_col: str = "starts"
) -> dict[int, set[int]]:
    """A fixed, model-independent stand-in "my team" for captaincy backtesting
    (ENGINE_IMPROVEMENTS_2.md D.1 / ENGINE_IMPROVEMENTS.md 3.1's own documented workaround — no
    real historical "my team" exists to backtest against): the ``squad_size`` highest
    total-season-minutes outfield players, selected **once from ground truth alone**, never from
    engine predictions, to avoid circularity. Each gameweek's eligible ("starting XI") set is
    restricted to whichever of those players actually started that gameweek — bench players were
    never really in contention for the armband (BUILD_PLAN 3.2) — via ``starts_col``.

    This is the exact caveat ENGINE_IMPROVEMENTS.md 3.1 flags: a stand-in squad, not a real
    manager's history, so the resulting hit-rate is illustrative rather than a claim about any
    specific real team's captaincy record.
    """
    totals = ground_truth.groupby("player_id")["minutes"].sum().nlargest(squad_size)
    squad_ids = set(totals.index)
    eligible = ground_truth[
        ground_truth["player_id"].isin(squad_ids) & (ground_truth[starts_col] > 0)
    ]
    return {
        int(gameweek): set(group["player_id"]) for gameweek, group in eligible.groupby("gameweek")
    }


def score_season(
    predictions: pd.DataFrame,
    ground_truth: pd.DataFrame,
    coverage: CoverageReport | None = None,
    player_rates: pd.DataFrame | None = None,
    simulation_predictions: pd.DataFrame | None = None,
) -> SeasonReport:
    """``player_rates``, if given (columns: ``player_id``, ``position``, ``gameweek``,
    ``npxg_per_90``, ``xa_per_90``, ``recent_minutes_ewma``), adds ``baselines.pure_xg_predictions``
    as a third baseline (ENGINE_IMPROVEMENTS_2.md D.2) — the gate's own summary claims "beats all
    baselines" while only two of the three named in BUILD_PLAN 3.3 were ever actually checked.

    ``simulation_predictions``, if given (columns: ``player_id``, ``gameweek``, ``floor``,
    ``ceiling``, ``prob_big_haul`` — see :func:`simulate_gameweek_pool`), scores the distributional
    outputs BUILD_PLAN 2.9's simulation layer produces but which nothing previously scored
    (ENGINE_IMPROVEMENTS_2.md D.3): floor/ceiling coverage against the realised outcome, and
    ``prob_big_haul`` calibration against ``total_points >= 10``.
    """
    accuracy = metrics.player_accuracy(predictions, ground_truth)
    bias = metrics.bias_by_group(predictions, ground_truth, group_col="position")
    top_n = metrics.top_n_mean_actual(predictions, ground_truth)
    rank_corr = metrics.rank_correlation(
        predictions, ground_truth, group_col="position", minutes_col="minutes"
    )
    minutes_diagnostics = metrics.minutes_model_diagnostics(predictions, ground_truth)

    gated_actual = ground_truth.copy()
    gated_actual["gated_clean_sheet"] = (
        (ground_truth["clean_sheets"] > 0) & (ground_truth["minutes"] >= 60)
    ).astype(float)
    calibration_rows = predictions.merge(
        gated_actual[["player_id", "gameweek", "gated_clean_sheet"]], on=["player_id", "gameweek"]
    )
    clean_sheet_calibration = metrics.component_calibration(
        calibration_rows["player_clean_sheet_probability"], calibration_rows["gated_clean_sheet"]
    )

    # --- A.4: per-component calibration, not just clean sheets ---------------------------------
    minutes_rows = predictions.merge(
        ground_truth[["player_id", "gameweek", "minutes"]], on=["player_id", "gameweek"]
    )
    minutes_played_calibration = metrics.component_calibration(
        1.0 - minutes_rows["p_zero"], (minutes_rows["minutes"] > 0).astype(float)
    )
    minutes_60_plus_calibration = metrics.component_calibration(
        minutes_rows["p_60_plus"], (minutes_rows["minutes"] >= 60).astype(float)
    )

    dc_rows = predictions.merge(
        ground_truth[["player_id", "gameweek", "defensive_contribution"]].rename(
            columns={"defensive_contribution": "actual_defensive_contribution"}
        ),
        on=["player_id", "gameweek"],
    )
    dc_rows = dc_rows[dc_rows["p_clears_threshold"].notna()]
    dc_threshold = dc_rows["position"].map(DEFENSIVE_CONTRIBUTION_THRESHOLD)
    dc_actual_clears = (dc_rows["actual_defensive_contribution"] >= dc_threshold).astype(float)
    defensive_contribution_calibration = metrics.component_calibration(
        dc_rows["p_clears_threshold"], dc_actual_clears
    )

    # predictions already carries its own "assists"/"bonus" columns (points, not raw counts), so
    # the ground-truth actuals need distinct names to avoid a silent _x/_y suffix collision.
    mean_rows = predictions.merge(
        ground_truth[["player_id", "gameweek", "goals_scored", "assists", "bonus"]].rename(
            columns={
                "goals_scored": "actual_goals_scored",
                "assists": "actual_assists",
                "bonus": "actual_bonus",
            }
        ),
        on=["player_id", "gameweek"],
    )
    mean_calibrations = {
        "goals": metrics.mean_calibration(
            mean_rows["expected_goals"], mean_rows["actual_goals_scored"]
        ),
        "assists": metrics.mean_calibration(
            mean_rows["expected_assists"], mean_rows["actual_assists"]
        ),
        "bonus": metrics.mean_calibration(mean_rows["expected_bonus"], mean_rows["actual_bonus"]),
    }

    engine_actuals = predictions.merge(
        ground_truth[["player_id", "gameweek", "total_points"]], on=["player_id", "gameweek"]
    )
    median_value = baselines.training_median(ground_truth)
    constant_preds = baselines.constant_predictions(
        predictions[["player_id", "position", "gameweek"]], median_value
    )
    naive_preds = baselines.naive_form_predictions(ground_truth)

    baseline_variants = [("constant_median", constant_preds), ("naive_form", naive_preds)]
    if player_rates is not None:
        # D.2: the gate's own summary claims "beats all baselines" — pure_xg is the third baseline
        # BUILD_PLAN 3.3 names but that was never actually wired in.
        baseline_variants.append(("pure_xg", baselines.pure_xg_predictions(player_rates)))

    baseline_results: dict[str, baselines.PairedBootstrapResult] = {}
    for name, baseline_preds in baseline_variants:
        joint = engine_actuals.merge(
            baseline_preds.merge(
                ground_truth[["player_id", "gameweek", "total_points"]],
                on=["player_id", "gameweek"],
            ),
            on=["player_id", "gameweek"],
            suffixes=("_engine", "_baseline"),
        )
        if len(joint) < 2:
            continue
        engine_err = (
            (joint["expected_points_engine"] - joint["total_points_engine"]).abs().to_numpy()
        )
        baseline_err = (
            (joint["expected_points_baseline"] - joint["total_points_baseline"]).abs().to_numpy()
        )
        # D.2: block by gameweek -- players in the same gameweek share shocks (a red card, a 5-0
        # drubbing, a rested squad), so an i.i.d.-by-row bootstrap understates uncertainty.
        baseline_results[name] = baselines.paired_bootstrap_test(
            engine_err, baseline_err, block_by=joint["gameweek"].to_numpy()
        )

    definition_of_done = gate.evaluate_definition_of_done(
        baseline_results=baseline_results,
        bias_reports={"position": bias},
        calibration_reports={
            "clean_sheet": clean_sheet_calibration,
            "minutes_played_at_all": minutes_played_calibration,
            "minutes_60_plus": minutes_60_plus_calibration,
            "defensive_contribution": defensive_contribution_calibration,
        },
        predictions_logged=True,
        trusted_by_user=False,
    )

    # --- D.1: captaincy hit-rate, via a stand-in squad (see build_stand_in_squad_starting_xi) ---
    captaincy = None
    if "starts" in ground_truth.columns:
        starting_xi_by_gameweek = build_stand_in_squad_starting_xi(ground_truth)
        if starting_xi_by_gameweek:
            try:
                captaincy = metrics.captaincy_hit_rate(
                    predictions, ground_truth, starting_xi_by_gameweek
                )
            except ValueError:
                captaincy = None  # too few gameweeks with overlapping eligible predictions/actuals

    # --- D.3: floor/ceiling coverage + prob_big_haul calibration, if a simulation run was given ---
    floor_ceiling_cov = None
    big_haul_calibration = None
    if simulation_predictions is not None:
        sim_merged = simulation_predictions.merge(
            ground_truth[["player_id", "gameweek", "total_points"]], on=["player_id", "gameweek"]
        )
        if len(sim_merged) >= 2:
            floor_ceiling_cov = metrics.floor_ceiling_coverage(
                sim_merged["floor"], sim_merged["ceiling"], sim_merged["total_points"]
            )
            big_haul_calibration = metrics.component_calibration(
                sim_merged["prob_big_haul"], (sim_merged["total_points"] >= 10).astype(float)
            )

    return SeasonReport(
        accuracy=accuracy,
        bias_by_position=bias,
        top_n=top_n,
        rank_correlation=rank_corr,
        clean_sheet_calibration=clean_sheet_calibration,
        minutes_diagnostics=minutes_diagnostics,
        minutes_played_calibration=minutes_played_calibration,
        minutes_60_plus_calibration=minutes_60_plus_calibration,
        defensive_contribution_calibration=defensive_contribution_calibration,
        mean_calibrations=mean_calibrations,
        baseline_results=baseline_results,
        definition_of_done=definition_of_done,
        coverage=coverage,
        captaincy=captaincy,
        floor_ceiling_coverage=floor_ceiling_cov,
        big_haul_calibration=big_haul_calibration,
    )


# =================================================================================================
# Top-level orchestration + CLI
# =================================================================================================


def compute_coverage_report(
    merged_gw: pd.DataFrame, crosswalk: list[CrosswalkEntry], engineered: pd.DataFrame
) -> CoverageReport:
    """Sample-coverage accounting (ENGINE_IMPROVEMENTS_2.md A.5) — reads the collapse/dropna counts
    ``engineer_features`` attaches to ``engineered.attrs`` immediately after it returns them, and
    the crosswalk match rate against real season minutes/points (C.1's own acceptance criterion)."""
    outfield = merged_gw[merged_gw["position"].isin([DEF, MID, FWD])]
    matched_ids = {entry.fpl_id for entry in crosswalk}
    per_player_minutes = outfield.groupby("element")["minutes"].sum()
    per_player_points = outfield.groupby("element")["total_points"].sum()
    total_season_points = float(per_player_points.sum())
    matched_mask = pd.Series(per_player_minutes.index.isin(matched_ids), index=per_player_minutes.index)
    unmatched_significant = per_player_minutes[(~matched_mask) & (per_player_minutes > 450)].index
    points_excluded = float(per_player_points.reindex(unmatched_significant).fillna(0.0).sum())
    points_excluded_share = points_excluded / total_season_points if total_season_points else float("nan")

    return CoverageReport(
        raw_outfield_rows=len(outfield),
        distinct_player_gameweeks=outfield.groupby(["element", "GW"]).ngroups,
        dgw_rows_collapsed=int(engineered.attrs.get("n_dgw_rows_collapsed", 0)),
        engineered_rows=len(engineered),
        rows_dropped_for_missing_features=int(
            engineered.attrs.get("n_rows_dropped_for_missing_features", 0)
        ),
        outfield_players=len(per_player_minutes),
        crosswalk_matched_players=int(matched_mask.sum()),
        crosswalk_match_rate=float(matched_mask.mean()) if len(per_player_minutes) else float("nan"),
        points_excluded_share=points_excluded_share,
    )


def run_backtest(
    season_start_year: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_training_gameweeks: int = 3,
    refresh: bool = False,
    min_crosswalk_minutes_share: float | None = 0.85,
    run_simulation: bool = False,
    simulation_n_runs: int = 200,
) -> tuple[pd.DataFrame, SeasonReport]:
    """``min_crosswalk_minutes_share``, if given, enforces
    :func:`engine.data.crosswalk.assert_matched_share` against real season minutes
    (ENGINE_IMPROVEMENTS_2.md C.1) — a coverage floor independent of ``build_season_crosswalk``'s
    own non-strict per-player tolerance, which only catches an unmatched Understat player and says
    nothing about an FPL player who was simply never matched at all. Pass ``None`` to skip the
    check (e.g. while iterating on a season where coverage is a known, accepted work in progress).

    ``run_simulation``, if True, additionally drives :func:`simulate_gameweek_pool` through its own
    walk-forward pass (ENGINE_IMPROVEMENTS_2.md D.3) to score floor/ceiling coverage and
    ``prob_big_haul`` calibration — off by default since it refits the engine a second time and
    runs a Monte Carlo simulation per fixture per gameweek, meaningfully more expensive than the
    point-estimate run alone. ``simulation_n_runs`` (default 200, well below
    :data:`engine.simulate.DEFAULT_N_RUNS`'s 2000) trades simulation precision for tractability
    across a full season's fixtures.
    """
    with httpx.Client(timeout=30.0) as http_client, UnderstatClient() as understat:
        merged_gw = fetch_vaastav_merged_gw(season_start_year, cache_dir, http_client, refresh)
        teams = fetch_vaastav_teams(season_start_year, cache_dir, http_client, refresh)
        league_data = fetch_understat_league_data_raw(
            season_start_year, cache_dir, understat, refresh
        )
        team_histories = build_team_rate_histories(
            league_data_to_dataframes(league_data)["teams_history"]
        )
        crosswalk = build_season_crosswalk(
            season_start_year, league_data, cache_dir, http_client, refresh
        )
        player_histories = fetch_understat_player_histories(
            crosswalk, season_start_year, cache_dir, understat, refresh
        )

    engineered = engineer_features(merged_gw, teams, team_histories, player_histories)
    coverage = compute_coverage_report(merged_gw, crosswalk, engineered)

    if min_crosswalk_minutes_share is not None:
        outfield = merged_gw[merged_gw["position"].isin([DEF, MID, FWD])]
        per_player_minutes = outfield.groupby("element")["minutes"].sum().to_dict()
        matched_ids = {entry.fpl_id for entry in crosswalk}
        assert_matched_share(
            per_player_minutes, matched_ids, min_share=min_crosswalk_minutes_share, label="season minutes"
        )

    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(
        gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=min_training_gameweeks
    )

    ground_truth = engineered[
        [
            "player_id",
            "gameweek",
            "position",
            "total_points",
            "minutes",
            "clean_sheets",
            "defensive_contribution",
            "goals_scored",
            "assists",
            "bonus",
            "starts",
        ]
    ]
    player_rates = engineered[
        ["player_id", "position", "gameweek", "npxg_per_90", "xa_per_90", "recent_minutes_ewma"]
    ]

    simulation_predictions = None
    if run_simulation:
        simulate_fn = make_simulate_predict_fn(engineered, n_runs=simulation_n_runs)
        sim_result = run_walk_forward(
            gameweeks,
            engineered,
            fit_fn,
            simulate_fn,
            min_training_gameweeks=min_training_gameweeks,
        )
        simulation_predictions = sim_result.predictions

    report = score_season(
        result.predictions,
        ground_truth,
        coverage=coverage,
        player_rates=player_rates,
        simulation_predictions=simulation_predictions,
    )
    return result.predictions, report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--season", type=int, default=2025, help="Season start year, e.g. 2025 for 2025/26"
    )
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--min-training-gameweeks", type=int, default=3)
    parser.add_argument("--refresh", action="store_true", help="Re-fetch even if cached")
    parser.add_argument("--report-path", type=Path, default=None)
    parser.add_argument(
        "--min-crosswalk-minutes-share",
        type=float,
        default=0.85,
        help="Fail the run if the ID crosswalk covers less than this share of season minutes "
        "(ENGINE_IMPROVEMENTS_2.md C.1); pass a negative number to skip the check",
    )
    args = parser.parse_args(argv)

    min_share = args.min_crosswalk_minutes_share if args.min_crosswalk_minutes_share >= 0 else None
    predictions, report = run_backtest(
        args.season,
        args.cache_dir,
        args.min_training_gameweeks,
        args.refresh,
        min_crosswalk_minutes_share=min_share,
    )
    summary = report.summary()
    print(summary)
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(summary)
        predictions.to_parquet(args.report_path.with_suffix(".parquet"), index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
