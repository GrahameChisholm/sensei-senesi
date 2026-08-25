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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd

from backtest import baselines, gate, metrics
from backtest.harness import run_walk_forward
from engine.data.crosswalk import (
    MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON,
    CrosswalkEntry,
    assert_matched_share,
    build_crosswalk,
    fetch_fpl_id_list,
    fetch_fpl_web_names,
    understat_players_from_league_data,
)
from engine.data.understat_client import (
    EARLIEST_SEASON,
    UnderstatClient,
    league_data_to_dataframes,
    player_data_to_dataframe,
)
from engine.models.assists import expected_assist_rate, fit_assist_share_of_team_xg
from engine.models.bonus import BonusModel
from engine.models.clean_sheets import (
    clean_sheet_probability,
    fit_dixon_coles_rho,
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
from engine.models.minutes import MinutesDistribution, MinutesModel, encode_status
from engine.models.saves import (
    fit_away_shot_multiplier,
    fit_save_conversion_rate,
    project_saves_from_own_rate,
)
from engine.pipeline import FittedConstants, project_gameweek_pool
from engine.rates import (
    effective_sample_minutes,
    effective_sample_minutes_asof,
    ewma_rate_asof,
    latest_ewma_rate,
    shrink_toward_prior,
)
from engine.scoring import DEF, DEFENSIVE_CONTRIBUTION_THRESHOLD, FWD, GK, MID, POSITIONS
from engine.simulate import (
    DEFAULT_N_RUNS,
    PlayerMatchInputs,
    TeamMatchInputs,
    simulate_fixture,
)

VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_CACHE_DIR = Path("data_store/season_cache")
DEFAULT_HALFLIFE = 10.0

# ENGINE_IMPROVEMENTS_4.md: goals and assists used to share one `SHRINKAGE_K` fed to both
# `project_goals`/`project_assists`'s thin-sample-rate shrinkage (ENGINE_IMPROVEMENTS_2.md B.2).
# Splitting them (`engine.pipeline.FittedConstants.goals_shrinkage_k`/`assists_shrinkage_k`) and
# re-sweeping each independently on the real 2025/26 walk-forward, scored on PLAYED rows only
# (minutes > 0 -- the previous all-rows sweeps were dominated by the minutes model, not this
# constant; see `score_season`'s `mean_calibrations_played`), found the two components disagree
# sharply:
#
# Assists' own-rate (Understat xA/90) played-only relative gap is -22.56% at k=0 (no shrinkage) and
# initially appeared to improve monotonically toward zero as k rose with the FLAT
# DEFAULT_ASSIST_SHARE_OF_TEAM_XG prior (0.12 for every position) -- but that sweep also drove a
# real, severe FWD over-prediction bias (mean_residual +0.259 at k=1000, exceeding the gate's
# effect-size floor): the flat share is systematically too high for FWDs relative to their real
# assist-per-team-xG rate, and heavy shrinkage just pulls every FWD toward that wrong number.
#
# Fix: `engine.pipeline.FittedConstants.assist_share_of_team_xg_by_position`
# (`engine.models.assists.fit_assist_share_of_team_xg`) replaces the flat constant with one fit
# per position from real data, closing the FWD-bias regression across the entire k range re-swept
# (FWD residual stays under +0.21, never severe, from k=0 to k=1e9). But re-sweeping the AGGREGATE
# calibration with the position-aware prior found something more fundamental: it asymptotes at
# -13.12% even at k=1e9 (predictions ≡ the fitted position prior, zero individual signal) --
# proof this is not a shrinkage-strength problem at all. The multiplicative
# `team_xg_per_90 * assist_share` functional form itself has a residual out-of-sample walk-forward
# miscalibration shrinkage cannot reach, however hard applied. The <5% gate target is genuinely
# unreachable this way; fixing it for real needs a different functional form or additional
# features, out of scope for a shrinkage-constant pass. Real sweeps, holding goals_k=20 fixed
# throughout (confirms the split is genuinely independent: goals% never moves as assists_k varies):
#
#     assists_k   0       90      200     500     1000    2000    4000    1e9
#     assists%    -22.56  -22.23  -21.53  -20.01  -18.40  -16.72  -15.32  -13.12
#     MAE         1.5681  1.5672  1.5677  1.5688  1.5700  1.5715  1.5728  1.5750
#     FWD severe  False   False   False   False   False   False   False   False
#
# Since the calibration target cannot be reached regardless of k, shrinkage's only real remaining
# job here is what it was originally for -- taming thin-sample outliers -- so k is chosen to
# minimize MAE rather than chase an unreachable calibration number: k=90 is the empirical MAE
# minimum in the swept range (1.5672, better than even k=0's 1.5681), with negligible calibration
# movement either way. This is coincidentally close to the pre-split shared value, but for an
# entirely different, now-verified reason.
#
# Goals shows no such pathology -- its own-rate calibration genuinely improves with moderate
# shrinkage and degrades again past its optimum (over-shrinking toward a coarser team-level
# proxy), so a single shared constant was forcing both components onto the wrong side of one or
# the other's optimum regardless of where it was set.
#
# Goals' own optimum, swept independently (unaffected by the assists split): coarse pass over
# {0, 30, 60, 90, 150, 300} found the minimum near 30 (played relative-gap 5.73%, MAE 1.5683), a
# finer pass over {10, 20, 25, 30, 35, 45} found it flatter and lower still at k=20 (5.71%, MAE
# 1.5682, RMSE 2.4408):
#
#     k        10      20      25      30      35      45
#     played%  +5.76   +5.71   +5.72   +5.73   +5.75   +5.81
#     MAE      1.5687  1.5682  1.5682  1.5683  1.5684  1.5688
#
# Old shared-k=90 comment, superseded — kept for provenance since that sweep predates both the
# played-only objective and the goals/assists split, so isn't directly comparable: "Re-swept over
# {0, 90, 180, 300, 450, 900} on the real 2025/26 walk-forward with every other Tier A-D change
# already landed: k=90 dominated on every metric (MAE 1.5836 vs 180's 1.5876, top-10 mean actual
# 5.21 vs 5.20, pooled rho 0.6343 vs 0.6343, max single-gameweek projection 7.36 vs 7.23) while k=0
# (disabled) let a single thin-sample outlier reach 24.8 points in one gameweek."
#
# Revisit only with a proper nested-CV walk-forward search; treated as fixed, evidence-based
# hyperparameters until then, not re-derived every gameweek.
GOALS_SHRINKAGE_K = 20.0
ASSISTS_SHRINKAGE_K = 90.0

# Card/own-goal thin-sample-rate shrinkage strength, in effective-minutes units
# (ENGINE_IMPROVEMENTS_3.md A.3) — a real point-in-time pull found an unshrunk red-card rate of
# 22.9 per 90 minutes sustained for six gameweeks off a single dismissal in a 3-minute cameo, the
# same defect B.2 fixed for goals/assists, never applied to cards. Yellows are common enough
# (median rate 0.09/90 across the 2025/26 archive) that a moderate prior weight is enough; reds and
# own goals are rare enough — and a single observation is different enough in kind from a real rate
# — that they lean almost entirely on the prior until a real sample accumulates. Initial
# evidence-based values, not yet swept the way SHRINKAGE_K/TEAM_RATE_SHRINKAGE_K were; revisit with
# a proper walk-forward sweep.
YELLOW_CARD_SHRINKAGE_K = 200.0
RED_CARD_SHRINKAGE_K = 1000.0
OWN_GOAL_SHRINKAGE_K = 500.0

# ENGINE_IMPROVEMENTS_3.md D.1: shrinkage strength (in effective vaastav-minutes) for the
# goalkeeper own-rate saves fallback. Saves are far more frequent than cards (a keeper faces
# several shots most matches), so a keeper's own EWMA carries real signal after just a couple of
# matches — a much smaller prior weight than the card constants above is appropriate. Initial
# evidence-based value, not yet swept.
SAVE_RATE_SHRINKAGE_K = 90.0

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
    "fetch_understat_multi_season_league_data",
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
    "UnmatchedSignificantPlayer",
    "compute_coverage_report",
    "build_stand_in_squad_starting_xi",
    "score_season",
    "SeasonBacktestData",
    "run_backtest",
    "MultiSeasonReport",
    "run_multi_season_backtest",
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


# ENGINE_IMPROVEMENTS_3.md A.4: team rates previously came from `season_start_year` alone,
# reintroducing the exact cold-start problem C.3 fixed for player rates (`engine/rates.py`'s own
# docstring: pass "a single chronologically-sorted history spanning prior seasons into the current
# one" so the EWMA decay handles the blend with no separate hand-coded rule for early gameweeks).
# 3 prior seasons is a bounded, cheap default (one extra Understat request per season, each cached
# independently) — enough for the halflife=10-match EWMA to have real history behind it by GW1
# without fetching the entire Understat archive back to 2014/15 for a rate that decays out within
# a season anyway.
N_PRIOR_SEASONS_FOR_TEAM_RATES = 3


def fetch_understat_multi_season_league_data(
    season_start_year: int,
    cache_dir: Path,
    understat: UnderstatClient,
    refresh: bool = False,
    n_prior_seasons: int = N_PRIOR_SEASONS_FOR_TEAM_RATES,
) -> dict[int, dict[str, Any]]:
    """``season_start_year``'s own league data plus up to ``n_prior_seasons`` before it
    (ENGINE_IMPROVEMENTS_3.md A.4), each fetched/cached independently via
    :func:`fetch_understat_league_data_raw`. A team promoted partway through this window simply
    has no data for the seasons before it was in the league — Understat's ``get_league_data`` only
    ever returns that season's actual EPL teams, so no special-casing is needed. Returns
    ``{season: league_data, ...}`` for whichever seasons are at or after
    :data:`engine.data.understat_client.EARLIEST_SEASON`.
    """
    seasons = [
        s
        for s in range(season_start_year - n_prior_seasons, season_start_year + 1)
        if s >= EARLIEST_SEASON
    ]
    return {
        season: fetch_understat_league_data_raw(season, cache_dir, understat, refresh)
        for season in seasons
    }


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
    # This season's own overlay slice, not the flat live-ingestion default — fpl_id is only
    # meaningful within the one season it was hand-verified against (see crosswalk.py's own
    # docstring on why the overlay is season-keyed).
    overlay = MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON.get(season_start_year, {})
    entries = build_crosswalk(
        understat_players,
        fpl_id_by_name,
        overlay=overlay,
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

    A6: a real live pull against Understat's pre-season endpoint found ``getLeagueData`` returns
    completely empty ``teams``/``players`` before a season's first match — a totally columnless
    ``teams_history`` (no ``"team_title"`` column at all), which ``.groupby("team_title")`` raises
    a bare ``KeyError`` on rather than degrading. An empty input is a real, expected state (a new
    season that hasn't started, or engine.data.live_adapter's own cold-start fix falling back to
    this alone when no prior-season data was supplied either) — it should produce an empty
    ``dict``, exactly what every downstream ``.get(team, empty)``-style lookup already handles,
    not crash the whole feature-engineering pipeline.
    """
    if teams_history.empty:
        return {}
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
    row in the sample (no prior appearance to measure from).

    T-E: a synthesized target row's own ``minutes`` is an unknown-outcome placeholder (0), never a
    real appearance, so it must never update ``last_appearance`` for a later row in the same
    player's horizon to read as real history, see this module's own T-E comment in
    ``engineer_features`` for the full horizon-decay mechanism this guards against. A ``gw`` with
    no ``is_synthesized_target`` column (a caller outside ``engineer_features``, which always adds
    it first) is treated as entirely real history, matching this function's behaviour before T-E.

    ENGINE_IMPROVEMENTS_5.md Tier 1.4 completes that guard: the value emitted for every synthesized
    row in one horizon is frozen at the first such row's value, rather than growing with elapsed
    calendar time. See the inline comment in ``_compute`` for the decay this removes.
    """
    if "is_synthesized_target" not in gw.columns:
        gw = gw.assign(is_synthesized_target=False)

    def _compute(g: pd.DataFrame) -> pd.Series:
        last_appearance: pd.Timestamp | None = None
        values = []
        first_synthesized_value: float | None = None
        for kickoff, minutes, is_synthesized in zip(
            g["kickoff_time"], g["minutes"], g["is_synthesized_target"], strict=True
        ):
            value = default_days if last_appearance is None else (kickoff - last_appearance).days
            # ENGINE_IMPROVEMENTS_5.md Tier 1.4: hold this feature flat across a multi-gameweek
            # horizon. T-E already stopped a synthesized row's placeholder minutes from *resetting*
            # the clock, but the clock still ran: with no appearance ever recorded for a
            # not-yet-played gameweek, the gap grows by ~7 days per horizon step, and the minutes
            # model reads a growing gap as an injury signal. That produced a systematic decay in
            # P(60+) the further out the horizon looked (0.292, 0.267, 0.252 across a three-gameweek
            # horizon on the real 2026-27 GW1 build) with no footballing reason for playing time to
            # fall. Freezing at the first horizon gameweek's value encodes the right assumption:
            # when projecting GW3 we are asking "if the season proceeds normally", not "if this
            # player has by then been absent for three more weeks".
            if is_synthesized:
                if first_synthesized_value is None:
                    first_synthesized_value = value
                value = first_synthesized_value
            values.append(value)
            if not is_synthesized and minutes > 0:
                last_appearance = kickoff
        return pd.Series(values, dtype=float)

    return _per_player_series(gw, _compute)


def compute_zero_minute_streak_length(gw: pd.DataFrame) -> pd.Series:
    """Consecutive prior gameweeks (strictly before this one) with zero minutes — directly
    separates "deep squad / unavailable" from "rotation risk" (ENGINE_IMPROVEMENTS.md 1.1).

    T-E: a synthesized target row's ``minutes`` placeholder (0) is not a real zero-minute
    gameweek, so it must not extend the streak that a later target row in the same horizon reads,
    see ``engineer_features``' own T-E comment for the full horizon-decay mechanism this guards
    against. The streak value emitted for a synthesized row itself is still the real streak as of
    that gameweek, just never advanced by that row's own fake outcome. A ``gw`` with no
    ``is_synthesized_target`` column (a caller outside ``engineer_features``, which always adds it
    first) is treated as entirely real history, matching this function's behaviour before T-E.
    """
    if "is_synthesized_target" not in gw.columns:
        gw = gw.assign(is_synthesized_target=False)

    def _compute(g: pd.DataFrame) -> pd.Series:
        streak = 0
        values = []
        for minutes, is_synthesized in zip(g["minutes"], g["is_synthesized_target"], strict=True):
            values.append(float(streak))
            if not is_synthesized:
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
        g.groupby(["player_id", "gameweek"], as_index=False, sort=False)
        .agg(agg)
        .reset_index(drop=True)
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


def _team_prior_match_count(team_history: pd.DataFrame, before: pd.Timestamp) -> int:
    """Matches strictly before ``before`` — the shrinkage weight for
    :func:`_team_rate_asof_shrunk` (ENGINE_IMPROVEMENTS_3.md A.1). A match count, not
    :func:`engine.rates.effective_sample_minutes`, since a team's own "sample size" for this
    purpose is naturally in matches, not minutes (unlike a player, who can appear for a handful of
    minutes and legitimately deserve a small weight)."""
    if team_history.empty:
        return 0
    return int((team_history["date"] < before).sum())


# ENGINE_IMPROVEMENTS_3.md A.1: shrinkage strength (in matches) for a team's own point-in-time
# xG/xGA rate toward the point-in-time league average. Selected via an end-to-end sweep over
# {2, 4, 8} team-fixtures matches against real 2025/26 clean-sheet outcomes — every metric (MACE,
# Brier, share of team-fixtures projected above 50%) was best at k=4; see the document's Tier A.1
# evidence table. Revisit with a proper walk-forward search across seasons, same caveat as
# SHRINKAGE_K.
TEAM_RATE_SHRINKAGE_K = 4.0

# Below this many league-wide prior matches, a venue multiplier can't be trusted — hold it at 1.0
# (venue-neutral) rather than fit one off a handful of games (only ever binds in the season's very
# first gameweek or two).
_MIN_MATCHES_FOR_VENUE_MULTIPLIER = 40


def _team_rate_asof_shrunk(
    team_history: pd.DataFrame,
    stat_col: str,
    before: pd.Timestamp,
    league_avg: float,
    shrinkage_k: float = TEAM_RATE_SHRINKAGE_K,
) -> float:
    """Point-in-time per-90 EWMA rate, shrunk toward ``league_avg`` (the same gameweek's own
    point-in-time league-average rate) by the team's own prior match count
    (ENGINE_IMPROVEMENTS_3.md A.1). Replaces the previous per-team home/away split
    (``_team_rate_asof_venue_split``, ENGINE_IMPROVEMENTS_2.md D.5), which measurably made
    clean-sheet calibration *worse than predicting the league base rate* (Brier 0.1895 vs a
    constant-base-rate Brier of 0.1872): splitting by venue halves the effective sample behind
    every team rate, and the home/away effect (~0.33 xGA) is close in size to the per-team-match
    noise (~0.87 std) — ~10 same-venue matches per team is too thin to estimate a team-specific
    venue split, even though the league-wide venue effect itself is real (see
    :func:`_league_venue_multipliers`, applied afterward in :func:`build_fixture_rate_frame`).

    A newly-promoted club has zero prior top-flight matches, so its own raw rate is NaN with
    zero weight. :func:`shrink_toward_prior` already returns the prior outright in that case, so
    the result here is the full league-average rate rather than a missing value, closing a gap
    where an established team's very first fixture against a debutant club would otherwise get
    a NaN opponent rate and be dropped entirely by the required-columns dropna downstream.
    """
    raw = _team_rate_asof(team_history, stat_col, before)
    if pd.isna(league_avg):
        return raw
    n_prior = _team_prior_match_count(team_history, before)
    return shrink_toward_prior(raw, float(n_prior), league_avg, shrinkage_k)


def _league_venue_multipliers(
    team_histories: dict[str, pd.DataFrame], before: pd.Timestamp
) -> tuple[float, float]:
    """Point-in-time, LEAGUE-WIDE home/away multiplier for xG and xGA (ENGINE_IMPROVEMENTS_3.md
    A.1) — a single pair of numbers fit across every team's matches strictly before ``before``,
    replacing the previous per-team venue split. ``xg_mult``/``xga_mult`` are each
    ``home_mean / overall_mean`` for that stat; by construction the away multiplier is
    ``2 - mult`` (the overall mean is the average of the home and away means), which is how
    :func:`build_fixture_rate_frame` applies this without a separate away-side computation. Falls
    back to ``(1.0, 1.0)`` — venue-neutral — when fewer than
    :data:`_MIN_MATCHES_FOR_VENUE_MULTIPLIER` league-wide matches are available yet (only binds in
    the season's first gameweek or two).
    """
    if not team_histories:
        return 1.0, 1.0
    all_matches = pd.concat(team_histories.values(), ignore_index=True)
    prior = all_matches[all_matches["date"] < before]
    if len(prior) < _MIN_MATCHES_FOR_VENUE_MULTIPLIER:
        return 1.0, 1.0
    home = prior[prior["is_home"]]
    overall_xg = float(prior["xG"].mean())
    overall_xga = float(prior["xGA"].mean())
    if home.empty or not overall_xg or not overall_xga:
        return 1.0, 1.0
    xg_mult = float(home["xG"].mean() / overall_xg)
    xga_mult = float(home["xGA"].mean() / overall_xga)
    return xg_mult, xga_mult


# B1: how hard a team's own venue multiplier is pulled toward the league-wide one, in units of that
# team's prior home matches. Swept on the real 2025/26 walk-forward (team-level clean-sheet MACE /
# Brier / overall MAE):
#
#     k       1e9*     60      40      20      10       5       0
#     MACE    0.0396  0.0353  0.0340  0.0322  0.0284  0.0248  0.0219
#     Brier   0.1763  0.1762  0.1762  0.1762  0.1762  0.1763  0.1765
#     MAE     1.5702  1.5704  1.5704  1.5706  1.5707  1.5709  1.5712
#     (* k -> infinity is the league-only multiplier, i.e. A.1's shipped behaviour)
#
# MACE falls monotonically as the estimate becomes more team-specific, while Brier is flat to
# k=10 and MAE degrades by 0.0005 across the whole range. k=10 clears the < 0.03 target with
# margin at no Brier cost, which is why it is preferred over the unshrunk k=0 that scores best on
# MACE alone.
#
# This does NOT contradict A.1's finding that per-team venue splits are harmful, and the
# distinction matters: A.1 split the *rate itself* by venue, halving the sample behind every team
# rate. Here the rate keeps its full sample and only the venue *adjustment* on top of it is
# team-specific — so the quantity being estimated from ~10 home matches is a multiplier near 1.0,
# not a rate from scratch, and shrinkage has something stable to pull toward.
TEAM_VENUE_SHRINKAGE_K = 10.0


def _team_venue_multipliers(
    team_history: pd.DataFrame,
    before: pd.Timestamp,
    league_multipliers: tuple[float, float],
    shrinkage_k: float = TEAM_VENUE_SHRINKAGE_K,
) -> tuple[float, float]:
    """This team's own home/away multiplier for xG and xGA, shrunk toward the league-wide pair.

    A.1 reverted the per-team venue split and left a single league-wide multiplier applied to every
    team. That fixed the Brier regression but leaves real signal on the table: home advantage
    genuinely differs by team, and the reason the previous attempt failed was thin-sample variance,
    not the absence of an effect. Shrinking each team's own multiplier toward the league one by its
    prior home-match count is the same empirical-Bayes treatment
    :func:`_team_rate_asof_shrunk` already applies to the rates themselves — the level of
    aggregation A.1's own closing note identified as the missing piece.

    Returns the league pair unchanged when this team has no prior home or away matches, so the
    early season degrades to exactly today's behaviour rather than to a one-match estimate.
    """
    league_xg_mult, league_xga_mult = league_multipliers
    if team_history.empty:
        return league_xg_mult, league_xga_mult
    prior = team_history[team_history["date"] < before]
    home = prior[prior["is_home"]]
    if home.empty or len(prior) == len(home):
        return league_xg_mult, league_xga_mult

    overall_xg = float(prior["xG"].mean())
    overall_xga = float(prior["xGA"].mean())
    if not overall_xg > 0 or not overall_xga > 0:
        return league_xg_mult, league_xga_mult

    n_home = float(len(home))
    return (
        shrink_toward_prior(
            float(home["xG"].mean()) / overall_xg, n_home, league_xg_mult, shrinkage_k
        ),
        shrink_toward_prior(
            float(home["xGA"].mean()) / overall_xga, n_home, league_xga_mult, shrinkage_k
        ),
    )


def build_fixture_rate_frame(
    gw: pd.DataFrame,
    team_histories: dict[str, pd.DataFrame],
    venue_shrinkage_k: float = TEAM_VENUE_SHRINKAGE_K,
) -> pd.DataFrame:
    """One row per (team, gameweek): that team's own and its opponent's point-in-time xG/xGA
    rates, the gameweek's league-average xGA, and a Dixon-Coles clean-sheet probability computed
    at the untuned :data:`~engine.models.clean_sheets.DEFAULT_DIXON_COLES_RHO` (used only to build
    the bonus regression's training features — see module docstring, simplification 3).

    Team rates are shrunk toward the point-in-time league average (:func:`_team_rate_asof_shrunk`)
    and adjusted by a single league-wide home/away multiplier (:func:`_league_venue_multipliers`),
    per ENGINE_IMPROVEMENTS_3.md A.1 — see those functions' docstrings for why this replaced the
    previous per-team venue split.
    """
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
    fixtures["_team_xg_raw"] = [
        _team_rate_asof(team_histories.get(team, empty), "xG", kickoff)
        for team, kickoff in zip(fixtures["team"], fixtures["kickoff_time"], strict=True)
    ]
    fixtures["_team_xga_raw"] = [
        _team_rate_asof(team_histories.get(team, empty), "xGA", kickoff)
        for team, kickoff in zip(fixtures["team"], fixtures["kickoff_time"], strict=True)
    ]
    # Point-in-time league average of the RAW (unshrunk) rate, per gameweek — the prior every
    # team's own rate that gameweek shrinks toward.
    fixtures["_league_avg_xg_raw"] = fixtures.groupby("gameweek")["_team_xg_raw"].transform("mean")
    fixtures["_league_avg_xga_raw"] = fixtures.groupby("gameweek")["_team_xga_raw"].transform(
        "mean"
    )
    fixtures["_team_xg_shrunk"] = [
        _team_rate_asof_shrunk(team_histories.get(team, empty), "xG", kickoff, league_avg)
        for team, kickoff, league_avg in zip(
            fixtures["team"], fixtures["kickoff_time"], fixtures["_league_avg_xg_raw"], strict=True
        )
    ]
    fixtures["_team_xga_shrunk"] = [
        _team_rate_asof_shrunk(team_histories.get(team, empty), "xGA", kickoff, league_avg)
        for team, kickoff, league_avg in zip(
            fixtures["team"],
            fixtures["kickoff_time"],
            fixtures["_league_avg_xga_raw"],
            strict=True,
        )
    ]

    # League-wide venue multiplier, point-in-time as of this gameweek's earliest kickoff — one fit
    # per gameweek — then each team's own multiplier shrunk toward it (B1).
    league_mults = {
        gameweek: _league_venue_multipliers(team_histories, kickoffs.min())
        for gameweek, kickoffs in fixtures.groupby("gameweek")["kickoff_time"]
    }
    gameweek_start = fixtures.groupby("gameweek")["kickoff_time"].min()
    team_mults = [
        _team_venue_multipliers(
            team_histories.get(team, empty),
            gameweek_start[gameweek],
            league_mults[gameweek],
            venue_shrinkage_k,
        )
        for team, gameweek in zip(fixtures["team"], fixtures["gameweek"], strict=True)
    ]
    xg_mult = pd.Series([m[0] for m in team_mults], index=fixtures.index)
    xga_mult = pd.Series([m[1] for m in team_mults], index=fixtures.index)
    fixtures["team_xg_per_90"] = np.where(
        fixtures["was_home"],
        fixtures["_team_xg_shrunk"] * xg_mult,
        fixtures["_team_xg_shrunk"] * (2.0 - xg_mult),
    )
    fixtures["team_xga_per_90"] = np.where(
        fixtures["was_home"],
        fixtures["_team_xga_shrunk"] * xga_mult,
        fixtures["_team_xga_shrunk"] * (2.0 - xga_mult),
    )
    fixtures = fixtures.drop(
        columns=[
            "_team_xg_raw",
            "_team_xga_raw",
            "_league_avg_xg_raw",
            "_league_avg_xga_raw",
            "_team_xg_shrunk",
            "_team_xga_shrunk",
        ]
    )
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


UNDERSTAT_RATE_COLUMNS = ("npxg_per_90", "xa_per_90")
TEAM_RATE_COLUMNS = (
    "team_xg_per_90",
    "team_xga_per_90",
    "opponent_xg_per_90",
    "opponent_xga_per_90",
    "league_avg_xga_per_90",
    "clean_sheet_probability_default_rho",
)


def classify_drop_reasons(gw: pd.DataFrame, required: Sequence[str]) -> pd.DataFrame:
    """Why each row `engineer_features` is about to drop is being dropped, and what share of the
    season's points goes with it.

    The dropna removes ~46% of raw rows, and until now the report showed only the total. That total
    is compatible with two very different worlds: a sample filtered almost entirely by the
    structural cold start (harmless — every player has a first gameweek), or one filtered toward
    established, well-covered players (which would quietly flatter every accuracy number in the
    report). ``points_share`` is what distinguishes them, so it is reported per reason rather than
    only in aggregate.

    Reasons are assigned by precedence, so they partition the dropped rows exactly rather than
    double-counting a row whose columns are NaN for several reasons at once. The precedence follows
    causation: a player's first appearance has *every* point-in-time EWMA NaN by construction, so
    reporting it as "unmatched crosswalk" would be misleading even though the Understat columns are
    indeed NaN on that row too.
    """
    present = [column for column in required if column in gw.columns]
    missing_any = gw[present].isna().any(axis=1) if present else pd.Series(False, index=gw.index)

    # First row per player, by kickoff — no prior match exists, so no lagged rate can be computed.
    first_appearance = gw.groupby("player_id")["kickoff_time"].rank(method="first") == 1
    understat_missing = _any_column_na(gw, UNDERSTAT_RATE_COLUMNS)
    team_rates_missing = _any_column_na(gw, TEAM_RATE_COLUMNS)

    reason = pd.Series("", index=gw.index, dtype=object)
    reason[missing_any & first_appearance] = "first_appearance"
    reason[missing_any & (reason == "") & understat_missing] = "unmatched_crosswalk"
    reason[missing_any & (reason == "") & team_rates_missing] = "missing_team_history"
    reason[missing_any & (reason == "")] = "other_missing_feature"

    total_points = float(gw["total_points"].sum()) if "total_points" in gw.columns else 0.0
    rows = []
    for name in (
        "first_appearance",
        "unmatched_crosswalk",
        "missing_team_history",
        "other_missing_feature",
    ):
        mask = reason == name
        dropped_points = float(gw.loc[mask, "total_points"].sum()) if total_points else 0.0
        rows.append(
            {
                "reason": name,
                "n_rows": int(mask.sum()),
                "row_share": float(mask.mean()) if len(gw) else float("nan"),
                "points_share": dropped_points / total_points if total_points else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _any_column_na(gw: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    present = [column for column in columns if column in gw.columns]
    if not present:
        return pd.Series(False, index=gw.index)
    return gw[present].isna().any(axis=1)


def _apply_live_availability(gw: pd.DataFrame, live_availability: pd.DataFrame) -> pd.DataFrame:
    """Overwrite ``chance_of_playing_next_round``/``status_score`` on every synthesized target row
    only, never a real played-history row, see :func:`engineer_features`'s own docstring for why.
    ``live_availability`` carries ``player_id``/``chance_of_playing_next_round``/``status`` (the
    raw FPL status code, encoded here via the same :func:`~engine.models.minutes.encode_status`
    the backtest default uses, so a caller passes the live field as-is rather than pre-encoding it).

    Keys off ``is_synthesized_target`` rather than ``gameweek == gw["gameweek"].max()``. With a
    single-gameweek horizon the two are identical, but a multi-gameweek horizon built in one pass
    (T-A/T-F, ENGINE_AUDIT_FIXES-implementation.md) synthesizes one target row per horizon
    gameweek, and the old maximum-gameweek check only ever patched the last one, leaving earlier
    target gameweeks (the ones a manager actually picks for) silently holding the fully fit
    default.

    KNOWN SIMPLIFICATION, documented deliberately rather than picked silently:
    ``chance_of_playing_next_round`` is semantically about the very next match only, yet this
    function applies the same live reading, unchanged, to every target row in the horizon (GW1,
    GW2 and GW3 alike). It is not decayed toward "recovered" nor zeroed out further out. This is
    wrong in both directions, a genuinely short-term knock stays pessimistic past a real recovery,
    and a fresh injury or return that happens after this snapshot is not reflected at all, but
    inventing a recovery-decay curve with no real data to calibrate it against would fabricate
    false precision. Holding today's reading flat across the horizon is the safer default for a
    first implementation; replacing it with a calibrated decay is real, separate follow-up work,
    not attempted here.
    """
    is_target = gw["is_synthesized_target"].astype(bool)
    availability = live_availability.set_index("player_id")

    chance = gw["player_id"].map(availability["chance_of_playing_next_round"])
    use_chance = is_target & chance.notna()
    gw.loc[use_chance, "chance_of_playing_next_round"] = chance[use_chance]

    status = gw["player_id"].map(availability["status"])
    use_status = is_target & status.notna()
    gw.loc[use_status, "status_score"] = status[use_status].map(encode_status)
    return gw


def engineer_features(
    merged_gw: pd.DataFrame,
    teams: pd.DataFrame,
    team_histories: dict[str, pd.DataFrame],
    player_histories: dict[int, pd.DataFrame],
    live_availability: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Join the vaastav ground truth, the FPL team-id map, Understat team-level rates, and
    Understat player-level rates into one point-in-time ``history`` frame keyed by
    ``(player_id, gameweek)`` — exactly what :func:`backtest.harness.run_walk_forward` and
    :func:`engine.pipeline.project_gameweek_pool` expect.

    Includes goalkeepers (ENGINE_IMPROVEMENTS_3.md D.1) — Tier 3.2's real blocker (no opponent
    shots-on-target data source) only affects the saves component's opponent adjustment, which is
    ~18% of GK scoring; appearance, clean sheets, goals-conceded and bonus (the other ~82%) need
    nothing new. Saves fall back to a shrunk own-rate EWMA (``own_save_rate_per_90``, see
    :func:`engine.models.saves.project_saves_from_own_rate`) rather than the real opponent-adjusted
    model. Goalkeepers essentially never register meaningful Understat npxG/xA (and the crosswalk
    doesn't attempt to match them at all), so ``npxg_per_90``/``xa_per_90`` default to 0.0 for GK
    rows specifically rather than the NaN an unmatched *outfield* player correctly gets (which must
    still be dropped, not defaulted, per ENGINE_IMPROVEMENTS_2.md C.1/C.2).

    ``live_availability`` (A1, T-F) real ``chance_of_playing_next_round``/``status`` are FPL
    live-only fields with no retained history, so the backtest hardcodes them to "fully fit"
    below for every row, past and present alike. Live, they are exactly the two fields FPL itself
    surfaces as its own injury/rotation signal, and B.3's crowd features (`transfers_out_share`
    especially) were only ever a retrospectively-available *proxy* for them. Supplying a frame
    with ``player_id``/``chance_of_playing_next_round``/``status`` columns overrides those two
    columns on every synthesized target row (``is_synthesized_target``, T-A), never a real
    played-history row, since "today's status" has no meaning applied retroactively and doing so
    would leak today's knowledge into training rows that must stay exactly as point-in-time as the
    backtest's own. With a multi-gameweek horizon this means every horizon gameweek gets patched,
    not only the last one, see :func:`_apply_live_availability` for that function's own known
    simplification about applying one live reading unchanged across the whole horizon. A player in
    ``merged_gw`` but missing from ``live_availability`` keeps the "fully fit" default rather than
    raising, a snapshot gap here is a data-quality issue to alert on upstream, not a reason to fail
    projection for every other player. ``None`` (the default) reproduces today's backtest-only
    behaviour exactly.
    """
    gw = merged_gw[merged_gw["position"].isin([GK, DEF, MID, FWD])].copy()
    gw["kickoff_time"] = pd.to_datetime(gw["kickoff_time"], utc=True)
    gw = gw.rename(columns={"element": "player_id", "GW": "gameweek"})

    # T-A: real vaastav merged_gw frames (the backtest path) carry no is_synthesized_target
    # column at all, since every one of their rows is real played history. Default it to False
    # there so the column contract matches the live path's (see engine.data.live_adapter's
    # MERGED_GW_COLUMNS), and so it survives this function's own merges/collapse_double_gameweeks
    # untouched below. T-E reads this column further down to keep a synthesized target row out of
    # another target row's lagged per-player feature windows.
    if "is_synthesized_target" not in gw.columns:
        gw["is_synthesized_target"] = False

    team_id_to_name = dict(zip(teams["id"], teams["name"], strict=True))
    gw["opponent_team_name"] = gw["opponent_team"].map(team_id_to_name)

    # --- Tier 1.1: minutes-model features -------------------------------------------------------
    # T-E: a horizon built in one pass (engine.data.live_adapter.build_merged_gw) synthesizes one
    # placeholder row per player per target gameweek, with minutes/starts filled with 0 since the
    # real outcome isn't known yet. Every lagged per-player feature below is computed with
    # .shift(1) over this player's own chronologically-sorted rows, so without correction a later
    # target gameweek's window would find an earlier target gameweek's placeholder row sitting
    # inside it and read the fake 0 as "this player was just benched", decaying the minutes model's
    # confidence purely as an artifact of horizon length rather than any real signal (see
    # ENGINE_AUDIT_FIXES-implementation.md T-E). Masking a synthesized row's own value to NaN
    # before shifting means the pandas ewm/rolling mean below skips it: the lagged feature carried
    # into a later target row is the last *real* observation, not a fabricated zero.
    gw["recent_start_rate"] = _per_player_series(
        gw,
        lambda g: g["starts"]
        .astype(float)
        .mask(g["is_synthesized_target"].astype(bool), np.nan)
        .shift(1)
        .ewm(halflife=DEFAULT_HALFLIFE, adjust=True)
        .mean(),
    )
    gw["recent_minutes_ewma"] = _per_player_series(
        gw,
        lambda g: g["minutes"]
        .astype(float)
        .mask(g["is_synthesized_target"].astype(bool), np.nan)
        .shift(1)
        .ewm(halflife=DEFAULT_HALFLIFE, adjust=True)
        .mean(),
    )
    gw["fixture_congestion"] = compute_fixture_congestion(gw)
    gw["chance_of_playing_next_round"] = 100.0  # live-only field, unavailable retrospectively
    gw["status_score"] = encode_status("a")  # live-only field, unavailable retrospectively
    if live_availability is not None and not gw.empty:
        gw = _apply_live_availability(gw, live_availability)
    gw["days_since_last_appearance"] = compute_days_since_last_appearance(gw)
    gw["zero_minute_streak_length"] = compute_zero_minute_streak_length(gw)
    for window in (3, 6, 15):
        gw[f"start_rate_last_{window}"] = _per_player_series(
            gw,
            lambda g, w=window: g["starts"]
            .astype(float)
            .mask(g["is_synthesized_target"].astype(bool), np.nan)
            .shift(1)
            .rolling(w, min_periods=1)
            .mean(),
        )
    gw["team_rotation_propensity"] = compute_team_rotation_propensity(gw)
    # ENGINE_IMPROVEMENTS_3.md Phase 3: see engine.models.minutes.FEATURE_COLUMNS' own comment for
    # why this is a real minutes-model feature (not the structural redundancy BUILD_PLAN 2.1 named
    # for the *bonus* model) once goalkeepers are in the pool.
    gw["is_goalkeeper"] = (gw["position"] == GK).astype(float)

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
    # Multi-season backtest Phase 2: defensive contribution's raw inputs (and the outcome column
    # itself) are absent from this archive for 2020/21-2024/25 (a real archive hole, not just a
    # rule-introduction artifact — verified against the real vaastav data; see
    # ENGINE_IMPROVEMENTS_3.md's multi-season plan). A season missing "defensive_contribution"
    # gets a neutral 0.0 placeholder for both the outcome and the rate, so downstream code that
    # unconditionally expects this column (ground-truth selection, pipeline validation) doesn't
    # need special-casing — `dc_data_available` (surfaced via `result.attrs` below) is what tells
    # the multi-season aggregator to exclude these rows from DC-specific calibration rather than
    # silently scoring a fabricated near-zero rate as if it meant something.
    dc_data_available = "defensive_contribution" in gw.columns
    if dc_data_available:
        gw["dc_per_90"] = _per_player_series(
            gw, lambda g: ewma_rate_asof(g, "defensive_contribution", minutes_col="minutes")
        )
    else:
        gw["defensive_contribution"] = 0.0
        gw["dc_per_90"] = 0.0
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
    # ENGINE_IMPROVEMENTS_3.md A.3: point-in-time evidence weight behind this player's own card/
    # own-goal rates above — vaastav's own `minutes` column (not Understat's), since that's what
    # those rates are themselves computed from. Also doubles as the shrinkage weight for D.1's
    # goalkeeper own-rate saves fallback below, the same "how much real vaastav-minutes history do
    # we have" quantity.
    gw["card_effective_minutes"] = _per_player_series(
        gw, lambda g: effective_sample_minutes_asof(g, minutes_col="minutes")
    )
    # ENGINE_IMPROVEMENTS_3.md D.1: goalkeeper own-rate saves fallback — see engineer_features'
    # own docstring and engine.models.saves.project_saves_from_own_rate.
    gw["own_save_rate_per_90"] = _per_player_series(
        gw, lambda g: ewma_rate_asof(g, "saves", minutes_col="minutes")
    )
    # ENGINE_IMPROVEMENTS_3.md D.2: point-in-time BPS-per-90 rate — the input the soft
    # within-fixture BPS-rank bonus diagnostic (backtest.diagnostics.rank_based_bonus_diagnostics)
    # needs but the shipped linear BonusModel does not (it regresses on BPS-relevant *component*
    # inputs, not this raw rate).
    gw["bps_per_90"] = _per_player_series(
        gw, lambda g: ewma_rate_asof(g, "bps", minutes_col="minutes")
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
    # ENGINE_IMPROVEMENTS_3.md D.1: goalkeepers essentially never register meaningful Understat
    # npxG/xA and the crosswalk doesn't try to match them, so an unmatched GK's rate here would be
    # NaN and (correctly, for an outfield player) get the row dropped below. 0.0 is the honest
    # modelling choice for a GK specifically — NOT for an outfield player, whose NaN must still
    # signal a real crosswalk miss (ENGINE_IMPROVEMENTS_2.md C.1/C.2).
    is_gk = gw["position"] == GK
    gw.loc[is_gk, "npxg_per_90"] = gw.loc[is_gk, "npxg_per_90"].fillna(0.0)
    gw.loc[is_gk, "xa_per_90"] = gw.loc[is_gk, "xa_per_90"].fillna(0.0)
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
    # T-K: only 38 players recorded a real penalty attempt across the entire 2025-26 training
    # window, so the EWMA above gives a newly appointed or transferred-in penalty taker exactly
    # zero taker_share until after they have already taken a live penalty for real. FPL's live
    # bootstrap penalties_order rank has no replayable per-gameweek history to backfill a backtest
    # with (engine/models/goals.py's own docstring already explains why), so this seeds coverage on
    # the live path only: a synthesized target row (engine.data.live_adapter.build_merged_gw) whose
    # penalties_order is exactly 1, the designated primary taker, gets a full 1.0 taker_share on
    # that row alone, regardless of how thin (or nonexistent) their own realized-attempt history
    # is. Every other rank, and every row with no live rank at all (every real played-history row,
    # and the entire backtest path, where the column is absent), keeps the realized-attempt EWMA
    # computed above unchanged.
    if "penalties_order" in gw.columns:
        is_live_primary_taker = gw["is_synthesized_target"].astype(bool) & (
            gw["penalties_order"] == 1
        )
        gw.loc[is_live_primary_taker, "taker_share"] = 1.0

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
        # Multi-season Phase 2: these used to be implicitly gated by dc_per_90's own cold-start
        # NaN (every per-player EWMA rate shares the same "no row before this player's first
        # gameweek" pattern) — but a season lacking DC's raw archive columns now gives dc_per_90 a
        # constant 0.0 placeholder rather than a real EWMA, so it's never NaN there any more and
        # stopped gating anything. Listing these explicitly restores the same cold-start drop
        # regardless of whether this season happens to have real DC data.
        "yellow_card_rate_per_90",
        "red_card_rate_per_90",
        "own_save_rate_per_90",
        "own_goal_rate_per_90",
        "clean_sheet_probability_default_rho",
    ]
    n_before_dropna = len(gw)
    drop_reasons = classify_drop_reasons(gw, required)
    result = gw.dropna(subset=required).reset_index(drop=True)
    # A.5 coverage reporting reads these off the returned frame immediately after the call, before
    # any further pandas operation that might not preserve them.
    result.attrs["n_dgw_rows_collapsed"] = n_dgw_rows_collapsed
    result.attrs["n_rows_before_dropna"] = n_before_dropna
    result.attrs["n_rows_dropped_for_missing_features"] = n_before_dropna - len(result)
    # Stored as records, not a DataFrame: pandas propagates `attrs` through `merge`, and when both
    # sides carry the same key it compares the two values with `==` to decide whether to keep it.
    # A DataFrame there makes that comparison ambiguous and merge raises.
    result.attrs["drop_reasons"] = drop_reasons.to_dict("records")
    result.attrs["dc_data_available"] = dc_data_available
    return result


# =================================================================================================
# Fit / predict, wired to backtest.harness.run_walk_forward
# =================================================================================================


@dataclass(frozen=True)
class FittedEngineState:
    minutes_model: MinutesModel
    bonus_model: BonusModel
    fitted_constants: FittedConstants


def _fit_rate_conversion_factor_by_position(
    training_history: pd.DataFrame,
    rate_col: str,
    count_col: str,
    min_rows: int = 200,
    positions: tuple[str, ...] = (DEF, MID, FWD, GK),
) -> dict[str, float]:
    """Point-in-time multiplier turning an xG/xA-derived per-90 rate into the quantity FPL actually
    awards (ENGINE_IMPROVEMENTS_5.md Tier 2.3), fitted per position on ``training_history`` alone
    and refit every gameweek by the walk-forward harness, the same discipline as every other
    constant here.

    The rate is evaluated at each row's **realised** minutes, not its modelled expected minutes.
    That is the key methodological point: it removes the minutes model from the comparison, and it
    removes the selection effect that makes the gate's played-rows calibration misleading (see
    ``score_season``). What is left is the rate model's own accuracy against real outcomes.

    Real 2025/26 numbers this was built from, evaluated exactly this way: goals predicted 1098.9
    against 929 actual (factor 0.845), assists predicted 772.4 against 864 actual (factor 1.119).
    The assist gap is definitional rather than a modelling error, since FPL credits the final pass
    however the goal arrived while an xA model does not; FPL's own realised xA shows the same
    under-count, at 1.360.

    Falls back to 1.0 (a no-op) for a position with too few rows or no predicted mass, so an early
    gameweek with thin history leaves the component exactly as it was rather than applying a wild
    multiplier fitted on a handful of matches.
    """
    fixture_adjustment = (
        training_history["opponent_xga_per_90"] / training_history["league_avg_xga_per_90"]
    )
    realised_minutes_share = training_history["minutes"].astype(float) / 90.0
    predicted_at_realised_minutes = (
        training_history[rate_col] * fixture_adjustment * realised_minutes_share
    )

    factors: dict[str, float] = {}
    for position in positions:
        mask = training_history["position"] == position
        subset_predicted = float(predicted_at_realised_minutes[mask].sum())
        subset_actual = float(training_history.loc[mask, count_col].astype(float).sum())
        if int(mask.sum()) < min_rows or subset_predicted <= 0:
            factors[position] = 1.0
            continue
        factors[position] = subset_actual / subset_predicted
    return factors


def _fit_league_avg_rate_by_position(
    training_history: pd.DataFrame,
    count_col: str,
    min_rows: int = 100,
    positions: tuple[str, ...] = (DEF, MID, FWD),
) -> dict[str, float]:
    """Point-in-time (``training_history`` only) per-position league-average per-90 rate of
    ``count_col`` — the shrinkage prior for the card/own-goal rates (ENGINE_IMPROVEMENTS_3.md
    A.3) and the goalkeeper own-rate saves fallback (D.1, via ``positions=(GK,)``), refit every
    gameweek by the walk-forward harness, same discipline as the other Tier 1.2 constants. A plain
    aggregate (total events / total minutes), not an EWMA — this is the prior the *individual* EWMA
    rate shrinks toward, so it should reflect the position's whole-window base rate, not its own
    recency-weighted view. Falls back to ``0.0`` for a too-thin sample (only ever binds before
    ``min_training_gameweeks`` worth of history has accumulated for a position, which the harness's
    own skip-gameweeks-with-too-little-history guarantee already keeps rare).
    """
    rates: dict[str, float] = {}
    for position in positions:
        subset = training_history[training_history["position"] == position]
        total_minutes = float(subset["minutes"].sum())
        if len(subset) < min_rows or total_minutes <= 0:
            rates[position] = 0.0
            continue
        rates[position] = float(subset[count_col].sum()) / total_minutes * 90.0
    return rates


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

    # ENGINE_IMPROVEMENTS_3.md A.2: bonus's training features must use the same MODELLED expected
    # minutes the predict path (engine.pipeline._project_one_player) uses, not each row's REALISED
    # minutes — the prior version's train/serve skew, and the reason nothing in the bonus feature
    # set previously depended on whether the player was even expected to play (clean_sheet_prob-
    # ability and defensive_action_rate are team/rate-level; expected_goals/expected_assists were
    # the only minutes-scaled inputs, and only at fit time).
    modelled_expected_minutes = pd.Series(
        [
            distribution.expected_minutes
            for distribution in minutes_model.predict(training_history[MINUTES_FEATURE_COLUMNS])
        ],
        index=training_history.index,
    )
    expected_goals = (
        training_history["npxg_per_90"]
        * (training_history["opponent_xga_per_90"] / training_history["league_avg_xga_per_90"])
        * (modelled_expected_minutes / 90.0)
    )
    expected_assists = (
        training_history["xa_per_90"]
        * (training_history["opponent_xga_per_90"] / training_history["league_avg_xga_per_90"])
        * (modelled_expected_minutes / 90.0)
    )
    bonus_features = pd.DataFrame(
        {
            "expected_goals": expected_goals,
            "expected_assists": expected_assists,
            "clean_sheet_probability": training_history["clean_sheet_probability_default_rho"],
            "defensive_action_rate": training_history["dc_per_90"],
            "expected_minutes": modelled_expected_minutes,
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

    # ENGINE_IMPROVEMENTS_3.md A.3: per-position league-average card/own-goal rates, the
    # shrinkage prior for the thin-sample individual rates.
    # ENGINE_IMPROVEMENTS_4.md: per-position replacement for the flat DEFAULT_ASSIST_SHARE_OF_
    # TEAM_XG — see engine.pipeline.FittedConstants.assist_share_of_team_xg_by_position's own
    # docstring for why a single flat share was the actual defect behind a severe FWD bias.
    assist_share_by_position = {}
    for position in (DEF, MID, FWD):
        subset = training_history[training_history["position"] == position]
        assist_share_by_position[position] = fit_assist_share_of_team_xg(
            subset["assists"], subset["team_xg_per_90"], subset["minutes"]
        )

    # Tier 2.3: xG/xA-rate -> FPL-awarded-quantity conversion, per position, training history only.
    goal_conversion_factor = _fit_rate_conversion_factor_by_position(
        training_history, "npxg_per_90", "goals_scored"
    )
    assist_conversion_factor = _fit_rate_conversion_factor_by_position(
        training_history, "xa_per_90", "assists"
    )
    league_avg_yellow_card_rate = _fit_league_avg_rate_by_position(training_history, "yellow_cards")
    league_avg_red_card_rate = _fit_league_avg_rate_by_position(training_history, "red_cards")
    league_avg_own_goal_rate = _fit_league_avg_rate_by_position(training_history, "own_goals")
    # ENGINE_IMPROVEMENTS_3.md D.1: league-average GK saves-per-90 rate, the shrinkage prior for
    # the own-rate saves fallback.
    league_avg_save_rate = _fit_league_avg_rate_by_position(
        training_history, "saves", positions=(GK,)
    ).get(GK, 0.0)

    fitted_constants = FittedConstants(
        dixon_coles_rho=rho,
        save_conversion_rate=save_conversion_rate,
        away_shot_multiplier=away_shot_multiplier,
        dc_overdispersion_alpha=dc_alpha_by_position,
        penalty_conversion_rate_by_player=penalty_rates_by_player,
        league_avg_penalty_conversion_rate=league_avg_penalty_rate,
        goals_shrinkage_k=GOALS_SHRINKAGE_K,
        assists_shrinkage_k=ASSISTS_SHRINKAGE_K,
        assist_share_of_team_xg_by_position=assist_share_by_position,
        league_avg_yellow_card_rate_by_position=league_avg_yellow_card_rate,
        league_avg_red_card_rate_by_position=league_avg_red_card_rate,
        league_avg_own_goal_rate_by_position=league_avg_own_goal_rate,
        yellow_card_shrinkage_k=YELLOW_CARD_SHRINKAGE_K,
        red_card_shrinkage_k=RED_CARD_SHRINKAGE_K,
        own_goal_shrinkage_k=OWN_GOAL_SHRINKAGE_K,
        league_avg_save_rate_per_90=league_avg_save_rate,
        save_rate_shrinkage_k=SAVE_RATE_SHRINKAGE_K,
        goal_conversion_factor_by_position=goal_conversion_factor,
        assist_conversion_factor_by_position=assist_conversion_factor,
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
        # ENGINE_IMPROVEMENTS_3.md D.1: same own-rate saves fallback as the point-estimate path
        # (engine.pipeline._project_one_player) — a full-match (90-minute) rate, since
        # engine.simulate itself scales by each run's own drawn minutes.
        saves_projection = (
            project_saves_from_own_rate(
                own_save_rate_per_90=float(row["own_save_rate_per_90"]),
                expected_minutes=90.0,
                individual_weight=float(row.get("card_effective_minutes", 0.0)),
                league_avg_save_rate_per_90=fitted_state.fitted_constants.league_avg_save_rate_per_90,
                shrinkage_k=fitted_state.fitted_constants.save_rate_shrinkage_k,
            )
            if position == GK
            else None
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
                expected_saves_full_match=(
                    saves_projection.expected_saves if saves_projection is not None else 0.0
                ),
                penalty_save_rate=(
                    saves_projection.penalty_save_rate if saves_projection is not None else 0.0
                ),
            )
        )
    team_expected_penalties = (
        float(team_players["team_expected_penalties"].iloc[0])
        if "team_expected_penalties" in team_players.columns
        else 0.0
    )
    return TeamMatchInputs(players=players, team_expected_penalties=team_expected_penalties)


# Sentinel player_id for the synthetic league-average opponent built by
# ``_synthetic_league_average_opponent`` (ENGINE_AUDIT_FIXES T-I). Never a real FPL player_id
# (those are positive), so it can be filtered back out of a fixture's simulated summaries without
# risk of colliding with an actual player.
_SYNTHETIC_OPPONENT_PLAYER_ID = -1

# DataFrame.attrs key ``simulate_gameweek_pool`` uses to surface how many fixtures in this
# gameweek's pool fell back to a synthetic opponent, rather than degrading silently (this repo's
# stated discipline, per e.g. ``engine/data/cold_start.py``'s own low-confidence flagging).
SIMULATE_FALLBACK_OPPONENT_COUNT_ATTR = "fallback_opponent_fixture_count"


def _synthetic_league_average_opponent(known_group: pd.DataFrame) -> TeamMatchInputs:
    """A one-"player" stand-in for an opponent whose entire squad is absent from the engineered
    pool (ENGINE_AUDIT_FIXES T-I) -- almost always a newly promoted club whose players have no
    engine features yet and were therefore dropped by ``engineer_features``'s required-columns
    ``dropna``, taking every fixture against them down with it.

    Rather than skip the fixture and leave the *known* side with ``simulation=None``, give that
    side a real opponent lambda to play against: a single full-90-minute attacker whose goal rate
    is this gameweek's ``league_avg_xga_per_90`` (already a column on every real pool row, so it
    needs no extra fetch). That figure is the point-in-time league average of team-level expected
    goals *conceded*, which by construction equals the league average of goals *scored* (every
    league goal is scored by one team and conceded by another), so it is a reasonable proxy for
    "an average Premier League team's attack" with no information at all about the actual missing
    club. This is used only to produce a plausible drawn scoreline for the known side's clean sheet
    and goals-conceded components; the synthetic side's own summary is discarded.
    """
    league_avg_rate = float(known_group["league_avg_xga_per_90"].iloc[0])
    synthetic_player = PlayerMatchInputs(
        player_id=_SYNTHETIC_OPPONENT_PLAYER_ID,
        position=MID,
        minutes_distribution=MinutesDistribution(
            p_zero=0.0,
            p_1_to_59=0.0,
            p_60_plus=1.0,
            expected_minutes_given_1_to_59=0.0,
            expected_minutes_given_60_plus=90.0,
        ),
        adjusted_goal_rate_per_90=league_avg_rate,
        adjusted_assist_rate_per_90=0.0,
    )
    return TeamMatchInputs(players=[synthetic_player])


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
    ``(team, opponent_team_name)`` — each team processed exactly once. A team whose opponent
    doesn't also appear in this gameweek's pool (ENGINE_AUDIT_FIXES T-I: almost always because the
    opponent's entire squad routed through the cold-start fallback and was dropped by
    ``engineer_features``) is simulated against a synthetic league-average opponent
    (:func:`_synthetic_league_average_opponent`) instead of being skipped, so the known side still
    gets a real floor/ceiling/prob_big_haul rather than ``None``. How many fixtures needed this
    fallback is counted and attached to the returned frame's
    ``attrs[SIMULATE_FALLBACK_OPPONENT_COUNT_ATTR]``, so degradation stays visible rather than
    silent.
    """
    columns = [
        "player_id",
        "gameweek",
        "sim_mean",
        "sim_median",
        "floor",
        "ceiling",
        "prob_big_haul",
    ]
    if players_gw.empty:
        result = pd.DataFrame(columns=columns)
        result.attrs[SIMULATE_FALLBACK_OPPONENT_COUNT_ATTR] = 0
        return result

    rows: list[dict[str, float]] = []
    processed_teams: set[str] = set()
    fallback_opponent_fixture_count = 0
    for team, group in players_gw.groupby("team"):
        if team in processed_teams:
            continue
        opponent = group["opponent_team_name"].iloc[0]
        opponent_group = players_gw[players_gw["team"] == opponent]
        is_home = bool(group["was_home"].iloc[0])
        known_player_ids: set[int] | None = None

        if opponent_group.empty:
            processed_teams.add(team)
            fallback_opponent_fixture_count += 1
            known_player_ids = set(group["player_id"].astype(int))
            known_inputs = _build_team_match_inputs(group, fitted_state)
            synthetic_inputs = _synthetic_league_average_opponent(group)
            home_inputs, away_inputs = (
                (known_inputs, synthetic_inputs) if is_home else (synthetic_inputs, known_inputs)
            )
        else:
            processed_teams.add(team)
            processed_teams.add(opponent)
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
            if known_player_ids is not None and player_id not in known_player_ids:
                continue
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
    output = pd.DataFrame(rows, columns=columns)
    output.attrs[SIMULATE_FALLBACK_OPPONENT_COUNT_ATTR] = fallback_opponent_fixture_count
    return output


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
class UnmatchedSignificantPlayer:
    """One outfield player with real playing time (>450 minutes) but no crosswalk match — the
    concrete "who" behind ``CoverageReport.points_excluded_share`` (crosswalk coverage Phase 1).
    Previously this share was only ever visible as an aggregate percentage; there was no way to
    see *which* players were actually missing without a fresh ad hoc script each time."""

    fpl_id: int
    name: str
    minutes: float
    points: float


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
    unmatched_significant_players: tuple[UnmatchedSignificantPlayer, ...] = ()
    drop_reasons: pd.DataFrame | None = None

    def summary(self) -> str:
        lines = [
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
        if self.drop_reasons is not None and not self.drop_reasons.empty:
            lines.append(
                "Why rows were dropped (by precedence, so reasons partition the dropped rows; "
                "points_share is the share of ALL points on dropped rows -- the number that says "
                "whether the scored sample is representative or filtered toward established "
                "players):"
            )
            lines.append(self.drop_reasons.to_string(index=False))
        if self.unmatched_significant_players:
            lines.append("Unmatched significant players (crosswalk coverage Phase 1):")
            lines.extend(
                f"  {p.name} (fpl_id={p.fpl_id}): {p.minutes:.0f} minutes, {p.points:.0f} points"
                for p in sorted(
                    self.unmatched_significant_players, key=lambda p: p.points, reverse=True
                )
            )
        return "\n".join(lines)


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
    # B3: the same components, scored only on rows where the player actually appeared -- see this
    # field's construction in `score_season` for why the all-rows figure above is not a
    # measurement of these components (it is dominated by the minutes model). This is the version
    # fed to the gate.
    mean_calibrations_played: dict[str, metrics.MeanCalibrationReport]
    baseline_results: dict[str, baselines.PairedBootstrapResult]
    definition_of_done: gate.DefinitionOfDoneReport
    coverage: CoverageReport | None = None
    captaincy: metrics.CaptaincyHitRateResult | None = None
    floor_ceiling_coverage: float | None = None
    big_haul_calibration: metrics.CalibrationReport | None = None
    # ENGINE_IMPROVEMENTS_3.md B.2/B.3 — computed only when ground_truth carries the extra
    # columns they need ("value" for price tiers; "team"/"was_home"/"team_h_score"/"team_a_score"
    # for the team-level clean-sheet check), so existing minimal ground_truth frames keep working.
    bias_by_price_tier: metrics.BiasReport | None = None
    team_clean_sheet_calibration: metrics.CalibrationReport | None = None
    brier_reports: dict[str, metrics.BrierComparisonReport] = field(default_factory=dict)
    # ENGINE_IMPROVEMENTS_5.md Tier 0.1 — the decision-relevant siblings of `rank_correlation` and
    # `bias_by_position`/`bias_by_price_tier` above. Reported alongside them rather than replacing
    # them: the pooled/unconditional figures answer "does the engine understand the whole pool",
    # these answer "can it be acted on", and the two diverge sharply (see score_season's own note).
    # Tier 2.3: goals/assists rate calibration at realised minutes -- the gate's component
    # instrument, and the only one of the three views free of both confounds.
    rate_calibrations: dict[str, metrics.MeanCalibrationReport] = field(default_factory=dict)
    decision_set_rank: metrics.DecisionSetRankReport | None = None
    conditional_bias_by_position: metrics.BiasReport | None = None
    conditional_bias_by_price_tier: metrics.BiasReport | None = None

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
            (
                f"Starters-only Spearman (minutes > 0): "
                f"{self.rank_correlation.overall_starters_only:.4f}"
                if self.rank_correlation.overall_starters_only is not None
                else "Starters-only Spearman: not computed (no minutes_col given)"
            ),
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
            f"  calibrated floor for that mass: "
            f"{self.minutes_diagnostics.calibrated_floor_mass_per_scored_row:.4f} / scored row "
            f"(excess attributable to miscalibration: "
            f"{self.minutes_diagnostics.zero_minute_mass_excess:.4f}) — B2: a correctly uncertain "
            "model still assigns points to players who then don't appear, so only the excess is a "
            "defect; the floor itself moves only with discrimination (AUC).",
        ]
        if self.minutes_diagnostics.zero_minute_mass_by_component is not None:
            lines.append(
                "  Zero-minute mass by component (B2 -- the minutes model is well calibrated, so "
                "anything above target is a downstream component not fully gated by availability; "
                "this names it rather than leaving it to be inferred):"
            )
            lines.append(
                "    "
                + self.minutes_diagnostics.zero_minute_mass_by_component.to_string(
                    index=False
                ).replace("\n", "\n    ")
            )
        lines += [
            "",
            f"Clean-sheet MACE (gated, apples-to-apples): "
            f"{self.clean_sheet_calibration.mean_absolute_calibration_error:.4f}",
            f"Minutes 'played at all' MACE: "
            f"{self.minutes_played_calibration.mean_absolute_calibration_error:.4f}",
            f"Minutes 'played 60+' MACE: "
            f"{self.minutes_60_plus_calibration.mean_absolute_calibration_error:.4f}",
            f"Defensive-contribution threshold MACE: "
            f"{self.defensive_contribution_calibration.mean_absolute_calibration_error:.4f}",
        ]
        if self.team_clean_sheet_calibration is not None:
            lines.append(
                "Team-level clean-sheet MACE (ENGINE_IMPROVEMENTS_3.md A.1/B.3 — the quantity "
                "that actually drives DEF rankings, distinct from the gated player-level number "
                f"above): {self.team_clean_sheet_calibration.mean_absolute_calibration_error:.4f}"
            )
        if self.brier_reports:
            lines += [
                "",
                "Brier vs. predicting the constant base rate (ENGINE_IMPROVEMENTS_3.md B.3 — a "
                "reliability curve can look reasonable bin-by-bin while still losing to a "
                "constant in aggregate):",
                *[
                    f"  {name}: brier={r.brier:.4f} constant={r.constant_brier:.4f} "
                    f"beats_constant={r.beats_constant}"
                    for name, r in self.brier_reports.items()
                ],
            ]
        lines += [
            "",
            "Mean calibration (continuous components, all rows -- informational only; dominated "
            "by the minutes model, see the played-only figures below for what the gate reads):",
            *[
                f"  {name}: predicted={report.mean_predicted:.4f} actual={report.mean_actual:.4f} "
                f"relative_gap={report.relative_gap:.2%}"
                for name, report in self.mean_calibrations.items()
            ],
            "",
            "Mean calibration, played rows only (B3 -- retained as a diagnostic; NOT the gate's "
            "instrument since Tier 2.3, it is confounded by selecting on a realised outcome):",
            *(
                [
                    f"  {name}: predicted={report.mean_predicted:.4f} "
                    f"actual={report.mean_actual:.4f} relative_gap={report.relative_gap:.2%}"
                    for name, report in self.mean_calibrations_played.items()
                ]
                or ["  not computed (no minutes column on ground_truth)"]
            ),
            "",
            "Rate calibration at REALISED minutes (Tier 2.3 -- this is what the gate checks. Free "
            "of the minutes model and of the played-rows selection effect, so it measures the rate "
            "model itself):",
            *(
                [
                    f"  {name}: predicted={report.mean_predicted:.4f} "
                    f"actual={report.mean_actual:.4f} relative_gap={report.relative_gap:.2%}"
                    for name, report in self.rate_calibrations.items()
                ]
                or ["  not computed (no minutes column on ground_truth)"]
            ),
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
                "Captaincy hit-rate (stand-in squad, see D.1 docstring — not the gate's own "
                "metric):",
                f"  raw hit-rate: {self.captaincy.raw_hit_rate:.4f} "
                f"(n={len(self.captaincy.per_gameweek)})",
                "  'played as expected' hit-rate: "
                f"{self.captaincy.played_as_expected_hit_rate:.4f}",
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
        if self.bias_by_price_tier is not None:
            lines = lines + [
                "",
                "Bias by price tier (ENGINE_IMPROVEMENTS_3.md B.2 — absolute-effect-floor only, "
                "since a relative floor scaled by the tier's own mean actual is backwards for "
                "the premium tier):",
                self.bias_by_price_tier.by_group.to_string(index=False),
            ]
        if self.decision_set_rank is not None:
            d = self.decision_set_rank
            lines = lines + [
                "",
                f"Shortlist ranking (ENGINE_IMPROVEMENTS_5.md Tier 0.1 — Spearman *within* each "
                f"gameweek's own top {d.top_n} by predicted points, then averaged across "
                f"gameweeks; this is the ordering a manager acts on, and it is a different "
                f"quantity from the pooled Spearman above, which is dominated by who plays):",
                f"  mean Spearman:   {d.mean_spearman:.4f}  (gate needs "
                f">= {gate.DEFAULT_MIN_DECISION_SET_SPEARMAN})",
                f"  median Spearman: {d.median_spearman:.4f}",
                f"  std / share of gameweeks positive: {d.std_spearman:.4f} / "
                f"{d.share_positive:.1%} of {d.n_gameweeks}",
                f"  MAE / bias within the shortlist: {d.mean_absolute_error:.4f} / "
                f"{d.mean_bias:+.4f}",
            ]
        if self.conditional_bias_by_position is not None:
            lines = lines + [
                "",
                "Bias of E[points | plays] among players who actually played 60+ minutes (Tier "
                "2.1 — scored against conditional_expected_points, NOT expected_points. Scoring "
                "the unconditional prediction here reads about -1.0, but an oracle that knows "
                "P(plays) and E[points|plays] exactly scores -1.31 on the same statistic, so that "
                "number measures the act of conditioning on a realised outcome rather than any "
                "defect):",
                self.conditional_bias_by_position.by_group.to_string(index=False),
            ]
            if self.conditional_bias_by_price_tier is not None:
                lines = lines + [
                    "",
                    "  ... same, by price tier:",
                    self.conditional_bias_by_price_tier.by_group.to_string(index=False),
                ]
        if self.coverage is not None:
            lines = ["Sample coverage:", self.coverage.summary(), ""] + lines
        return "\n".join(lines)

    def headline_summary(self) -> dict[str, Any]:
        """A curated, JSON-serializable subset of this report — the Model Performance screen's
        actual data source (BUILD_PLAN 5.2: "prediction vs actual, captaincy hit-rate, component
        calibration"), not a full serialization of every nested report. Several fields
        (``by_bin``/``by_group`` DataFrames, raw bootstrap arrays) aren't JSON-friendly and aren't
        what that screen needs — this is deliberately a headline view, not a dump.
        """
        top_n = {int(row["n"]): float(row["mean_actual"]) for _, row in self.top_n.by_n.iterrows()}
        gate = self.definition_of_done
        return {
            "overall_mae": self.accuracy.overall_mae,
            "overall_rmse": self.accuracy.overall_rmse,
            "pooled_spearman": self.rank_correlation.overall,
            "top_n_mean_actual": top_n,
            "clean_sheet_mace": self.clean_sheet_calibration.mean_absolute_calibration_error,
            "minutes_played_at_all_mace": (
                self.minutes_played_calibration.mean_absolute_calibration_error
            ),
            "minutes_60_plus_mace": (
                self.minutes_60_plus_calibration.mean_absolute_calibration_error
            ),
            "defensive_contribution_mace": (
                self.defensive_contribution_calibration.mean_absolute_calibration_error
            ),
            "mean_calibrations_played": {
                name: {
                    "predicted": report.mean_predicted,
                    "actual": report.mean_actual,
                    "relative_gap": report.relative_gap,
                }
                for name, report in self.mean_calibrations_played.items()
            },
            "captaincy_hit_rate": self.captaincy.raw_hit_rate if self.captaincy else None,
            # Tier 0.1: reported next to `pooled_spearman` deliberately. On its own the pooled
            # figure reads as "the engine ranks players well" when it mostly ranks availability;
            # these two say whether it can be acted on.
            "shortlist_spearman": (
                self.decision_set_rank.mean_spearman if self.decision_set_rank else None
            ),
            # Tier 2.1: the bias of E[points | plays], not of expected_points. See score_season.
            "conditional_played_60_plus_bias": (
                float(self.conditional_bias_by_position.by_group["mean_residual"].mean())
                if self.conditional_bias_by_position is not None
                else None
            ),
            "gate": {
                "beats_baselines": gate.beats_baselines,
                "no_severe_bias": gate.no_severe_bias,
                "no_severe_conditional_bias": gate.no_severe_conditional_bias,
                "calibration_acceptable": (
                    gate.calibration_acceptable and gate.mean_calibration_acceptable
                ),
                "decision_set_ranking_acceptable": gate.decision_set_ranking_acceptable,
                "predictions_logged": gate.predictions_logged,
                "trusted_by_user": gate.trusted_by_user,
                "passed": gate.passed,
            },
        }


# Outfield shape of a real FPL squad (the 15 minus its 2 goalkeepers), used to build the stand-in
# captaincy squad below. Selecting on raw totals with no positional constraint produces a squad no
# real manager would ever field -- see that function's docstring.
DEFAULT_STAND_IN_SQUAD_SHAPE = {DEF: 5, MID: 5, FWD: 3}


def build_stand_in_squad_starting_xi(
    ground_truth: pd.DataFrame,
    squad_size: int = 13,
    starts_col: str = "starts",
    selection_col: str = "total_points",
    shape: Mapping[str, int] | None = None,
) -> dict[int, set[int]]:
    """A fixed, model-independent stand-in "my team" for captaincy backtesting
    (ENGINE_IMPROVEMENTS_2.md D.1 / ENGINE_IMPROVEMENTS.md 3.1's own documented workaround — no
    real historical "my team" exists to backtest against): the best outfield players by
    ``selection_col``, filled to the positional ``shape`` of a real FPL squad, selected **once from
    ground truth alone**, never from engine predictions, to avoid circularity. Each gameweek's
    eligible ("starting XI") set is restricted to whichever of those players actually started that
    gameweek — bench players were never really in contention for the armband (BUILD_PLAN 3.2) — via
    ``starts_col``.

    **Why total points and a positional shape, not raw minutes.** Selecting the highest-minutes
    players with no positional constraint (this function's original behaviour) yields a squad that
    is 7 defenders, 4 midfielders and 2 forwards on real 2025/26 data — defenders and holding
    midfielders are simply the least-rotated players in football, so a pure minutes sort finds them
    first. No FPL manager fields that squad, and nobody captains it: the resulting "eligible
    options" are dominated by low-ceiling defenders whose gameweek scores are near-indistinguishable
    noise, which depresses the measured hit-rate for reasons that have nothing to do with the
    engine. On the same predictions, the same 35 gameweeks and the same scoring code, the measured
    raw hit-rate moves 0.171 -> 0.286 purely by changing this selection, and the engine's captain
    earns 4.83 -> 7.31 mean actual points. Its lift over a random eligible pick is a steady
    2.2x-3.5x under *every* variant tried, which is the part that actually reflects the engine.

    Both bases are equally hindsight-informed (season-end totals aren't knowable at GW5 either), so
    this is not a leakage trade — it is the same illustrative device, shaped like a squad someone
    might really own.

    Excludes goalkeepers when a ``position`` column is present (ENGINE_IMPROVEMENTS_3.md D.1) —
    no real manager captains their keeper.

    Falls back to a flat top-``squad_size`` selection when no ``position`` column is available (the
    shape is unenforceable without one). This is the exact caveat ENGINE_IMPROVEMENTS.md 3.1 flags:
    a stand-in squad, not a real manager's history, so the resulting hit-rate is illustrative rather
    than a claim about any specific real team's captaincy record.
    """
    if "position" not in ground_truth.columns:
        totals = ground_truth.groupby("player_id")[selection_col].sum()
        squad_ids = set(totals.nlargest(squad_size).index)
    else:
        outfield = ground_truth[ground_truth["position"] != GK]
        totals = outfield.groupby("player_id").agg(
            position=("position", "first"), value=(selection_col, "sum")
        )
        shape = DEFAULT_STAND_IN_SQUAD_SHAPE if shape is None else shape
        squad_ids = set()
        for position, count in shape.items():
            in_position = totals[totals["position"] == position]
            squad_ids |= set(in_position.nlargest(count, "value").index)

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

    dc_join_cols = ["player_id", "gameweek", "defensive_contribution"]
    if "dc_data_available" in ground_truth.columns:
        dc_join_cols.append("dc_data_available")
    dc_rows = predictions.merge(
        ground_truth[dc_join_cols].rename(
            columns={"defensive_contribution": "actual_defensive_contribution"}
        ),
        on=["player_id", "gameweek"],
    )
    dc_rows = dc_rows[dc_rows["p_clears_threshold"].notna()]
    if "dc_data_available" in dc_rows.columns:
        # Multi-season Phase 2: a season lacking defensive contribution's raw archive columns
        # (2020/21-2024/25 in this data source) gets a neutral 0.0 placeholder rate/outcome
        # upstream (engineer_features) so nothing crashes — but that placeholder must never be
        # scored as if it were real calibration signal.
        dc_rows = dc_rows[dc_rows["dc_data_available"]]
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
    # ENGINE_IMPROVEMENTS_5.md Tier 2.3: the rate model scored at each row's REALISED minutes.
    #
    # This is the instrument the two views below cannot provide. The all-rows figure is dominated by
    # the minutes model; the played-rows figure is confounded by selection, because
    # `expected_goals`/`expected_assists` are unconditional expectations and restricting to rows
    # where the player played selects the branch on which they were always going to look low. That
    # confound does not merely add noise, it flips conclusions: played-rows reported goals 5.7% over
    # and assists 22.2% under, while this view reports goals 18% over and assists 12% under. The
    # all-rows view agrees with this one (goals 24% over, assists 8% under), not with played-rows.
    rate_calibrations = None
    if "minutes" in ground_truth.columns:
        rate_rows = mean_rows.merge(
            ground_truth[["player_id", "gameweek", "minutes"]], on=["player_id", "gameweek"]
        )
        if not rate_rows.empty:
            rate_calibrations = {
                "goals": metrics.rate_calibration_at_realised_minutes(
                    rate_rows["expected_goals"],
                    rate_rows["expected_minutes"],
                    rate_rows["minutes"],
                    rate_rows["actual_goals_scored"],
                ),
                "assists": metrics.rate_calibration_at_realised_minutes(
                    rate_rows["expected_assists"],
                    rate_rows["expected_minutes"],
                    rate_rows["minutes"],
                    rate_rows["actual_assists"],
                ),
            }

    # B3: the same components scored on rows where the player actually appeared.
    #
    # Retained as a diagnostic, but NOT the gate's instrument any more (Tier 2.3): see the
    # selection-effect note immediately above for why it disagreed with both other views.
    if "minutes" in ground_truth.columns:
        played_rows = mean_rows.merge(
            ground_truth[["player_id", "gameweek", "minutes"]], on=["player_id", "gameweek"]
        )
        played_rows = played_rows[played_rows["minutes"] > 0]
        if len(played_rows) >= 2:
            mean_calibrations_played = {
                "goals": metrics.mean_calibration(
                    played_rows["expected_goals"], played_rows["actual_goals_scored"]
                ),
                "assists": metrics.mean_calibration(
                    played_rows["expected_assists"], played_rows["actual_assists"]
                ),
                "bonus": metrics.mean_calibration(
                    played_rows["expected_bonus"], played_rows["actual_bonus"]
                ),
            }
        else:
            mean_calibrations_played = {}
    else:
        mean_calibrations_played = {}
    # Phase 3: saves had no calibration check at all -- every other component (goals, assists,
    # bonus, clean sheets, DC) has one. `expected_saves` is NaN for outfield players (mirrors
    # `p_clears_threshold`'s own GK-only split), so filter to GK rows before scoring.
    if "expected_saves" in predictions.columns and "saves" in ground_truth.columns:
        saves_rows = predictions.merge(
            ground_truth[["player_id", "gameweek", "saves"]].rename(
                columns={"saves": "actual_saves"}
            ),
            on=["player_id", "gameweek"],
        )
        saves_rows = saves_rows[saves_rows["expected_saves"].notna()]
        if len(saves_rows) >= 2:
            mean_calibrations["saves"] = metrics.mean_calibration(
                saves_rows["expected_saves"], saves_rows["actual_saves"]
            )

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

    # --- B.2: price-tier bias, alongside position — BUILD_PLAN 3.2 names both, but only position
    # was ever wired in. min_relative_effect=0.0 (absolute floor only): a floor that scales with
    # the tier's own mean actual is backwards for the premium tier, where a large mean actual makes
    # a large absolute bias easier, not harder, to clear.
    bias_by_price_tier = None
    if "value" in ground_truth.columns:
        price_tier_actuals = ground_truth.copy()
        price_tier_actuals["price_tier"] = pd.qcut(
            price_tier_actuals["value"], 5, duplicates="drop"
        ).astype(str)
        bias_by_price_tier = metrics.bias_by_group(
            predictions, price_tier_actuals, group_col="price_tier", min_relative_effect=0.0
        )

    # --- B.3: team-level clean-sheet calibration — the quantity that actually drives DEF
    # rankings, distinct from the gated (team_prob * p_60_plus) player-level number above, which
    # compresses toward zero and can look well-calibrated while the underlying team probability is
    # not (ENGINE_IMPROVEMENTS_3.md A.1). Reads the real fixture scoreline, not
    # `max(goals_conceded == 0)` across a team's players — ENGINE_IMPROVEMENTS.md Correction 1's
    # own warning about exactly that measurement bug.
    team_clean_sheet_calibration = None
    team_clean_sheet_brier = None
    _team_cs_cols = {"team", "was_home", "team_h_score", "team_a_score"}
    if _team_cs_cols.issubset(ground_truth.columns):
        team_rows = predictions.merge(
            ground_truth[["player_id", "gameweek", *_team_cs_cols]], on=["player_id", "gameweek"]
        ).drop_duplicates(subset=["team", "gameweek"])
        team_conceded = np.where(
            team_rows["was_home"], team_rows["team_a_score"], team_rows["team_h_score"]
        )
        team_kept_clean_sheet = (team_conceded == 0).astype(float)
        team_clean_sheet_calibration = metrics.component_calibration(
            team_rows["clean_sheet_probability"], team_kept_clean_sheet
        )
        team_clean_sheet_brier = metrics.brier_vs_constant(
            team_rows["clean_sheet_probability"], team_kept_clean_sheet
        )

    # --- B.3: Brier-vs-constant for every probability component — a reliability curve can look
    # reasonable bin-by-bin while the component is still worse than predicting the base rate, per
    # the team-level clean-sheet finding above.
    brier_reports = {
        "clean_sheet_gated": metrics.brier_vs_constant(
            calibration_rows["player_clean_sheet_probability"],
            calibration_rows["gated_clean_sheet"],
        ),
        "minutes_played_at_all": metrics.brier_vs_constant(
            1.0 - minutes_rows["p_zero"], (minutes_rows["minutes"] > 0).astype(float)
        ),
        "minutes_60_plus": metrics.brier_vs_constant(
            minutes_rows["p_60_plus"], (minutes_rows["minutes"] >= 60).astype(float)
        ),
        "defensive_contribution": metrics.brier_vs_constant(
            dc_rows["p_clears_threshold"], dc_actual_clears
        ),
    }
    if team_clean_sheet_brier is not None:
        brier_reports["team_clean_sheet"] = team_clean_sheet_brier

    calibration_reports_for_gate = {
        "clean_sheet": clean_sheet_calibration,
        "minutes_played_at_all": minutes_played_calibration,
        "minutes_60_plus": minutes_60_plus_calibration,
        "defensive_contribution": defensive_contribution_calibration,
    }
    if team_clean_sheet_calibration is not None:
        calibration_reports_for_gate["team_clean_sheet"] = team_clean_sheet_calibration

    bias_reports_for_gate = {"position": bias}
    if bias_by_price_tier is not None:
        bias_reports_for_gate["price_tier"] = bias_by_price_tier

    # --- ENGINE_IMPROVEMENTS_5.md Tier 0.1 / 2.1: the quantities a manager actually acts on ------
    # Within-shortlist ranking, and the bias of E[points | plays] on players who did play.
    #
    # Tier 2.1 correction: this check originally scored `expected_points` on played-60+ rows and
    # read -0.990, which was reported as a large hidden bias. It is not. `expected_points` is
    # P(plays) * E[points | plays], so selecting only rows where the player *did* play selects the
    # branch on which an unconditional expectation was always going to look low. A simulated model
    # that knows P(plays) and E[points | plays] exactly scores -1.31 on the same statistic, i.e.
    # worse than the engine, so the number measures the conditioning, not a defect, and no
    # correctly-calibrated model could ever pass it.
    #
    # The meaningful check is the *conditional* prediction against the same rows, which is what
    # `conditional_expected_points` (Tier 2.1) is for. On this walk-forward it reads -0.088 against
    # an actual mean of 3.87, so calibration on players who play is in fact good. Falls back to a
    # no-op when the column is absent (a frame produced before Tier 2.1) rather than silently
    # reinstating the meaningless version.
    conditional_bias_reports_for_gate: dict[str, metrics.BiasReport] = {}
    conditional_bias_by_position = None
    conditional_bias_by_price_tier = None
    if "conditional_expected_points" in predictions.columns:
        conditional_predictions = predictions.assign(
            expected_points=predictions["conditional_expected_points"]
        )
        conditional_bias_by_position = metrics.bias_by_group(
            conditional_predictions, ground_truth, group_col="position", minutes_col="minutes"
        )
        conditional_bias_reports_for_gate["position_played_60_plus"] = conditional_bias_by_position
        if "value" in ground_truth.columns:
            conditional_bias_by_price_tier = metrics.bias_by_group(
                conditional_predictions,
                price_tier_actuals,
                group_col="price_tier",
                min_relative_effect=0.0,
                minutes_col="minutes",
            )
            conditional_bias_reports_for_gate["price_tier_played_60_plus"] = (
                conditional_bias_by_price_tier
            )

    decision_set_rank = metrics.decision_set_rank_correlation(
        predictions, ground_truth, top_n=gate.DEFAULT_DECISION_SET_TOP_N
    )

    definition_of_done = gate.evaluate_definition_of_done(
        baseline_results=baseline_results,
        bias_reports=bias_reports_for_gate,
        calibration_reports=calibration_reports_for_gate,
        predictions_logged=True,
        trusted_by_user=False,
        # Tier 2.3: gate on the rate-level view, which is free of both the minutes model and the
        # played-rows selection effect. Falls back to the played-rows view when ground_truth
        # carries no minutes column, so a minimal frame still gets *some* component check.
        mean_calibration_reports=(
            rate_calibrations if rate_calibrations is not None else mean_calibrations_played
        ),
        decision_set_rank_report=decision_set_rank,
        conditional_bias_reports=conditional_bias_reports_for_gate,
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
        mean_calibrations_played=mean_calibrations_played,
        baseline_results=baseline_results,
        definition_of_done=definition_of_done,
        coverage=coverage,
        captaincy=captaincy,
        floor_ceiling_coverage=floor_ceiling_cov,
        big_haul_calibration=big_haul_calibration,
        bias_by_price_tier=bias_by_price_tier,
        team_clean_sheet_calibration=team_clean_sheet_calibration,
        brier_reports=brier_reports,
        rate_calibrations=rate_calibrations or {},
        decision_set_rank=decision_set_rank,
        conditional_bias_by_position=conditional_bias_by_position,
        conditional_bias_by_price_tier=conditional_bias_by_price_tier,
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
    matched_mask = pd.Series(
        per_player_minutes.index.isin(matched_ids), index=per_player_minutes.index
    )
    unmatched_significant = per_player_minutes[(~matched_mask) & (per_player_minutes > 450)].index
    points_excluded = float(per_player_points.reindex(unmatched_significant).fillna(0.0).sum())
    points_excluded_share = (
        points_excluded / total_season_points if total_season_points else float("nan")
    )

    # Crosswalk coverage Phase 1: the concrete "who" behind points_excluded_share, not just the
    # aggregate share — see UnmatchedSignificantPlayer's own docstring.
    name_by_id = outfield.drop_duplicates("element").set_index("element")["name"]
    unmatched_significant_players = tuple(
        UnmatchedSignificantPlayer(
            fpl_id=int(fpl_id),
            name=str(name_by_id.get(fpl_id, "<unknown>")),
            minutes=float(per_player_minutes.loc[fpl_id]),
            points=float(per_player_points.loc[fpl_id]),
        )
        for fpl_id in unmatched_significant
    )

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
        crosswalk_match_rate=(
            float(matched_mask.mean()) if len(per_player_minutes) else float("nan")
        ),
        points_excluded_share=points_excluded_share,
        unmatched_significant_players=unmatched_significant_players,
        drop_reasons=(
            pd.DataFrame(records) if (records := engineered.attrs.get("drop_reasons")) else None
        ),
    )


@dataclass(frozen=True)
class SeasonBacktestData:
    """Everything one season's walk-forward run produces, before scoring — the shared core
    :func:`run_backtest` and :func:`run_multi_season_backtest` both need (Phase 2 of the
    multi-season plan). Split out so the multi-season driver can pool several seasons'
    ``predictions``/``ground_truth``/``player_rates`` and call :func:`score_season` **once** on the
    combined frame, rather than duplicating the fetch/engineer/walk-forward logic itself."""

    season_start_year: int
    predictions: pd.DataFrame
    ground_truth: pd.DataFrame
    player_rates: pd.DataFrame
    coverage: CoverageReport
    simulation_predictions: pd.DataFrame | None = None


def _prepare_season_backtest_data(
    season_start_year: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_training_gameweeks: int = 3,
    refresh: bool = False,
    min_crosswalk_minutes_share: float | None = 0.85,
    run_simulation: bool = False,
    simulation_n_runs: int = 200,
) -> SeasonBacktestData:
    """Fetch, engineer, and walk-forward one season — everything :func:`run_backtest` needs before
    handing off to :func:`score_season`. See :func:`run_backtest` for parameter docs."""
    with httpx.Client(timeout=30.0) as http_client, UnderstatClient() as understat:
        merged_gw = fetch_vaastav_merged_gw(season_start_year, cache_dir, http_client, refresh)
        teams = fetch_vaastav_teams(season_start_year, cache_dir, http_client, refresh)
        league_data = fetch_understat_league_data_raw(
            season_start_year, cache_dir, understat, refresh
        )
        # A.4: team rates draw on this season plus N_PRIOR_SEASONS_FOR_TEAM_RATES before it — the
        # crosswalk/player-id lookups below still use `league_data` (this season alone), since the
        # player pool to project is always this season's own squad.
        multi_season_league_data = fetch_understat_multi_season_league_data(
            season_start_year, cache_dir, understat, refresh
        )
        multi_season_teams_history = pd.concat(
            [
                league_data_to_dataframes(data)["teams_history"]
                for data in multi_season_league_data.values()
            ],
            ignore_index=True,
        )
        team_histories = build_team_rate_histories(multi_season_teams_history)
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
            per_player_minutes,
            matched_ids,
            min_share=min_crosswalk_minutes_share,
            label="season minutes",
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
            "saves",  # Phase 3: saves mean-calibration check
            "starts",
            "value",  # B.2: price-tier bias
            "team",  # B.3: team-level clean-sheet calibration
            "was_home",
            "team_h_score",
            "team_a_score",
        ]
    ].copy()
    # Multi-season Phase 2: whether this season's archive actually carries DC's raw inputs — see
    # engineer_features' own dc_data_available note. A constant column (not per-row) since this is
    # a whole-season data-availability fact, not something that varies player to player.
    ground_truth["dc_data_available"] = bool(engineered.attrs.get("dc_data_available", True))
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

    return SeasonBacktestData(
        season_start_year=season_start_year,
        predictions=result.predictions,
        ground_truth=ground_truth,
        player_rates=player_rates,
        coverage=coverage,
        simulation_predictions=simulation_predictions,
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

    See :func:`run_multi_season_backtest` to run and pool this across several seasons at once.
    """
    data = _prepare_season_backtest_data(
        season_start_year,
        cache_dir,
        min_training_gameweeks,
        refresh,
        min_crosswalk_minutes_share,
        run_simulation,
        simulation_n_runs,
    )
    report = score_season(
        data.predictions,
        data.ground_truth,
        coverage=data.coverage,
        player_rates=data.player_rates,
        simulation_predictions=data.simulation_predictions,
    )
    return data.predictions, report


# Multiplier applied to a season's own gameweek number to build a globally-unique composite key
# across pooled seasons (ENGINE_IMPROVEMENTS_3.md multi-season Phase 2) — e.g. season 2024's GW5
# becomes 202405, season 2025's GW5 becomes 202505, so a plain `groupby("gameweek")` anywhere in
# `backtest.metrics`/`score_season` never accidentally merges two different seasons' same-numbered
# gameweek. No season is expected to run anywhere near 100 gameweeks, so this can't collide.
_SEASON_GAMEWEEK_MULTIPLIER = 100


def _composite_gameweek(season_start_year: int, gameweek: pd.Series) -> pd.Series:
    return season_start_year * _SEASON_GAMEWEEK_MULTIPLIER + gameweek


@dataclass(frozen=True)
class MultiSeasonReport:
    """Pooled backtest results across several independently-run seasons (ENGINE_IMPROVEMENTS_3.md
    multi-season Phase 2). ``per_season`` keeps each season's own natural-gameweek-numbered
    :class:`SeasonReport` (so one bad season doesn't hide inside a pooled average unnoticed);
    ``pooled`` is a single :class:`SeasonReport` computed by re-keying every season's own gameweek
    number to a globally-unique composite (see :func:`_composite_gameweek`) and concatenating —
    every metric ``score_season`` computes (MAE, top-N, calibration, captaincy hit-rate, ...)
    therefore reuses that same, already-tested scoring code unchanged, just over a larger pooled
    sample. Defensive-contribution calibration within ``pooled`` still only ever reflects seasons
    with real DC archive data (see ``dc_data_available`` in :func:`_prepare_season_backtest_data`)
    — pooling more seasons doesn't manufacture DC history that was never recorded.

    The stand-in captaincy squad (:func:`build_stand_in_squad_starting_xi`) is built **once**, from
    the pooled ground truth across every season, not freshly per season — the same top-(by total
    minutes) outfield players it already illustrates real squads with, just now spanning multiple
    seasons. This is a known simplification: a real manager's squad changes season to season, which
    this pooled selection doesn't model — consistent with that function's own "illustrative, not a
    real manager's history" caveat.
    """

    per_season: dict[int, SeasonReport]
    pooled: SeasonReport


def _pool_season_backtest_data(per_season_data: dict[int, SeasonBacktestData]) -> MultiSeasonReport:
    """The actual pooling/scoring logic behind :func:`run_multi_season_backtest`, split out as a
    pure function (no fetching) so it's testable against synthetic per-season data without a real
    network call — see :func:`run_multi_season_backtest` for the full docstring."""
    per_season_reports = {
        season: score_season(
            data.predictions,
            data.ground_truth,
            coverage=data.coverage,
            player_rates=data.player_rates,
        )
        for season, data in per_season_data.items()
    }

    pooled_predictions = pd.concat(
        [
            data.predictions.assign(
                gameweek=_composite_gameweek(season, data.predictions["gameweek"])
            )
            for season, data in per_season_data.items()
        ],
        ignore_index=True,
    )
    pooled_ground_truth = pd.concat(
        [
            data.ground_truth.assign(
                gameweek=_composite_gameweek(season, data.ground_truth["gameweek"])
            )
            for season, data in per_season_data.items()
        ],
        ignore_index=True,
    )
    pooled_player_rates = pd.concat(
        [
            data.player_rates.assign(
                gameweek=_composite_gameweek(season, data.player_rates["gameweek"])
            )
            for season, data in per_season_data.items()
        ],
        ignore_index=True,
    )
    pooled_report = score_season(
        pooled_predictions, pooled_ground_truth, player_rates=pooled_player_rates
    )
    return MultiSeasonReport(per_season=per_season_reports, pooled=pooled_report)


def run_multi_season_backtest(
    season_start_years: list[int],
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_training_gameweeks: int = 3,
    refresh: bool = False,
    min_crosswalk_minutes_share: float | None = 0.85,
) -> MultiSeasonReport:
    """Run :func:`run_backtest`'s own walk-forward pipeline independently for each season in
    ``season_start_years`` (ENGINE_IMPROVEMENTS_3.md multi-season Phase 2 — each season scores its
    own cold start against its own real prior-season Understat history via
    :func:`fetch_understat_multi_season_league_data`/:func:`fetch_understat_player_histories`,
    rather than one continuous timeline spanning season boundaries), then pool every season's
    predictions/ground_truth/player_rates into one combined :class:`SeasonReport` via a single
    call to :func:`score_season` (see :func:`_pool_season_backtest_data`) — see
    :data:`_SEASON_GAMEWEEK_MULTIPLIER` for how gameweek numbers stay disambiguated across seasons
    in that pooled call.

    Simulation (``run_simulation``) is intentionally not offered here — it's expensive per season
    already; run it per season via :func:`run_backtest` directly if needed.
    """
    per_season_data = {
        season: _prepare_season_backtest_data(
            season, cache_dir, min_training_gameweeks, refresh, min_crosswalk_minutes_share
        )
        for season in season_start_years
    }
    return _pool_season_backtest_data(per_season_data)


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
        # A4: the Model Performance screen's actual data source -- see headline_summary's own
        # docstring for why this is a curated subset, not every nested report.
        args.report_path.with_suffix(".json").write_text(
            json.dumps(report.headline_summary(), indent=2)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
