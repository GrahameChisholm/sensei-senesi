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
    build_crosswalk,
    fetch_fpl_id_list,
    understat_players_from_league_data,
)
from engine.data.understat_client import (
    UnderstatClient,
    league_data_to_dataframes,
    player_data_to_dataframe,
)
from engine.models.bonus import BonusModel
from engine.models.clean_sheets import (
    clean_sheet_probability,
    fit_dixon_coles_rho,
    team_expected_goals_rate,
)
from engine.models.defensive_contribution import fit_overdispersion
from engine.models.goals import fit_penalty_conversion_rates, realized_penalty_goals
from engine.models.minutes import FEATURE_COLUMNS as MINUTES_FEATURE_COLUMNS
from engine.models.minutes import MinutesModel, encode_status
from engine.models.saves import fit_away_shot_multiplier, fit_save_conversion_rate
from engine.pipeline import FittedConstants, project_gameweek_pool
from engine.rates import ewma_rate_asof, latest_ewma_rate
from engine.scoring import DEF, FWD, MID, POSITIONS

VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
DEFAULT_CACHE_DIR = Path("data_store/season_cache")
DEFAULT_HALFLIFE = 10.0

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
    "engineer_features",
    "fit_fn",
    "make_predict_fn",
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
    """
    cache_path = cache_dir / "crosswalk" / f"{season_start_year}.parquet"
    if cache_path.exists() and not refresh:
        records = pd.read_parquet(cache_path).to_dict("records")
        return [CrosswalkEntry(**record) for record in records]
    understat_players = understat_players_from_league_data(league_data)
    fpl_id_by_name = fetch_fpl_id_list(season_start_year, client)
    entries = build_crosswalk(understat_players, fpl_id_by_name, strict=False)
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
    """FPL id -> that player's Understat match history for ``season_start_year``, chronologically
    sorted (oldest first) — the point-in-time-safe order :mod:`engine.rates` requires. Understat's
    ``get_player_data`` has no bulk/multi-player form (the same one-request-per-player limitation
    ``FPLClient.iter_element_summaries`` documents), so this is one request per matched player,
    cached individually so an interrupted run doesn't lose earlier progress.
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
        season_matches = df[df["season"].astype(str) == str(season_start_year)].copy()
        if season_matches.empty:
            continue
        for col in ("time", "npxG", "xA", "goals", "npg"):
            season_matches[col] = season_matches[col].astype(float)
        season_matches["date"] = pd.to_datetime(season_matches["date"], utc=True)
        histories[entry.fpl_id] = season_matches.sort_values("date").reset_index(drop=True)
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

    empty = pd.DataFrame(columns=["date", "xG", "xGA", "minutes"])
    fixtures["team_xg_per_90"] = [
        _team_rate_asof(team_histories.get(team, empty), "xG", kickoff)
        for team, kickoff in zip(fixtures["team"], fixtures["kickoff_time"], strict=True)
    ]
    fixtures["team_xga_per_90"] = [
        _team_rate_asof(team_histories.get(team, empty), "xGA", kickoff)
        for team, kickoff in zip(fixtures["team"], fixtures["kickoff_time"], strict=True)
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
    return gw.dropna(subset=required).reset_index(drop=True)


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
# Scoring / gate
# =================================================================================================


@dataclass(frozen=True)
class SeasonReport:
    accuracy: metrics.AccuracyReport
    bias_by_position: metrics.BiasReport
    top_n: metrics.TopNReport
    rank_correlation: metrics.RankCorrelationReport
    clean_sheet_calibration: metrics.CalibrationReport
    baseline_results: dict[str, baselines.PairedBootstrapResult]
    definition_of_done: gate.DefinitionOfDoneReport

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
            f"Overall Spearman (predicted vs actual): {self.rank_correlation.overall:.4f}",
            "",
            f"Clean-sheet MACE (gated, apples-to-apples): "
            f"{self.clean_sheet_calibration.mean_absolute_calibration_error:.4f}",
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
        return "\n".join(lines)


def score_season(predictions: pd.DataFrame, ground_truth: pd.DataFrame) -> SeasonReport:
    accuracy = metrics.player_accuracy(predictions, ground_truth)
    bias = metrics.bias_by_group(predictions, ground_truth, group_col="position")
    top_n = metrics.top_n_mean_actual(predictions, ground_truth)
    rank_corr = metrics.rank_correlation(
        predictions, ground_truth, group_col="position", minutes_col="minutes"
    )

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

    engine_actuals = predictions.merge(
        ground_truth[["player_id", "gameweek", "total_points"]], on=["player_id", "gameweek"]
    )
    median_value = baselines.training_median(ground_truth)
    constant_preds = baselines.constant_predictions(
        predictions[["player_id", "position", "gameweek"]], median_value
    )
    naive_preds = baselines.naive_form_predictions(ground_truth)

    baseline_results: dict[str, baselines.PairedBootstrapResult] = {}
    for name, baseline_preds in (("constant_median", constant_preds), ("naive_form", naive_preds)):
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
        baseline_results[name] = baselines.paired_bootstrap_test(engine_err, baseline_err)

    definition_of_done = gate.evaluate_definition_of_done(
        baseline_results=baseline_results,
        bias_reports={"position": bias},
        calibration_reports={"clean_sheet": clean_sheet_calibration},
        predictions_logged=True,
        trusted_by_user=False,
    )
    return SeasonReport(
        accuracy=accuracy,
        bias_by_position=bias,
        top_n=top_n,
        rank_correlation=rank_corr,
        clean_sheet_calibration=clean_sheet_calibration,
        baseline_results=baseline_results,
        definition_of_done=definition_of_done,
    )


# =================================================================================================
# Top-level orchestration + CLI
# =================================================================================================


def run_backtest(
    season_start_year: int,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    min_training_gameweeks: int = 3,
    refresh: bool = False,
) -> tuple[pd.DataFrame, SeasonReport]:
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

    gameweeks = sorted(engineered["gameweek"].unique())
    predict_fn = make_predict_fn(engineered)
    result = run_walk_forward(
        gameweeks, engineered, fit_fn, predict_fn, min_training_gameweeks=min_training_gameweeks
    )

    ground_truth = engineered[
        ["player_id", "gameweek", "position", "total_points", "minutes", "clean_sheets"]
    ]
    report = score_season(result.predictions, ground_truth)
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
    args = parser.parse_args(argv)

    predictions, report = run_backtest(
        args.season, args.cache_dir, args.min_training_gameweeks, args.refresh
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
