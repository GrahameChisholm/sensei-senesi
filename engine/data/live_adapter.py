"""Live-snapshot → ``backtest.run_season.engineer_features`` adapter (A1).

The blocker this module resolves, stated precisely: ``engineer_features(merged_gw, teams,
team_histories, player_histories)`` wants a per-gameweek history frame of **already-played**
gameweeks, built from the vaastav archive. ``engine.data.snapshots`` (via
``engine.data.ingest.capture_current_gameweek``) captures a **current-state** view instead: FPL
bootstrap, Understat league data, and (since A1) two per-gameweek history sources —
``fpl_element_summaries`` (one player-gameweek history row per element, the live equivalent of
vaastav's ``merged_gw.csv``) and ``understat_player_histories`` (one player-match row per
crosswalk-matched player, the live equivalent of the backtest driver's own
``fetch_understat_player_histories``). This module maps those four snapshot sources into the exact
shape ``engineer_features`` expects, plus one synthesized row per player for the **target**
gameweek — the one that hasn't been played yet, which is the whole point of calling this live
rather than replaying history.

**Point-in-time discipline.** Every column built for the target gameweek's synthesized rows must
be derivable from data timestamped strictly before its deadline — real fixtures (known in advance),
real current price/ownership (the live equivalent of BUILD_PLAN 2.1's crowd features), and real
current ``status``/``chance_of_playing_next_round`` (the strongest injury signal the backtest could
never reconstruct). Outcome columns (minutes, goals, bonus, ...) are never known for that row and
are filled with a neutral placeholder (0) — safe only because ``engineer_features`` computes every
per-player feature via ``.shift(1)``/point-in-time EWMA helpers that structurally exclude a row's
own outcome values from its own features; the placeholder is written but never read.

**Known caveat, not yet verified against live data (ENGINE_IMPROVEMENTS.md's own crowd-feature
caveat, restated for the live path):** past gameweeks' ``selected``/``transfers_*`` come straight
from ``fpl_element_summaries`` history rows, in the same raw-count units vaastav's archive uses —
no conversion needed. The **target** gameweek's current ownership has no raw-count field in FPL's
live bootstrap payload, only ``selected_by_percent`` — converted here via ``total_managers``, a
caller-supplied estimate of the live playing population. Confirm this conversion once against a
real live pull before trusting it (see :data:`DEFAULT_TOTAL_MANAGERS`).

**A real gap this module's own test suite couldn't have caught, found by an actual live pull
against the 2026/27 pre-season Understat endpoint (A6):** ``getLeagueData/EPL/2026`` returns
**zero** players and **zero** teams before the season's first ball is kicked — Understat only
populates a season's data once matches exist. A live snapshot's own ``understat`` source is
therefore empty at exactly the moment ``snapshot_to_feature_inputs`` is first called for real
(GW1, pre-deadline) — and confirmed by testing, the failure mode is worse than a silently thin
target gameweek: with every ``team_xg_per_90``/``opponent_xg_per_90`` NaN, ``engineer_features``'
dropna empties the *training* rows too, not just the target row, so
``backtest.run_season.fit_fn`` crashes outright trying to fit sklearn models on zero samples.
``backtest.run_season`` already solved the equivalent backtest-time cold start
(ENGINE_IMPROVEMENTS_3.md A.4:
:func:`~backtest.run_season.fetch_understat_multi_season_league_data`, concatenating
``N_PRIOR_SEASONS_FOR_TEAM_RATES`` seasons before building team histories) — this
module now reuses exactly that for the seasons *before* the current one, concatenated with the
current season's own live (point-in-time) team history from the snapshot. The live snapshot's own
data is never substituted for by a prior season, only supplemented — once real fixtures
accumulate this season, its own rows dominate the EWMA as designed. This does **not** fix the
narrower, deeper case of a season's true opening gameweek, where *no player* has any
this-season minutes/starts history at all yet either (``training_history`` for GW1 is empty
regardless of team rates) — carrying player-level history across seasons too is real, separate,
larger follow-up work, not attempted here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from engine.data.snapshots import DEFAULT_BASE_DIR, load_snapshot_tables
from engine.data.understat_client import UnderstatClient, league_data_to_dataframes
from engine.scoring import ELEMENT_TYPE_TO_POSITION

__all__ = [
    "DEFAULT_TOTAL_MANAGERS",
    "MERGED_GW_COLUMNS",
    "FeatureInputs",
    "build_merged_gw",
    "build_player_histories_from_live_snapshot",
    "snapshot_to_feature_inputs",
]

# FPL bootstrap's live ``elements`` payload carries ``selected_by_percent`` (a percentage), not
# the raw manager count vaastav's archive (and FPL's own element-summary history rows) carry as
# ``selected``. Converting one to the other needs an estimate of the total live playing
# population — bootstrap-static's own ``total_players`` top-level field has this exactly, when
# present; this default is a rough fallback for a caller that hasn't threaded it through. UNVERIFIED
# against a real live pull — see this module's own docstring.
DEFAULT_TOTAL_MANAGERS = 11_000_000.0

# Outcome/outcome-derived columns the target gameweek's synthesized rows can't know yet. Filled
# with 0 -- safe only because engineer_features never reads a row's own value for these when
# computing that same row's features (every per-player rate is a lagged/point-in-time EWMA that
# structurally excludes the current row). See this module's own docstring.
_UNKNOWN_OUTCOME_COLUMNS = [
    "minutes",
    "starts",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "defensive_contribution",
    "own_goals",
    "yellow_cards",
    "red_cards",
    "saves",
    "bps",
    "bonus",
    "penalties_missed",
]

# vaastav/element-summary-history column name -> engineer_features' expected name.
_HISTORY_RENAME = {"element": "player_id", "round": "GW"}

MERGED_GW_COLUMNS = [
    "player_id",
    "GW",
    "position",
    "team",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "team_h_score",
    "team_a_score",
    "value",
    "selected",
    "transfers_out",
    "transfers_balance",
    *_UNKNOWN_OUTCOME_COLUMNS,
]


@dataclass(frozen=True)
class FeatureInputs:
    """The four positional arguments ``backtest.run_season.engineer_features`` takes, assembled
    from one live snapshot."""

    merged_gw: pd.DataFrame
    teams: pd.DataFrame
    team_histories: dict[str, pd.DataFrame]
    player_histories: dict[int, pd.DataFrame]


def _position_by_element(elements: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.id): ELEMENT_TYPE_TO_POSITION[int(row.element_type)]
        for row in elements.itertuples()
    }


def _team_name_by_id(teams: pd.DataFrame) -> dict[int, str]:
    return {int(row.id): row.name for row in teams.itertuples()}


def _own_team_name_by_element(
    elements: pd.DataFrame, team_name_by_id: dict[int, str]
) -> dict[int, str]:
    return {int(row.id): team_name_by_id[int(row.team)] for row in elements.itertuples()}


def _played_rows_from_element_summaries(
    histories: pd.DataFrame,
    position_by_element: dict[int, str],
    own_team_name_by_element: dict[int, str],
    gameweek: int,
) -> pd.DataFrame:
    """Every already-finished gameweek's real history row, renamed and joined onto
    position/team — the live equivalent of a season's worth of vaastav ``merged_gw.csv`` rows for
    every player who has appeared at least once. Defensively excludes any row at or after the
    target gameweek (should never occur pre-deadline; not assumed) — see this module's own
    docstring on point-in-time discipline.
    """
    if histories.empty:
        return pd.DataFrame(columns=MERGED_GW_COLUMNS)

    played = histories.rename(columns=_HISTORY_RENAME).copy()
    played = played[played["GW"] < gameweek]
    if played.empty:
        return pd.DataFrame(columns=MERGED_GW_COLUMNS)

    played["position"] = played["player_id"].map(position_by_element)
    played["team"] = played["player_id"].map(own_team_name_by_element)
    for col in _UNKNOWN_OUTCOME_COLUMNS:
        if col not in played.columns:
            played[col] = 0
    for col in ("value", "selected", "transfers_out", "transfers_balance"):
        if col not in played.columns:
            played[col] = 0
    missing = [c for c in MERGED_GW_COLUMNS if c not in played.columns]
    if missing:
        raise ValueError(
            f"fpl_element_summaries history rows are missing expected column(s): {missing} — "
            "check the real FPL element-summary payload shape against this adapter's assumptions"
        )
    return played[MERGED_GW_COLUMNS]


def _target_gameweek_rows(
    elements: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    gameweek: int,
    position_by_element: dict[int, str],
    own_team_name_by_element: dict[int, str],
    team_name_by_id: dict[int, str],
    total_managers: float,
) -> pd.DataFrame:
    """Synthesize this gameweek's not-yet-played row(s) — one per real fixture the player's team
    has this gameweek (0 for a blank, 2 for a double — ``engineer_features``'s own
    ``collapse_double_gameweeks`` call handles a double exactly as it does for the backtest, so
    this deliberately does not collapse doubles itself).
    """
    gw_fixtures = fixtures[fixtures["event"] == gameweek]
    if gw_fixtures.empty:
        return pd.DataFrame(columns=MERGED_GW_COLUMNS)

    team_id_by_element = dict(zip(elements["id"], elements["team"], strict=True))
    rows: list[dict] = []
    for element in elements.itertuples():
        player_id = int(element.id)
        team_id = int(team_id_by_element[player_id])
        home_fixtures = gw_fixtures[gw_fixtures["team_h"] == team_id]
        away_fixtures = gw_fixtures[gw_fixtures["team_a"] == team_id]
        for fixture in home_fixtures.itertuples():
            rows.append(
                _target_row(
                    element,
                    player_id,
                    gameweek,
                    position_by_element,
                    own_team_name_by_element,
                    opponent_team=int(fixture.team_a),
                    was_home=True,
                    kickoff_time=fixture.kickoff_time,
                    total_managers=total_managers,
                )
            )
        for fixture in away_fixtures.itertuples():
            rows.append(
                _target_row(
                    element,
                    player_id,
                    gameweek,
                    position_by_element,
                    own_team_name_by_element,
                    opponent_team=int(fixture.team_h),
                    was_home=False,
                    kickoff_time=fixture.kickoff_time,
                    total_managers=total_managers,
                )
            )
    if not rows:
        return pd.DataFrame(columns=MERGED_GW_COLUMNS)
    return pd.DataFrame(rows)[MERGED_GW_COLUMNS]


def _target_row(
    element,
    player_id: int,
    gameweek: int,
    position_by_element: dict[int, str],
    own_team_name_by_element: dict[int, str],
    opponent_team: int,
    was_home: bool,
    kickoff_time,
    total_managers: float,
) -> dict:
    selected_by_percent = float(getattr(element, "selected_by_percent", 0.0) or 0.0)
    transfers_in = float(getattr(element, "transfers_in_event", 0.0) or 0.0)
    transfers_out = float(getattr(element, "transfers_out_event", 0.0) or 0.0)
    row = {
        "player_id": player_id,
        "GW": gameweek,
        "position": position_by_element[player_id],
        "team": own_team_name_by_element[player_id],
        "opponent_team": opponent_team,
        "was_home": was_home,
        "kickoff_time": kickoff_time,
        "team_h_score": np.nan,
        "team_a_score": np.nan,
        "value": float(getattr(element, "now_cost", 0.0) or 0.0),
        "selected": selected_by_percent / 100.0 * total_managers,
        "transfers_out": transfers_out,
        "transfers_balance": transfers_in - transfers_out,
    }
    for col in _UNKNOWN_OUTCOME_COLUMNS:
        row[col] = 0
    return row


def build_merged_gw(
    elements: pd.DataFrame,
    teams: pd.DataFrame,
    fixtures: pd.DataFrame,
    element_summary_histories: pd.DataFrame,
    gameweek: int,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
    target_gameweeks: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Every played gameweek's real history plus the target gameweek(s)' synthesized row(s) — the
    ``merged_gw`` argument :func:`backtest.run_season.engineer_features` expects, built entirely
    from one live snapshot's ``fpl``/``fpl_element_summaries`` sources.

    ``gameweek`` is the real "now" — the played/not-yet-played split point (``GW < gameweek`` is
    played history) — regardless of which gameweek(s) get a synthesized row. ``target_gameweeks``
    lists which not-yet-played gameweek(s) to synthesize target rows for, e.g. ``[gameweek,
    gameweek + 1, gameweek + 2]`` for a live 3-gameweek planning horizon; defaults to
    ``[gameweek]``, reproducing this function's original single-target-gameweek behavior exactly.
    A requested gameweek with no fixtures (past the end of the season, or a blank gameweek for
    every team) simply contributes no target rows — the same "no fixtures this gameweek" case
    already handled per-team below.
    """
    position_by_element = _position_by_element(elements)
    team_name_by_id = _team_name_by_id(teams)
    own_team_name_by_element = _own_team_name_by_element(elements, team_name_by_id)

    played = _played_rows_from_element_summaries(
        element_summary_histories, position_by_element, own_team_name_by_element, gameweek
    )
    targets = [
        _target_gameweek_rows(
            elements,
            teams,
            fixtures,
            target_gameweek,
            position_by_element,
            own_team_name_by_element,
            team_name_by_id,
            total_managers,
        )
        for target_gameweek in (target_gameweeks if target_gameweeks is not None else [gameweek])
    ]
    merged = pd.concat([played, *targets], ignore_index=True)
    merged["kickoff_time"] = pd.to_datetime(merged["kickoff_time"], utc=True)
    return merged


def build_player_histories_from_live_snapshot(
    histories: pd.DataFrame, season_start_year: int
) -> dict[int, pd.DataFrame]:
    """FPL id -> that player's Understat match history, chronologically sorted — the
    ``player_histories`` argument ``engineer_features`` expects, built from the live snapshot's
    ``understat_player_histories`` source (one flat table, ``fpl_id`` already attached by
    :func:`engine.data.ingest.capture_current_gameweek`). Same "no future" discipline as the
    backtest driver's own :func:`backtest.run_season.fetch_understat_player_histories`: rows from
    a season after ``season_start_year`` are dropped, in case a stale/cached table is ever reused.
    """
    if histories.empty:
        return {}
    filtered = histories[histories["season"].astype(int) <= season_start_year].copy()
    result: dict[int, pd.DataFrame] = {}
    for fpl_id, group in filtered.groupby("fpl_id"):
        g = group.copy()
        for col in ("time", "npxG", "xA", "goals", "npg"):
            if col in g.columns:
                g[col] = g[col].astype(float)
        g["date"] = pd.to_datetime(g["date"], utc=True)
        result[int(fpl_id)] = g.sort_values("date").reset_index(drop=True)
    return result


def snapshot_to_feature_inputs(
    season: str,
    gameweek: int,
    captured_at: datetime,
    understat_season_start_year: int,
    base_dir: Path = DEFAULT_BASE_DIR,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
    understat_client: UnderstatClient | None = None,
    prior_season_cache_dir: Path | None = None,
    n_prior_seasons: int | None = None,
    target_gameweeks: Sequence[int] | None = None,
) -> FeatureInputs:
    """Load one live snapshot's four sources and assemble them into
    :func:`backtest.run_season.engineer_features`'s expected input shape.

    ``target_gameweeks``, if given, is passed straight through to :func:`build_merged_gw` — see
    that function's own docstring. Defaults to ``[gameweek]``.

    Deliberately does not import ``backtest.run_season`` at module scope — reused via a local
    import inside this function, since ``engine/`` modules otherwise never depend on ``backtest/``
    and this keeps that dependency contained to the one function that actually needs it (this
    adapter's whole purpose is to feed that module, so the direction is intentional here, not a
    layering violation elsewhere).

    ``understat_client``/``prior_season_cache_dir`` opt into this module's own A.4-equivalent
    cold-start fix (see this module's docstring) — supplying an
    :class:`~engine.data.understat_client.UnderstatClient` fetches and caches
    ``N_PRIOR_SEASONS_FOR_TEAM_RATES`` (or ``n_prior_seasons``, if given) prior seasons' team
    histories and concatenates them with the snapshot's own current-season data before building
    team rates. Omitting ``understat_client`` (the default) uses the current season's snapshot
    data alone, exactly as before — a real gap only at the very start of a season with no
    completed matches yet, but real (confirmed against Understat's actual pre-season 2026/27
    response: zero players, zero teams), so opt-in rather than silently different behavior.
    """
    from backtest.run_season import (
        N_PRIOR_SEASONS_FOR_TEAM_RATES,
        build_team_rate_histories,
        fetch_understat_multi_season_league_data,
    )

    fpl = load_snapshot_tables(base_dir, season, gameweek, captured_at, "fpl")
    understat = load_snapshot_tables(base_dir, season, gameweek, captured_at, "understat")
    element_summaries = load_snapshot_tables(
        base_dir, season, gameweek, captured_at, "fpl_element_summaries"
    )
    understat_histories = load_snapshot_tables(
        base_dir, season, gameweek, captured_at, "understat_player_histories"
    )

    merged_gw = build_merged_gw(
        fpl["elements"],
        fpl["teams"],
        fpl["fixtures"],
        element_summaries.get("histories", pd.DataFrame()),
        gameweek,
        total_managers,
        target_gameweeks=target_gameweeks,
    )

    teams_history_frames = [understat["teams_history"]]
    if understat_client is not None:
        n_prior = n_prior_seasons if n_prior_seasons is not None else N_PRIOR_SEASONS_FOR_TEAM_RATES
        prior_data = fetch_understat_multi_season_league_data(
            understat_season_start_year - 1,
            prior_season_cache_dir or (base_dir / "understat_prior_seasons"),
            understat_client,
            n_prior_seasons=n_prior - 1,
        )
        teams_history_frames = [
            league_data_to_dataframes(data)["teams_history"] for data in prior_data.values()
        ] + teams_history_frames
    combined_teams_history = pd.concat(teams_history_frames, ignore_index=True)
    team_histories = build_team_rate_histories(combined_teams_history)

    player_histories = build_player_histories_from_live_snapshot(
        understat_histories.get("histories", pd.DataFrame()), understat_season_start_year
    )

    return FeatureInputs(
        merged_gw=merged_gw,
        teams=fpl["teams"],
        team_histories=team_histories,
        player_histories=player_histories,
    )
