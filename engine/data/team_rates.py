"""Team-level xG/xGA rate snapshots, shared by the season backtest and the live app.

Moved out of ``backtest/run_season.py`` (ENGINE_IMPROVEMENTS_3.md A.1/B1) so both the season
backtest's per-gameweek walk-forward replay (``backtest.run_season.build_fixture_rate_frame``) and
the live app's "every team's rate right now" snapshot (:func:`build_current_team_rates`, used to
populate ``features.fixtures.TeamRates`` for the live fixture-difficulty model) share one
implementation of the shrinkage/venue-multiplier math, rather than risking two copies drifting
apart. ``backtest/run_season.py`` still owns ``build_fixture_rate_frame`` itself, since that
function's per-gameweek point-in-time replay is backtest-specific orchestration, not shared math.

**Not a true home/away split.** A real per-team split of each team's own rate by venue
(``_team_rate_asof_venue_split``, the direct predecessor of this module's approach) was tried and
reverted: it measurably made clean-sheet calibration *worse than predicting the league base rate*
(Brier 0.1895 vs a constant-base-rate Brier of 0.1872), because splitting by venue halves the
effective sample behind every team rate while the home/away effect itself is close in size to the
per-team-match noise. The approach here instead keeps one full-sample, league-shrunk rate per team
(:func:`_team_rate_asof_shrunk`) and applies a separate, much more data-efficient venue
*multiplier* on top (:func:`_league_venue_multipliers`/:func:`_team_venue_multipliers`) — see each
function's own docstring for the evidence.

``engine/`` never depends on ``features/`` (the app-facing layer sits on top of the engine, not the
other way around), so :class:`TeamRateSnapshot` here is a plain engine-level equivalent of
``features.fixtures.TeamRates`` rather than that class itself. ``scripts/build_projections.py``
serializes a snapshot's four rates into the projection cache; ``api/state.py`` deserializes them
back into the real ``features.fixtures.TeamRates`` the fixture-difficulty model consumes.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from engine.rates import latest_ewma_rate, league_average_rate, shrink_toward_prior

__all__ = [
    "UNDERSTAT_TO_FPL_TEAM_NAME",
    "TEAM_RATE_SHRINKAGE_K",
    "TEAM_VENUE_SHRINKAGE_K",
    "TeamRateSnapshot",
    "build_team_rate_histories",
    "build_current_team_rates",
]

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
    "Coventry": "Coventry City",
    "Hull": "Hull City",
    "Ipswich": "Ipswich Town",
}


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
    :func:`_league_venue_multipliers`, applied afterward in
    ``backtest.run_season.build_fixture_rate_frame``).

    A newly-promoted club has zero prior top-flight matches, so its own raw rate is NaN with
    zero weight. :func:`~engine.rates.shrink_toward_prior` already returns the prior outright in
    that case, so the result here is the full league-average rate rather than a missing value,
    closing a gap where an established team's very first fixture against a debutant club would
    otherwise get a NaN opponent rate and be dropped entirely by the required-columns dropna
    downstream.
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
    ``backtest.run_season.build_fixture_rate_frame`` applies this without a separate away-side
    computation. Falls back to ``(1.0, 1.0)`` — venue-neutral — when fewer than
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
# Brier / overall MAE); k=10 clears the < 0.03 MACE target with margin at no Brier cost, preferred
# over the unshrunk k=0 that scores best on MACE alone. See backtest/run_season.py's git history
# for the full sweep table this constant was chosen from.
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
    :func:`_team_rate_asof_shrunk` already applies to the rates themselves.

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


@dataclass(frozen=True)
class TeamRateSnapshot:
    """A team's current home/away xG/xGA per-90 rates — the engine-level equivalent of
    ``features.fixtures.TeamRates`` (see this module's own docstring for why they're kept as two
    separate types rather than one)."""

    home_xg_per_90: float
    away_xg_per_90: float
    home_xga_per_90: float
    away_xga_per_90: float


def build_current_team_rates(
    team_histories: dict[str, pd.DataFrame], as_of: pd.Timestamp
) -> dict[str, TeamRateSnapshot]:
    """Every team's current (as of ``as_of``) home/away xG/xGA rate snapshot, keyed by FPL team
    name (matching ``team_histories``'s own keys) — the "every team's rate right now" entry point
    the live app needs, as opposed to ``backtest.run_season.build_fixture_rate_frame``'s
    per-gameweek walk-forward replay. Uses the exact same shrunk-rate-plus-venue-multiplier method
    as that function, just evaluated once at a single point in time for every team rather than once
    per historical gameweek — see :func:`_team_rate_asof_shrunk`/:func:`_league_venue_multipliers`/
    :func:`_team_venue_multipliers` for the method itself.

    A team with no prior history at all (a true-GW1 promoted club with an empty snapshot) gets a
    snapshot of all-NaN rates rather than being dropped — matching this module's existing
    "degrade, don't crash" convention (see :func:`build_team_rate_histories`'s own docstring) — and
    a snapshot where every team is historyless (a real pre-season pull with no matches played
    anywhere yet) degrades the same way rather than raising, since there is genuinely no rate to
    report yet.
    """
    if not team_histories:
        return {}

    raw_xg = {
        team: _team_rate_asof(history, "xG", as_of) for team, history in team_histories.items()
    }
    raw_xga = {
        team: _team_rate_asof(history, "xGA", as_of) for team, history in team_histories.items()
    }
    valid_xg = {team: rate for team, rate in raw_xg.items() if not pd.isna(rate)}
    valid_xga = {team: rate for team, rate in raw_xga.items() if not pd.isna(rate)}
    league_avg_xg = league_average_rate(valid_xg) if valid_xg else float("nan")
    league_avg_xga = league_average_rate(valid_xga) if valid_xga else float("nan")

    league_multipliers = _league_venue_multipliers(team_histories, as_of)

    snapshots: dict[str, TeamRateSnapshot] = {}
    for team, history in team_histories.items():
        shrunk_xg = _team_rate_asof_shrunk(history, "xG", as_of, league_avg_xg)
        shrunk_xga = _team_rate_asof_shrunk(history, "xGA", as_of, league_avg_xga)
        xg_mult, xga_mult = _team_venue_multipliers(history, as_of, league_multipliers)
        snapshots[team] = TeamRateSnapshot(
            home_xg_per_90=shrunk_xg * xg_mult,
            away_xg_per_90=shrunk_xg * (2.0 - xg_mult),
            home_xga_per_90=shrunk_xga * xga_mult,
            away_xga_per_90=shrunk_xga * (2.0 - xga_mult),
        )
    return snapshots
