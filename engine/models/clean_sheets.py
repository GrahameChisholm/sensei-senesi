"""Clean sheets & goals conceded — team xG/xGA-based, Dixon-Coles correlated scorelines (2.4).

Derives clean-sheet probability from team-level expected goals against (Understat), adjusted for
the specific opponent's attacking xG, through a Poisson goals model — no market data required, as
deeply backtestable as the individual goals model (2.2) since Understat carries team xG/xGA per
match back to 2014/15.

**Correlated, not independent, scorelines.** Two teams' goal counts in the same match aren't
independent Poisson draws — low-scoring outcomes (0-0, 1-0, 1-1) are measurably more common than
independence predicts (Dixon & Coles, 1997). :func:`scoreline_distribution` applies that
correction before any clean-sheet probability is read off the result, concentrated exactly at the
low scorelines that determine clean sheets.

**Internal consistency requirement (BUILD_PLAN 2.4).** The "opponent's attacking xG" fed into
this model must be the *same number* the goals model (2.2) uses for that opponent's players — in
the full simulation (2.9) that means Sigma of the on-pitch players' individual goal rates, not an
independently-fit team-level parameter. This module stays agnostic about *how* its lambda
parameters were derived (team-xG-based for standalone backtesting of this component in isolation,
or summed player lambdas when wired into the full simulation) — :func:`team_expected_goals_rate`
covers the former; simulate.py owns the latter substitution.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import poisson

from engine.models._discrete import expected_floor_division
from engine.scoring import (
    CLEAN_SHEET_POINTS,
    GOALS_CONCEDED_PENALTY,
    GOALS_CONCEDED_PER_PENALTY,
    GOALS_CONCEDED_POSITIONS,
)

# Dixon-Coles correlation parameter (rho). Negative, per the original paper's convention -- it
# raises P(0-0) and P(1-1) and lowers P(1-0)/P(0-1) relative to independent Poisson. Not asserted
# precise; Phase 3 backtesting is what actually calibrates this per BUILD_PLAN 2.4.
DEFAULT_DIXON_COLES_RHO = -0.1

# Truncation point for the scoreline grid -- P(either side scores 11+) is negligible for any
# realistic Premier League lambda, so this captures effectively all probability mass.
DEFAULT_MAX_GOALS = 10


def team_expected_goals_rate(
    attacking_xg_per_90: float,
    defending_xga_per_90: float,
    league_avg_xga_per_90: float,
) -> float:
    """Expected goals the attacking side scores against this specific defence, opponent-adjusted.

    Same multiplicative functional form as the individual goals model (BUILD_PLAN 2.2), applied
    at team level with no minutes-scaling — a team's clean-sheet outcome doesn't depend on any one
    player's minutes. Call once with (this team attacks, opponent defends) for the team's own
    scoring lambda, and once with the roles swapped for its goals-against lambda.
    """
    if league_avg_xga_per_90 <= 0:
        raise ValueError("league_avg_xga_per_90 must be positive")
    if attacking_xg_per_90 < 0 or defending_xga_per_90 < 0:
        raise ValueError("rates must be non-negative")
    return attacking_xg_per_90 * (defending_xga_per_90 / league_avg_xga_per_90)


def split_by_venue(
    matches: pd.DataFrame, is_home_col: str = "is_home"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a team's match history into home and away subsets (BUILD_PLAN 2.4: home defence is
    measurably stronger than away defence). Feed each subset independently into
    :mod:`engine.rates`' EWMA functions to get separate home/away xG and xGA rates."""
    return matches[matches[is_home_col]], matches[~matches[is_home_col]]


def dixon_coles_tau(
    home_goals: int, away_goals: int, home_lambda: float, away_lambda: float, rho: float
) -> float:
    """The Dixon-Coles (1997) low-score correction factor, applied only to the four cells where
    it's non-trivial (0-0, 1-0, 0-1, 1-1) -- every other scoreline is unadjusted (tau = 1)."""
    if home_goals == 0 and away_goals == 0:
        return 1.0 - home_lambda * away_lambda * rho
    if home_goals == 0 and away_goals == 1:
        return 1.0 + home_lambda * rho
    if home_goals == 1 and away_goals == 0:
        return 1.0 + away_lambda * rho
    if home_goals == 1 and away_goals == 1:
        return 1.0 - rho
    return 1.0


def scoreline_distribution(
    home_lambda: float,
    away_lambda: float,
    rho: float = DEFAULT_DIXON_COLES_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> np.ndarray:
    """A ``(max_goals + 1, max_goals + 1)`` matrix of P(home_goals=i, away_goals=j), Dixon-Coles
    adjusted and renormalized to sum to 1. "Home"/"away" here are just the two correlated sides'
    slot labels for the correction formula -- venue itself is handled upstream, by feeding
    home/away-split rates (:func:`split_by_venue`) into whichever lambda is actually the home team.
    """
    if home_lambda < 0 or away_lambda < 0:
        raise ValueError("lambdas must be non-negative")
    goals = np.arange(max_goals + 1)
    home_pmf = poisson.pmf(goals, home_lambda)
    away_pmf = poisson.pmf(goals, away_lambda)
    joint = np.outer(home_pmf, away_pmf)
    for x in (0, 1):
        for y in (0, 1):
            joint[x, y] *= dixon_coles_tau(x, y, home_lambda, away_lambda, rho)
    joint = np.clip(joint, 0.0, None)
    return joint / joint.sum()


def clean_sheet_probability(
    team_for_lambda: float,
    team_against_lambda: float,
    rho: float = DEFAULT_DIXON_COLES_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> float:
    """P(this team's opponent scores 0) — the whole-match, team-level clean-sheet probability.
    Not gated by any individual player's minutes; the 60+-minute eligibility for the clean-sheet
    *points* is applied separately (see :meth:`CleanSheetProjection.expected_points`)."""
    joint = scoreline_distribution(team_for_lambda, team_against_lambda, rho, max_goals)
    return float(joint[:, 0].sum())


def expected_goals_conceded_penalty(
    team_against_lambda: float,
    expected_minutes: float = 90.0,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> float:
    """Expected value of the ``-1 per 2 goals conceded`` penalty (GK/DEF only), scaled by this
    player's own exposure fraction of the match (``expected_minutes / 90``) — a substitute who
    plays 20 minutes has correspondingly less exposure to concede goals within their own window.

    Uses the full outcome distribution (not ``floor(mean / 2)``) since the conversion is
    non-linear in goals conceded.
    """
    if team_against_lambda < 0:
        raise ValueError("team_against_lambda must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    exposure_fraction = expected_minutes / 90.0
    scaled_lambda = team_against_lambda * exposure_fraction
    expected_units = expected_floor_division(scaled_lambda, GOALS_CONCEDED_PER_PENALTY, max_goals)
    return expected_units * GOALS_CONCEDED_PENALTY


@dataclass(frozen=True)
class CleanSheetProjection:
    """A team's clean-sheet-component projection for one gameweek, from one player's perspective
    (the goals-conceded penalty is already scaled by that player's own expected minutes)."""

    clean_sheet_probability: float
    expected_goals_conceded_penalty: float
    team_for_lambda: float
    team_against_lambda: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.clean_sheet_probability <= 1.0:
            raise ValueError("clean_sheet_probability must be in [0, 1]")
        if self.expected_goals_conceded_penalty > 0:
            raise ValueError("expected_goals_conceded_penalty must be <= 0")

    def expected_points(self, position: str, p_60_plus: float) -> float:
        """Clean-sheet points (position-weighted, gated by the minutes model's P(60+) — BUILD_PLAN
        2.4/scoring.py) plus the goals-conceded penalty for GK/DEF (already minutes-exposure
        scaled, not further gated by the 60+ threshold — FPL applies it to any minutes played)."""
        if position not in CLEAN_SHEET_POINTS:
            raise ValueError(f"unknown position: {position!r}")
        if not 0.0 <= p_60_plus <= 1.0:
            raise ValueError("p_60_plus must be in [0, 1]")
        points = self.clean_sheet_probability * CLEAN_SHEET_POINTS[position] * p_60_plus
        if position in GOALS_CONCEDED_POSITIONS:
            points += self.expected_goals_conceded_penalty
        return points


def project_clean_sheet(
    team_xg_per_90: float,
    team_xga_per_90: float,
    opponent_xg_per_90: float,
    opponent_xga_per_90: float,
    league_avg_xga_per_90: float,
    expected_minutes: float = 90.0,
    rho: float = DEFAULT_DIXON_COLES_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
) -> CleanSheetProjection:
    """Top-level entry point for standalone use/backtesting: derive both lambdas from team-level
    xG/xGA rates. When wired into the full simulation (2.9), build a :class:`CleanSheetProjection`
    directly from the internal-consistency-adjusted lambdas instead of calling this."""
    team_for_lambda = team_expected_goals_rate(
        team_xg_per_90, opponent_xga_per_90, league_avg_xga_per_90
    )
    team_against_lambda = team_expected_goals_rate(
        opponent_xg_per_90, team_xga_per_90, league_avg_xga_per_90
    )
    return CleanSheetProjection(
        clean_sheet_probability=clean_sheet_probability(
            team_for_lambda, team_against_lambda, rho, max_goals
        ),
        expected_goals_conceded_penalty=expected_goals_conceded_penalty(
            team_against_lambda, expected_minutes, max_goals
        ),
        team_for_lambda=team_for_lambda,
        team_against_lambda=team_against_lambda,
    )
