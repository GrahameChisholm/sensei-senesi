"""Assists model — xA-based scoring rate, opponent-adjusted, symmetric with goals (2.3).

Bookmakers barely price assists, so this is stats-led, kept structurally symmetric with the
goals model (2.2) — same multiplicative rate form, same opponent-xGA adjustment, goals ~
Poisson(lambda) via the simulation layer (2.9).

**Deliberately not blended with team xG as a standing factor.** xA is already, by construction, a
rate stat built from that player's own key passes in actual matches for that actual team, so a
team-xG blend on top of every player's rate risks double-counting the same team-attacking-quality
signal (the same redundancy logic BUILD_PLAN 2.1 used to cut squad depth as a minutes-model
input). Team xG per 90 is used *only* as a shrinkage prior for low-sample players (a new signing,
or a player back from injury with only a few games of individual xA) — see
:func:`shrunk_player_xa_per_90`, opt-in via ``project_assists``' shrinkage arguments, not applied
to every player as an ongoing input.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from engine.rates import shrink_toward_prior
from engine.scoring import ASSIST_POINTS

# Rough prior: fraction of a team's own expected goals a generic creative player's rate implies
# via assists, used only to build a *shrinkage prior* for thin-sample players (BUILD_PLAN 2.3) —
# not asserted precise; Phase 3 backtesting is what actually calibrates this.
DEFAULT_ASSIST_SHARE_OF_TEAM_XG = 0.12


def expected_assist_rate(
    player_xa_per_90: float,
    opponent_xga_per_90: float,
    league_avg_xga_per_90: float,
    expected_minutes: float,
) -> float:
    """The Poisson rate parameter (lambda) for assists this gameweek.

    ``expected_assist_rate = player_xA90 x (opponent_xGA90 / league_avg_xGA90) x
    (expected_minutes / 90)`` — the same functional family as goals (BUILD_PLAN 2.3), since
    creating a chance is harder against a well-organised low block than a leaky defence, exactly
    the same fixture-difficulty logic as scoring one.
    """
    if league_avg_xga_per_90 <= 0:
        raise ValueError("league_avg_xga_per_90 must be positive")
    if any(math.isnan(x) for x in (player_xa_per_90, opponent_xga_per_90, expected_minutes)):
        raise ValueError(
            "rates and expected_minutes must not be NaN (a missing upstream value must fail "
            "loudly here, not propagate to a silent NaN expected_points; ENGINE_IMPROVEMENTS_2.md "
            "C.2)"
        )
    if player_xa_per_90 < 0 or opponent_xga_per_90 < 0 or expected_minutes < 0:
        raise ValueError("rates and expected_minutes must be non-negative")
    fixture_adjustment = opponent_xga_per_90 / league_avg_xga_per_90
    minutes_scaling = expected_minutes / 90.0
    return player_xa_per_90 * fixture_adjustment * minutes_scaling


def prior_assist_rate_from_team_xg(
    team_xg_per_90: float, assist_share: float = DEFAULT_ASSIST_SHARE_OF_TEAM_XG
) -> float:
    """A team-level stand-in for a player's own xA/90, used only as a shrinkage prior (2.3)."""
    if team_xg_per_90 < 0:
        raise ValueError("team_xg_per_90 must be non-negative")
    if not 0.0 <= assist_share <= 1.0:
        raise ValueError("assist_share must be in [0, 1]")
    return team_xg_per_90 * assist_share


def shrunk_player_xa_per_90(
    player_xa_per_90: float,
    individual_weight: float,
    team_xg_per_90: float,
    shrinkage_k: float,
    assist_share: float = DEFAULT_ASSIST_SHARE_OF_TEAM_XG,
) -> float:
    """Blend a thin-sample player's own xA/90 toward a team-xG-derived prior (BUILD_PLAN 2.3).
    ``individual_weight`` should come from :func:`engine.rates.effective_sample_minutes` — more
    accumulated minutes means less shrinkage."""
    prior_rate = prior_assist_rate_from_team_xg(team_xg_per_90, assist_share)
    return shrink_toward_prior(player_xa_per_90, individual_weight, prior_rate, shrinkage_k)


@dataclass(frozen=True)
class AssistProjection:
    """A player's assists-component projection for one gameweek."""

    assist_rate: float  # Poisson lambda

    def __post_init__(self) -> None:
        if self.assist_rate < 0:
            raise ValueError("assist_rate must be non-negative")

    @property
    def expected_assists(self) -> float:
        return self.assist_rate

    @property
    def expected_points(self) -> float:
        """Assist points — flat 3 per assist, same for every position (BUILD_PLAN scoring.py)."""
        return self.assist_rate * ASSIST_POINTS


def project_assists(
    player_xa_per_90: float,
    opponent_xga_per_90: float,
    league_avg_xga_per_90: float,
    expected_minutes: float,
    *,
    individual_weight: float | None = None,
    team_xg_per_90: float | None = None,
    shrinkage_k: float = 0.0,
    assist_share_of_team_xg: float = DEFAULT_ASSIST_SHARE_OF_TEAM_XG,
    conversion_factor: float = 1.0,
) -> AssistProjection:
    """Top-level entry point. Shrinkage toward the team-xG prior only kicks in when the caller
    supplies ``individual_weight``, ``team_xg_per_90``, and a positive ``shrinkage_k`` — omitting
    them uses the player's own xA/90 rate unmodified, per BUILD_PLAN 2.3 ("not applied to every
    player as an ongoing input").

    ``conversion_factor`` (ENGINE_IMPROVEMENTS_5.md Tier 2.3) scales the finished rate, closing the
    documented definitional gap between xA as a statistic and an assist as FPL awards it. FPL
    credits the final pass regardless of how the goal arrived, including deflections, rebounds and
    won penalties, none of which an xA model attributes to the passer, so xA structurally
    under-counts. Measured on real 2025/26 data with no model or selection effect in it, FPL's own
    realised xA totals 538.8 against 733 actual assists (a factor of 1.360), while its realised xG
    tracks goals at 0.971 — the asymmetry is specific to assists, not a general xG/xA problem.

    Applied after shrinkage, since it converts the finished rate's *units* rather than expressing
    any uncertainty about the player's own creativity. 1.0 (the default) leaves this function's
    pre-Tier-2.3 behaviour untouched; the fitted per-position values live in
    ``engine.pipeline.FittedConstants`` and are fitted on training history only."""
    if conversion_factor < 0:
        raise ValueError("conversion_factor must be non-negative")
    effective_xa_per_90 = player_xa_per_90
    if individual_weight is not None and team_xg_per_90 is not None and shrinkage_k > 0:
        effective_xa_per_90 = shrunk_player_xa_per_90(
            player_xa_per_90,
            individual_weight,
            team_xg_per_90,
            shrinkage_k,
            assist_share_of_team_xg,
        )
    rate = expected_assist_rate(
        effective_xa_per_90, opponent_xga_per_90, league_avg_xga_per_90, expected_minutes
    )
    return AssistProjection(assist_rate=rate * conversion_factor)


def fit_assist_share_of_team_xg(
    assists: pd.Series,
    team_xg_per_90: pd.Series,
    minutes: pd.Series,
    min_rows: int = 100,
) -> float:
    """The empirical version of :data:`DEFAULT_ASSIST_SHARE_OF_TEAM_XG`, fit from real assists
    against real team-xG-implied opportunity (ENGINE_IMPROVEMENTS_4.md).

    A single flat 0.12 share applied to every position turned out to be the actual defect behind a
    real measured finding: heavy shrinkage toward that flat prior fixed assists' aggregate
    calibration but introduced a severe FWD over-prediction bias (real players assist less per
    unit of team xG than the flat share implies) that a uniform shrinkage-strength sweep alone
    could not fix — the prior itself, not just how hard to lean on it, needed to vary by position.
    Callers fit one of these per position group (same aggregation style as
    ``backtest.run_season._fit_league_avg_rate_by_position``: a plain whole-window total, not an
    EWMA, since this is the base rate the individual EWMA rate shrinks toward).

    Inverts :func:`prior_assist_rate_from_team_xg`'s ``team_xg_per_90 * assist_share`` form: the
    share that would make the team-xG-derived prior exactly reproduce this group's total assists,
    given its total team-xG-implied opportunity (``team_xg_per_90 * minutes / 90``). Falls back to
    :data:`DEFAULT_ASSIST_SHARE_OF_TEAM_XG` for a too-thin sample or zero opportunity, matching
    every other Tier 1.2 fit function's thin-sample contract.
    """
    if len(assists) < min_rows:
        return DEFAULT_ASSIST_SHARE_OF_TEAM_XG
    total_opportunity = float((team_xg_per_90 * minutes / 90.0).sum())
    if total_opportunity <= 0:
        return DEFAULT_ASSIST_SHARE_OF_TEAM_XG
    return float(assists.sum()) / total_opportunity
