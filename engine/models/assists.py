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
) -> AssistProjection:
    """Top-level entry point. Shrinkage toward the team-xG prior only kicks in when the caller
    supplies ``individual_weight``, ``team_xg_per_90``, and a positive ``shrinkage_k`` — omitting
    them uses the player's own xA/90 rate unmodified, per BUILD_PLAN 2.3 ("not applied to every
    player as an ongoing input")."""
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
    return AssistProjection(assist_rate=rate)
