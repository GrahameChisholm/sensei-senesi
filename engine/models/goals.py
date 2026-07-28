"""Goals model — xG-based scoring rate, opponent-adjusted, plus the penalty sub-model (2.2).

Deeply backtestable — over a decade of matched (xG-at-the-time, actual-goals) pairs — because
it's built entirely on the player's own non-penalty xG per 90 (Understat) and the opponent's
defensive xG-against trend, deliberately excluding bookmaker odds (BUILD_PLAN 2.2 — the market
comparison happens later, at decision time, in the Phase 4b overlay, never inside the backtested
core).

Two pieces, kept structurally separate because they're genuinely different processes:
1. **Open-play goals** — a multiplicative rate model (Dixon-Coles-style attack/defence strength),
   producing a Poisson rate parameter (lambda), not a point estimate — the simulation layer (2.9)
   draws discrete goal counts from this per run.
2. **Penalties** — a *addition*, not covered by non-penalty xG by construction. Needs its own
   small sub-model: the team's expected penalties won this game (opponent-adjusted the same way
   as open play), split by the taker's historical conversion rate into expected penalty goals
   (added to this component) and expected penalty misses (its own deduction line).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from engine.rates import shrink_toward_prior
from engine.scoring import GOAL_POINTS, PENALTY_MISS_POINTS, POSITIONS

# League-wide average penalty conversion rate, used only when a taker's own historical sample is
# too thin to trust (BUILD_PLAN 2.2 penalty sub-model). Not asserted precise — a reasonable prior
# pending Phase 3 calibration against real data.
DEFAULT_PENALTY_CONVERSION_RATE = 0.76

# Rough prior: fraction of a team's own expected goals a generic individual player's own npxG/90
# rate implies, used only to build a *shrinkage prior* for thin-sample players (ENGINE_IMPROVEMENTS_2.md
# B.2, symmetric with assists.py's DEFAULT_ASSIST_SHARE_OF_TEAM_XG) — not asserted precise; Phase 3
# backtesting is what actually calibrates this. Real point-in-time Understat rates showed
# physically-impossible outliers (npxG/90 up to 35.4) concentrated entirely in sub-90-minutes-of-
# prior-history players, because unlike xA (2.3) this rate had no shrinkage path wired in at all.
DEFAULT_NPXG_SHARE_OF_TEAM_XG = 0.12


def realized_penalty_goals(
    player_matches: pd.DataFrame, goals_col: str = "goals", npg_col: str = "npg"
) -> pd.Series:
    """Per-match realized penalty goals = ``goals - npg`` (Understat's own non-penalty split),
    clipped at 0. This is the retrospectively-available proxy for "this player took and scored a
    penalty" the penalty sub-model's training data needs (ENGINE_IMPROVEMENTS.md 1.3).

    FPL's bootstrap-static data has no retained per-gameweek history of ``penalties_order``
    (who's the current designated taker) to replay for a backtest — it's a live-only squad-role
    field, not a point-in-time snapshot series. Understat's per-match ``goals`` vs ``npg`` (already
    fetched by ``engine/data/understat_client.py``) gives the same signal retrospectively, at full
    historical depth, with zero new client code. Combine with FPL/vaastav's ``penalties_missed``
    column for full attempt counts (goals + misses), and aggregate across players/gameweeks in the
    backtest driver's feature-engineering stage, not here — this function only derives the
    per-match realized-goals column itself.
    """
    return (player_matches[goals_col] - player_matches[npg_col]).clip(lower=0)


def expected_non_penalty_goal_rate(
    player_npxg_per_90: float,
    opponent_xga_per_90: float,
    league_avg_xga_per_90: float,
    expected_minutes: float,
) -> float:
    """The Poisson rate parameter (lambda) for open-play (non-penalty) goals this gameweek.

    ``expected_goal_rate = player_npxG90 x (opponent_xGA90 / league_avg_xGA90) x
    (expected_minutes / 90)`` — BUILD_PLAN 2.2's functional form, in the same family as standard
    football rate models (Dixon-Coles-style attack/defence strength).
    """
    if league_avg_xga_per_90 <= 0:
        raise ValueError("league_avg_xga_per_90 must be positive")
    if any(math.isnan(x) for x in (player_npxg_per_90, opponent_xga_per_90, expected_minutes)):
        raise ValueError(
            "rates and expected_minutes must not be NaN (a missing upstream value — e.g. an "
            "unmatched ID-crosswalk player — must fail loudly here, not propagate to a silent "
            "NaN expected_points that quietly vanishes from every ranking; ENGINE_IMPROVEMENTS_2.md "
            "C.2)"
        )
    if player_npxg_per_90 < 0 or opponent_xga_per_90 < 0 or expected_minutes < 0:
        raise ValueError("rates and expected_minutes must be non-negative")
    fixture_adjustment = opponent_xga_per_90 / league_avg_xga_per_90
    minutes_scaling = expected_minutes / 90.0
    return player_npxg_per_90 * fixture_adjustment * minutes_scaling


def prior_npxg_rate_from_team_xg(
    team_xg_per_90: float, npxg_share: float = DEFAULT_NPXG_SHARE_OF_TEAM_XG
) -> float:
    """A team-level stand-in for a player's own npxG/90, used only as a shrinkage prior
    (ENGINE_IMPROVEMENTS_2.md B.2) — symmetric with assists.py's
    ``prior_assist_rate_from_team_xg``."""
    if team_xg_per_90 < 0:
        raise ValueError("team_xg_per_90 must be non-negative")
    if not 0.0 <= npxg_share <= 1.0:
        raise ValueError("npxg_share must be in [0, 1]")
    return team_xg_per_90 * npxg_share


def shrunk_player_npxg_per_90(
    player_npxg_per_90: float,
    individual_weight: float,
    team_xg_per_90: float,
    shrinkage_k: float,
    npxg_share: float = DEFAULT_NPXG_SHARE_OF_TEAM_XG,
) -> float:
    """Blend a thin-sample player's own npxG/90 toward a team-xG-derived prior
    (ENGINE_IMPROVEMENTS_2.md B.2). ``individual_weight`` should come from
    :func:`engine.rates.effective_sample_minutes` computed over the player's own prior Understat
    match history — more accumulated minutes means less shrinkage. Symmetric with assists.py's
    ``shrunk_player_xa_per_90``, which this model had no equivalent of before: a real point-in-time
    Understat pull showed npxG/90 outliers up to 35.4 concentrated entirely in players with under
    90 minutes of prior history, precisely because this rate had no shrinkage path at all.
    """
    prior_rate = prior_npxg_rate_from_team_xg(team_xg_per_90, npxg_share)
    return shrink_toward_prior(player_npxg_per_90, individual_weight, prior_rate, shrinkage_k)


def expected_team_penalties(
    team_penalty_win_rate_per_game: float,
    opponent_xga_per_90: float,
    league_avg_xga_per_90: float,
) -> float:
    """Expected penalties won by the player's team this game, opponent-adjusted the same way as
    open-play goals (BUILD_PLAN 2.2) — a leaky defence concedes more penalties too, not just more
    open-play chances."""
    if league_avg_xga_per_90 <= 0:
        raise ValueError("league_avg_xga_per_90 must be positive")
    if team_penalty_win_rate_per_game < 0 or opponent_xga_per_90 < 0:
        raise ValueError("rates must be non-negative")
    return team_penalty_win_rate_per_game * (opponent_xga_per_90 / league_avg_xga_per_90)


def fit_penalty_conversion_rates(
    penalty_attempts: pd.DataFrame,
    player_id_col: str = "player_id",
    scored_col: str = "scored",
    shrinkage_k: float = 8.0,
) -> tuple[dict[int, float], float]:
    """Per-player penalty conversion rates, shrunk toward the league average for thin samples
    (ENGINE_IMPROVEMENTS.md 1.2 — this was a hardcoded placeholder despite the regression layer
    existing), reusing :func:`engine.rates.shrink_toward_prior` — the same shrinkage already used
    by the assists model (2.3) for low-sample players, rather than inventing new statistical
    machinery. ``penalty_attempts`` is one row per realized penalty attempt, ``scored_col`` a
    0/1 outcome column (see :func:`realized_penalty_goals` for deriving these retrospectively from
    Understat data). Returns ``(per_player_rate, league_average_rate)`` — refit every gameweek by
    the walk-forward harness from ``training_history`` only.
    """
    if penalty_attempts.empty:
        return {}, DEFAULT_PENALTY_CONVERSION_RATE
    league_avg = float(penalty_attempts[scored_col].mean())
    rates: dict[int, float] = {}
    for player_id, group in penalty_attempts.groupby(player_id_col):
        individual_rate = float(group[scored_col].mean())
        individual_weight = float(len(group))
        rates[int(player_id)] = shrink_toward_prior(
            individual_rate, individual_weight, league_avg, shrinkage_k
        )
    return rates, league_avg


@dataclass(frozen=True)
class PenaltyOutcome:
    expected_penalty_goals: float
    expected_penalty_misses: float


def penalty_goals_and_misses(
    team_expected_penalties: float,
    taker_share: float = 0.0,
    conversion_rate: float = DEFAULT_PENALTY_CONVERSION_RATE,
) -> PenaltyOutcome:
    """Split a team's expected penalties into this player's expected goals vs misses.

    ``taker_share``: fraction of the team's penalties this player takes — 1.0 for the sole
    designated taker, 0.0 for everyone else (BUILD_PLAN 2.2: "identify each team's primary
    penalty taker"). ``conversion_rate``: the taker's own historical conversion rate, falling
    back to :data:`DEFAULT_PENALTY_CONVERSION_RATE` for a taker with too thin a sample.
    """
    if not 0.0 <= taker_share <= 1.0:
        raise ValueError("taker_share must be in [0, 1]")
    if not 0.0 <= conversion_rate <= 1.0:
        raise ValueError("conversion_rate must be in [0, 1]")
    if team_expected_penalties < 0:
        raise ValueError("team_expected_penalties must be non-negative")
    taker_penalties = team_expected_penalties * taker_share
    return PenaltyOutcome(
        expected_penalty_goals=taker_penalties * conversion_rate,
        expected_penalty_misses=taker_penalties * (1.0 - conversion_rate),
    )


@dataclass(frozen=True)
class GoalProjection:
    """A player's full goals-component projection for one gameweek."""

    non_penalty_goal_rate: float  # Poisson lambda for open-play goals
    expected_penalty_goals: float
    expected_penalty_misses: float

    def __post_init__(self) -> None:
        for name in ("non_penalty_goal_rate", "expected_penalty_goals", "expected_penalty_misses"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")

    @property
    def expected_goals(self) -> float:
        """Total expected goals (open-play + penalty) — what the simulation layer (2.9) apportions
        across a team's drawn scoreline via a multinomial draw (BUILD_PLAN 2.9 step 3)."""
        return self.non_penalty_goal_rate + self.expected_penalty_goals

    def expected_points(self, position: str) -> float:
        """Goal points (position-weighted) plus the penalty-miss deduction — both lines of the
        BUILD_PLAN 2.2/scoring.py conversion this component is responsible for."""
        if position not in POSITIONS:
            raise ValueError(f"unknown position: {position!r}")
        return (
            self.expected_goals * GOAL_POINTS[position]
            + self.expected_penalty_misses * PENALTY_MISS_POINTS
        )


def project_goals(
    player_npxg_per_90: float,
    opponent_xga_per_90: float,
    league_avg_xga_per_90: float,
    expected_minutes: float,
    team_expected_penalties: float = 0.0,
    taker_share: float = 0.0,
    penalty_conversion_rate: float = DEFAULT_PENALTY_CONVERSION_RATE,
    *,
    individual_weight: float | None = None,
    team_xg_per_90: float | None = None,
    shrinkage_k: float = 0.0,
    npxg_share_of_team_xg: float = DEFAULT_NPXG_SHARE_OF_TEAM_XG,
) -> GoalProjection:
    """Top-level entry point: combine the open-play rate and the penalty sub-model into one
    :class:`GoalProjection` for a single player in a single gameweek.

    Shrinkage toward the team-xG prior (ENGINE_IMPROVEMENTS_2.md B.2) only kicks in when the
    caller supplies ``individual_weight``, ``team_xg_per_90``, and a positive ``shrinkage_k`` —
    omitting them uses the player's own npxG/90 rate unmodified, matching every existing caller
    and keeping this symmetric with ``engine.models.assists.project_assists``'s own opt-in
    shrinkage arguments.
    """
    effective_npxg_per_90 = player_npxg_per_90
    if individual_weight is not None and team_xg_per_90 is not None and shrinkage_k > 0:
        effective_npxg_per_90 = shrunk_player_npxg_per_90(
            player_npxg_per_90, individual_weight, team_xg_per_90, shrinkage_k, npxg_share_of_team_xg
        )
    non_penalty_rate = expected_non_penalty_goal_rate(
        effective_npxg_per_90, opponent_xga_per_90, league_avg_xga_per_90, expected_minutes
    )
    penalty_outcome = penalty_goals_and_misses(
        team_expected_penalties, taker_share, penalty_conversion_rate
    )
    return GoalProjection(
        non_penalty_goal_rate=non_penalty_rate,
        expected_penalty_goals=penalty_outcome.expected_penalty_goals,
        expected_penalty_misses=penalty_outcome.expected_penalty_misses,
    )
