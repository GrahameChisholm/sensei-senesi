"""Monte Carlo simulation of correlated match scripts -> full outcome distributions (2.9).

A single expected-points number (2.7) hides the *shape* of the outcome, and captaincy/chip
decisions specifically need that shape: median, floor, ceiling, P(big haul). This module runs one
fixture thousands of times, each run following one coherent generative story that correlates
components which genuinely move together in a real match (BUILD_PLAN 2.9), rather than drawing
every component independently:

1. **Minutes** — each player's bucket (0 / 1-59 / 60+) is drawn from their
   :class:`~engine.models.minutes.MinutesDistribution`, then that bucket's conditional expected
   minutes is used as the run's minutes value (the minutes model provides a mean per bucket, not
   a within-bucket distribution, so this is the model's own resolution — not further invented
   noise on top of it).
2. **Match script** — each team's goal-scoring lambda *for that run* is the sum of its on-pitch
   players' own adjusted goal rates scaled by *that run's* drawn minutes — the BUILD_PLAN 2.4
   internal-consistency requirement satisfied literally, since the lineup (and hence the team
   lambda) genuinely varies run to run. A correlated (Dixon-Coles) scoreline is then drawn from
   the two teams' lambdas.
3. **Individual goals/assists** — the team's drawn goal total is apportioned across on-pitch
   players by each player's share of the team's total adjusted goal rate that run (a multinomial
   draw), with the penalty sub-model layered on top for the designated taker. Assists are
   apportioned the same way, weighted by adjusted assist rate, with a residual "unassisted"
   probability.
4. **Clean sheet / goals conceded** — read directly off the same drawn scoreline.
5. **Defensive contribution** — drawn independently per player (Negative Binomial), since it
   genuinely doesn't hinge on the scoreline.
6. **Bonus** — the regression proxy (2.6) applied to each player's *realized* stats for that run.
7. **Cards** — drawn independently per player from historical rates.
8. Sum everything for the run; repeat thousands of times to build the distribution.

**Two documented extensions beyond BUILD_PLAN 2.9's 8-step list**, needed to actually total the
points but not spelled out there:
- **Goals-conceded exposure**: which of the team's conceded goals happened while a given
  substitute was actually on the pitch isn't modelled at goal-timing granularity. Each conceded
  goal is instead treated as independently having probability ``minutes / 90`` of having happened
  while this player was on, i.e. ``own_conceded ~ Binomial(team_goals_conceded, minutes / 90)``.
- **Saves (GK)**: not mentioned in the 8-step list at all. Drawn independently per run (Poisson),
  the same "doesn't hinge on the scoreline" treatment BUILD_PLAN gives defensive contribution and
  cards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from engine.models.bonus import BonusModel
from engine.models.bonus import build_features as build_bonus_features
from engine.models.clean_sheets import DEFAULT_DIXON_COLES_RHO, DEFAULT_MAX_GOALS
from engine.models.defensive_contribution import DEFAULT_OVERDISPERSION
from engine.models.minutes import MinutesDistribution
from engine.scoring import (
    ASSIST_POINTS,
    CLEAN_SHEET_MIN_MINUTES,
    CLEAN_SHEET_POINTS,
    DEFENSIVE_CONTRIBUTION_POINTS,
    DEFENSIVE_CONTRIBUTION_THRESHOLD,
    GOAL_POINTS,
    GOALS_CONCEDED_PENALTY,
    GOALS_CONCEDED_PER_PENALTY,
    GOALS_CONCEDED_POSITIONS,
    PENALTY_MISS_POINTS,
    PENALTY_SAVE_POINTS,
    RED_CARD_POINTS,
    SAVES_PER_POINT,
    YELLOW_CARD_POINTS,
)

DEFAULT_N_RUNS = 2000
DEFAULT_UNASSISTED_PROBABILITY = 0.35
BIG_HAUL_THRESHOLD = 10.0
FLOOR_PERCENTILE = 10.0
CEILING_PERCENTILE = 90.0
# Appearance points by minutes bucket (0 / 1-59 / 60+), in that order -- matches
# engine.scoring.APPEARANCE_POINTS' values exactly.
_APPEARANCE_POINTS_BY_BUCKET = np.array([0.0, 1.0, 2.0])


@dataclass(frozen=True)
class PlayerMatchInputs:
    """One player's per-match inputs, already opponent/fixture-adjusted where each component's
    functional form calls for it (BUILD_PLAN 2.2-2.6) — this module owns only the stochastic
    integration across components, not the adjustment math itself."""

    player_id: int
    position: str
    minutes_distribution: MinutesDistribution
    adjusted_goal_rate_per_90: float
    adjusted_assist_rate_per_90: float
    is_penalty_taker: bool = False
    penalty_conversion_rate: float = 0.0
    adjusted_defensive_action_rate_per_90: float = 0.0
    dc_overdispersion_alpha: float = DEFAULT_OVERDISPERSION
    yellow_card_rate_per_90: float = 0.0
    red_card_rate_per_90: float = 0.0
    expected_saves_full_match: float = 0.0
    expected_penalties_faced_full_match: float = 0.0
    penalty_save_rate: float = 0.0


@dataclass(frozen=True)
class TeamMatchInputs:
    players: list[PlayerMatchInputs]
    team_expected_penalties: float = 0.0

    def __post_init__(self) -> None:
        if not self.players:
            raise ValueError("players must not be empty")


@dataclass(frozen=True)
class PlayerSimulationSummary:
    """The distributional output BUILD_PLAN 2.9 is for: median, floor, ceiling, P(big haul) —
    not just a single expected value."""

    player_id: int
    mean: float
    median: float
    floor: float
    ceiling: float
    prob_big_haul: float
    raw_points: np.ndarray = field(repr=False)
    # MINI_LEAGUE_PLAN M9: persisted so head-to-head gap variance doesn't have to fall back to the
    # cruder (ceiling - floor) / 2.5631 normal-spread approximation. Optional so a summary built
    # (or deserialized) before this field existed remains a valid construction.
    std: float | None = None


@dataclass(frozen=True)
class FixtureSimulationResult:
    player_summaries: dict[int, PlayerSimulationSummary]


def _draw_minutes(
    rng: np.random.Generator, distributions: list[MinutesDistribution], n_runs: int
) -> tuple[np.ndarray, np.ndarray]:
    """Returns ``(bucket_idx, minutes)``, each shape ``(n_players, n_runs)``. Bucket index 0/1/2
    maps to 0 / 1-59 / 60+; minutes is that bucket's conditional expected value (see module
    docstring — the minutes model gives a mean per bucket, not a within-bucket distribution)."""
    n_players = len(distributions)
    bucket_idx = np.empty((n_players, n_runs), dtype=int)
    minutes = np.empty((n_players, n_runs), dtype=float)
    for i, dist in enumerate(distributions):
        buckets = rng.choice(3, size=n_runs, p=[dist.p_zero, dist.p_1_to_59, dist.p_60_plus])
        bucket_idx[i] = buckets
        minutes[i] = np.select(
            [buckets == 0, buckets == 1, buckets == 2],
            [0.0, dist.expected_minutes_given_1_to_59, dist.expected_minutes_given_60_plus],
        )
    return bucket_idx, minutes


def _draw_scoreline(
    rng: np.random.Generator,
    home_lambda: np.ndarray,
    away_lambda: np.ndarray,
    rho: float,
    max_goals: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized Dixon-Coles-adjusted scoreline draw, one independent draw per run (lambdas vary
    per run since the on-pitch lineup varies per run). Returns ``(home_goals, away_goals)``, each
    shape ``(n_runs,)``."""
    n_runs = home_lambda.shape[0]
    goals = np.arange(max_goals + 1)
    # Poisson pmf via the recurrence-free closed form (avoids adding a scipy call per element):
    # pmf(k; lam) = exp(-lam) * lam^k / k!
    log_factorial = np.cumsum(np.log(np.arange(1, max_goals + 2)))
    log_factorial = np.insert(log_factorial, 0, 0.0)[: max_goals + 1]

    def _poisson_pmf(lam: np.ndarray) -> np.ndarray:
        lam = np.clip(lam, 1e-12, None)
        log_pmf = -lam[:, None] + goals[None, :] * np.log(lam[:, None]) - log_factorial[None, :]
        return np.exp(log_pmf)

    home_pmf = _poisson_pmf(home_lambda)  # (n_runs, G)
    away_pmf = _poisson_pmf(away_lambda)  # (n_runs, G)
    joint = home_pmf[:, :, None] * away_pmf[:, None, :]  # (n_runs, G, G)

    for x in (0, 1):
        for y in (0, 1):
            tau = 1.0 - home_lambda * away_lambda * rho if (x, y) == (0, 0) else None
            if (x, y) == (0, 1):
                tau = 1.0 + home_lambda * rho
            elif (x, y) == (1, 0):
                tau = 1.0 + away_lambda * rho
            elif (x, y) == (1, 1):
                tau = np.full(n_runs, 1.0 - rho)
            joint[:, x, y] *= tau

    joint = np.clip(joint, 0.0, None)
    joint = joint.reshape(n_runs, -1)
    joint /= joint.sum(axis=1, keepdims=True)

    cumsum = np.cumsum(joint, axis=1)
    u = rng.uniform(0.0, 1.0, size=n_runs)
    flat_idx = np.sum(cumsum < u[:, None], axis=1)
    flat_idx = np.clip(flat_idx, 0, joint.shape[1] - 1)
    n_goal_values = max_goals + 1
    return flat_idx // n_goal_values, flat_idx % n_goal_values


def _draw_defensive_actions(rng: np.random.Generator, mu: np.ndarray, alpha: float) -> np.ndarray:
    n_param = 1.0 / alpha
    with np.errstate(divide="ignore", invalid="ignore"):
        p_param = np.where(mu > 0, n_param / (n_param + mu), 1.0)
    draws = rng.negative_binomial(n_param, p_param)
    return np.where(mu > 0, draws, 0)


def _apportion_goals_and_assists(
    rng: np.random.Generator,
    team_goals: np.ndarray,
    minutes: np.ndarray,
    goal_rates_per_90: np.ndarray,
    assist_rates_per_90: np.ndarray,
    is_taker: np.ndarray,
    conversion_rates: np.ndarray,
    team_expected_penalties: float,
    unassisted_probability: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per-run multinomial apportionment of a team's drawn goal total across its on-pitch players
    (BUILD_PLAN 2.9 step 3). Kept as an explicit per-run loop: each run's goal total and each
    player's on-pitch weight both vary run to run, and numpy has no vectorized "different n and
    different pvals per row" multinomial primitive. Goal counts are small (usually 0-3), so this
    stays cheap even across thousands of runs.

    Returns ``(goals, assists, penalty_goals, penalty_misses)``, each shape ``(n_players, n_runs)``.
    """
    n_players, n_runs = minutes.shape
    goals = np.zeros((n_players, n_runs), dtype=int)
    assists = np.zeros((n_players, n_runs), dtype=int)
    penalty_goals = np.zeros((n_players, n_runs), dtype=int)
    penalty_misses = np.zeros((n_players, n_runs), dtype=int)

    penalty_counts = (
        rng.poisson(team_expected_penalties, size=n_runs)
        if team_expected_penalties > 0
        else np.zeros(n_runs, dtype=int)
    )
    taker_idx = np.where(is_taker)[0]

    for run in range(n_runs):
        on_pitch = minutes[:, run] > 0

        n_pens = penalty_counts[run]
        if n_pens > 0 and taker_idx.size > 0:
            eligible = taker_idx[on_pitch[taker_idx]]
            if eligible.size > 0:
                taker = eligible[0]
                makes = rng.binomial(n_pens, conversion_rates[taker])
                penalty_goals[taker, run] = makes
                penalty_misses[taker, run] = n_pens - makes

        n_goals = int(team_goals[run])
        if n_goals <= 0:
            continue

        weights = goal_rates_per_90 * minutes[:, run] / 90.0 * on_pitch
        total_weight = weights.sum()
        if total_weight <= 0:
            continue
        probs = weights / total_weight
        goals[:, run] = rng.multinomial(n_goals, probs)

        assist_weights = assist_rates_per_90 * minutes[:, run] / 90.0 * on_pitch
        total_assist_weight = assist_weights.sum()
        if total_assist_weight > 0:
            assist_probs = assist_weights / total_assist_weight * (1.0 - unassisted_probability)
            full_probs = np.append(assist_probs, unassisted_probability)
            assist_draw = rng.multinomial(n_goals, full_probs)
            assists[:, run] = assist_draw[:-1]

    goals += penalty_goals
    return goals, assists, penalty_goals, penalty_misses


def simulate_fixture(
    home: TeamMatchInputs,
    away: TeamMatchInputs,
    bonus_model: BonusModel,
    n_runs: int = DEFAULT_N_RUNS,
    rho: float = DEFAULT_DIXON_COLES_RHO,
    max_goals: int = DEFAULT_MAX_GOALS,
    unassisted_probability: float = DEFAULT_UNASSISTED_PROBABILITY,
    seed: int | None = None,
) -> FixtureSimulationResult:
    """Run one fixture ``n_runs`` times end-to-end (BUILD_PLAN 2.9 steps 1-8) and summarize each
    player's resulting points distribution."""
    rng = np.random.default_rng(seed)

    home_bucket, home_minutes = _draw_minutes(
        rng, [p.minutes_distribution for p in home.players], n_runs
    )
    away_bucket, away_minutes = _draw_minutes(
        rng, [p.minutes_distribution for p in away.players], n_runs
    )

    home_goal_rates = np.array([p.adjusted_goal_rate_per_90 for p in home.players])
    away_goal_rates = np.array([p.adjusted_goal_rate_per_90 for p in away.players])
    home_lambda = (home_goal_rates[:, None] * home_minutes / 90.0).sum(axis=0)
    away_lambda = (away_goal_rates[:, None] * away_minutes / 90.0).sum(axis=0)

    home_goals, away_goals = _draw_scoreline(rng, home_lambda, away_lambda, rho, max_goals)

    home_assist_rates = np.array([p.adjusted_assist_rate_per_90 for p in home.players])
    away_assist_rates = np.array([p.adjusted_assist_rate_per_90 for p in away.players])
    home_is_taker = np.array([p.is_penalty_taker for p in home.players])
    away_is_taker = np.array([p.is_penalty_taker for p in away.players])
    home_conversion = np.array([p.penalty_conversion_rate for p in home.players])
    away_conversion = np.array([p.penalty_conversion_rate for p in away.players])

    home_indiv_goals, home_assists, _, home_pen_misses = _apportion_goals_and_assists(
        rng,
        home_goals,
        home_minutes,
        home_goal_rates,
        home_assist_rates,
        home_is_taker,
        home_conversion,
        home.team_expected_penalties,
        unassisted_probability,
    )
    away_indiv_goals, away_assists, _, away_pen_misses = _apportion_goals_and_assists(
        rng,
        away_goals,
        away_minutes,
        away_goal_rates,
        away_assist_rates,
        away_is_taker,
        away_conversion,
        away.team_expected_penalties,
        unassisted_probability,
    )

    home_summaries = _score_team(
        rng,
        home.players,
        home_bucket,
        home_minutes,
        home_indiv_goals,
        home_assists,
        home_pen_misses,
        team_goals_conceded=away_goals,
        team_clean_sheet=away_goals == 0,
        bonus_model=bonus_model,
    )
    away_summaries = _score_team(
        rng,
        away.players,
        away_bucket,
        away_minutes,
        away_indiv_goals,
        away_assists,
        away_pen_misses,
        team_goals_conceded=home_goals,
        team_clean_sheet=home_goals == 0,
        bonus_model=bonus_model,
    )

    return FixtureSimulationResult(player_summaries={**home_summaries, **away_summaries})


def _score_team(
    rng: np.random.Generator,
    players: list[PlayerMatchInputs],
    minutes_bucket: np.ndarray,
    minutes: np.ndarray,
    indiv_goals: np.ndarray,
    assists: np.ndarray,
    penalty_misses: np.ndarray,
    team_goals_conceded: np.ndarray,
    team_clean_sheet: np.ndarray,
    bonus_model: BonusModel,
) -> dict[int, PlayerSimulationSummary]:
    n_players, n_runs = minutes.shape
    minutes_fraction = minutes / 90.0

    appearance_points = _APPEARANCE_POINTS_BY_BUCKET[minutes_bucket]
    goals_points = np.array([GOAL_POINTS[p.position] for p in players])[:, None] * indiv_goals
    assists_points = assists * ASSIST_POINTS
    penalty_miss_points = penalty_misses * PENALTY_MISS_POINTS

    clean_sheet_points_by_pos = np.array([CLEAN_SHEET_POINTS[p.position] for p in players])[:, None]
    played_60_plus = minutes >= CLEAN_SHEET_MIN_MINUTES
    clean_sheet_points = clean_sheet_points_by_pos * team_clean_sheet[None, :] * played_60_plus

    own_conceded = rng.binomial(
        np.broadcast_to(team_goals_conceded[None, :], (n_players, n_runs)),
        np.clip(minutes_fraction, 0.0, 1.0),
    )
    goals_conceded_units = own_conceded // GOALS_CONCEDED_PER_PENALTY
    is_conceded_position = np.array([p.position in GOALS_CONCEDED_POSITIONS for p in players])[
        :, None
    ]
    goals_conceded_points = goals_conceded_units * GOALS_CONCEDED_PENALTY * is_conceded_position

    dc_rates = [
        (
            p.adjusted_defensive_action_rate_per_90
            if p.position in DEFENSIVE_CONTRIBUTION_THRESHOLD
            else 0.0
        )
        for p in players
    ]
    dc_mu = np.array(dc_rates)[:, None] * minutes_fraction
    defensive_actions = np.array(
        [
            (
                _draw_defensive_actions(rng, dc_mu[i], players[i].dc_overdispersion_alpha)
                if players[i].position in DEFENSIVE_CONTRIBUTION_THRESHOLD
                else np.zeros(n_runs, dtype=int)
            )
            for i in range(n_players)
        ]
    )
    dc_thresholds = np.array(
        [DEFENSIVE_CONTRIBUTION_THRESHOLD.get(p.position, np.inf) for p in players]
    )[:, None]
    dc_points = (defensive_actions >= dc_thresholds) * DEFENSIVE_CONTRIBUTION_POINTS

    yellow_mu = np.array([p.yellow_card_rate_per_90 for p in players])[:, None] * minutes_fraction
    red_mu = np.array([p.red_card_rate_per_90 for p in players])[:, None] * minutes_fraction
    yellows = rng.poisson(yellow_mu)
    reds = rng.poisson(red_mu)
    cards_points = yellows * YELLOW_CARD_POINTS + reds * RED_CARD_POINTS

    saves_mu = np.array([p.expected_saves_full_match for p in players])[:, None] * minutes_fraction
    penalties_faced_mu = (
        np.array([p.expected_penalties_faced_full_match for p in players])[:, None]
        * minutes_fraction
    )
    saves = rng.poisson(saves_mu)
    penalties_faced = rng.poisson(penalties_faced_mu)
    penalty_save_rates = np.array([p.penalty_save_rate for p in players])[:, None]
    penalty_saves = rng.binomial(penalties_faced, np.clip(penalty_save_rates, 0.0, 1.0))
    saves_points = (saves // SAVES_PER_POINT) + penalty_saves * PENALTY_SAVE_POINTS

    bonus_points = _predict_bonus(
        bonus_model,
        players,
        indiv_goals,
        assists,
        team_clean_sheet,
        played_60_plus,
        defensive_actions,
        minutes,
    )

    total_points = (
        appearance_points
        + goals_points
        + assists_points
        + penalty_miss_points
        + clean_sheet_points
        + goals_conceded_points
        + dc_points
        + cards_points
        + saves_points
        + bonus_points
    )

    summaries: dict[int, PlayerSimulationSummary] = {}
    for i, player in enumerate(players):
        points = total_points[i]
        summaries[player.player_id] = PlayerSimulationSummary(
            player_id=player.player_id,
            mean=float(np.mean(points)),
            median=float(np.median(points)),
            floor=float(np.percentile(points, FLOOR_PERCENTILE)),
            ceiling=float(np.percentile(points, CEILING_PERCENTILE)),
            prob_big_haul=float(np.mean(points >= BIG_HAUL_THRESHOLD)),
            raw_points=points,
            std=float(np.std(points)),
        )
    return summaries


def _predict_bonus(
    bonus_model: BonusModel,
    players: list[PlayerMatchInputs],
    indiv_goals: np.ndarray,
    assists: np.ndarray,
    team_clean_sheet: np.ndarray,
    played_60_plus: np.ndarray,
    defensive_actions: np.ndarray,
    minutes: np.ndarray,
) -> np.ndarray:
    """Bonus via the regression proxy applied to each player's *realized* stats this run (BUILD_PLAN
    2.9 step 6) -- flattened into one batched DataFrame across all (player, run) pairs for one
    fast vectorized ``predict`` call rather than one call per run. ``minutes`` is this same run's
    own *drawn* minutes (ENGINE_IMPROVEMENTS_3.md A.2) -- the simulation-layer equivalent of the
    point-estimate path's modelled expected minutes, so a run where a player is drawn to play 10
    minutes doesn't get bonus on the strength of its other realized-but-minutes-independent
    features alone."""
    n_players, n_runs = indiv_goals.shape
    rows = []
    for i, player in enumerate(players):
        clean_sheet_realized = team_clean_sheet.astype(float) * played_60_plus[i].astype(float)
        for run in range(n_runs):
            rows.append(
                build_bonus_features(
                    expected_goals=float(indiv_goals[i, run]),
                    expected_assists=float(assists[i, run]),
                    clean_sheet_probability=float(clean_sheet_realized[run]),
                    defensive_action_rate=float(defensive_actions[i, run]),
                    position=player.position,
                    expected_minutes=float(minutes[i, run]),
                )
            )
    features = pd.DataFrame(rows)
    predicted = np.array([p.expected_bonus for p in bonus_model.predict(features)])
    return predicted.reshape(n_players, n_runs)
