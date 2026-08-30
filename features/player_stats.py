"""Actual-performance summarization for the Player Stats page (PLAYER_STATS_PLAN D2/D12/D13) --
sums a player's per-gameweek actual counts and points (``engine.data.player_history``) over a
gameweek range picked in the UI. Pure functions over already-loaded ``AppState`` data, no API/HTTP
concerns, matching every other ``features/`` module's layering.

Every other filter on the Player Stats page (search, team, position, price -- D14) is client-side
over one bulk fetch, so nothing here takes those as parameters; the only server-side parameter is
the gameweek range, since it changes which stats get summed rather than just which rows are shown.

**Overperformance is a shrunk rate ratio, never a raw difference.** "Is this player beating their
expected stats, and is it real?" is answered by :class:`~engine.rates.RateRatio` (gamma-Poisson
posterior, see that module for the model), on two axes:

* *attacking*, actual goals + assists against summed xGI, for every position;
* *defensive*, actual clean sheets against summed per-match clean-sheet probability, for the
  positions a clean sheet actually pays (:data:`CLEAN_SHEET_RATIO_POSITIONS`).

Two decisions here are deliberate and worth not re-litigating:

**The defensive axis counts clean sheets, it does not average goals conceded.** Clean-sheet points
are a step function (``engine.scoring.CLEAN_SHEET_POINTS``, paid at zero conceded and nothing at
one or more), so a mean-based metric actively misranks: a defender conceding 0,0,0,6 has a worse
mean goals-conceded per 90 (1.5) than one conceding 1,1,1,1 (1.0), while scoring three clean
sheets to the other's zero. Only a count respects the thing that pays.

**The prior strength is fitted per position, not asserted.** A defender's xGI exposure per match
is roughly a quarter of a forward's, so one shared prior would either under-shrink forwards or
flatten every defender to exactly 1.00. :func:`fit_overperformance_priors` fits it from the full
season to date regardless of the range being displayed, since it is a population parameter and
estimating it from a one-gameweek view would be least stable exactly when shrinkage matters most.

Two approximations, both documented rather than hidden, and both erring toward "not yet a signal":

* Clean sheets are Poisson-binomial, not Poisson (``Var = sum p(1-p) < sum p``), so reusing the
  gamma-Poisson machinery gives slightly *wider* intervals than strictly necessary.
* :func:`expected_clean_sheets` uses ``exp(-xGC)``, omitting the Dixon-Coles low-score correction
  ``engine.models.clean_sheets`` applies in the forward model: retrospectively only the team's own
  xGC is available, not the opponent's lambda. It slightly understates clean-sheet probability,
  marginally overstating defensive luck.

Known gap, deliberate: goalkeeper save points are not modelled here at all. They are a large share
of keeper scoring but are driven mostly by opponent shot volume rather than keeper skill, and the
raw counts they would need were removed from this page. A keeper's save value stays visible via
their actual points total and the engine's own forward projections.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.aggregate import ComponentBreakdown
from engine.data.player_history import PlayerGameweekActual, actual_points_for_gameweek
from engine.rates import RateRatio, fit_rate_ratio_prior, rate_ratio_posterior
from engine.scoring import CLEAN_SHEET_POINTS

__all__ = [
    "SMALL_SAMPLE_APPS_THRESHOLD",
    "CLEAN_SHEET_RATIO_POSITIONS",
    "PlayerActualStats",
    "summarize_actual_stats",
    "expected_clean_sheets",
    "fit_overperformance_priors",
    "build_actual_stats_by_player",
]

# D12: fewer than this many gameweeks with minutes played in the selected range is flagged as a
# small sample, so a single good game doesn't read as an established trend.
SMALL_SAMPLE_APPS_THRESHOLD = 3

# Positions a clean-sheet ratio is meaningful for -- read off the scoring table rather than
# hardcoded, so it stays correct if the point values move. Midfielders are included because a
# clean sheet genuinely pays them (1 point); forwards earn nothing for one, so the ratio would be
# a number with no decision attached to it.
CLEAN_SHEET_RATIO_POSITIONS = frozenset(
    position for position, points in CLEAN_SHEET_POINTS.items() if points > 0
)


@dataclass(frozen=True)
class PlayerActualStats:
    """One player's actual output, summed over ``[gameweek_from, gameweek_to]`` -- raw counts
    (D13's "numbers a manager actually thinks in") plus the same points converted per component
    (``points_breakdown``, D13's direct-comparison-against-predictions half)."""

    gameweek_from: int
    gameweek_to: int
    apps: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_missed: int
    penalties_saved: int
    saves: int
    bonus: int
    yellow_cards: int
    red_cards: int
    total_points: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    points_breakdown: ComponentBreakdown
    # Ownership under whichever lens the caller supplied. The Player Stats page always passes a
    # mini-league percentage (or nothing at all, when the league can't be resolved), never FPL's
    # own population-wide figure, so the FPL-specific name this field used to carry would be
    # actively misleading.
    ownership_percent: float | None
    small_sample: bool
    # Actual-vs-expected ratios (see module docstring). None means "no exposure to judge on",
    # a real state rather than a zero -- and defensive_ratio is additionally None for any position
    # a clean sheet doesn't pay.
    attacking_ratio: RateRatio | None = None
    defensive_ratio: RateRatio | None = None
    # FPL's designated first-choice penalty taker. Penalty *volume* persists, penalty *conversion*
    # regresses hard toward the ~79% baseline, so a taker on a hot run posts an inflated attacking
    # ratio that means much less than the same number from open play.
    is_penalty_taker: bool = False


def summarize_actual_stats(
    history: Sequence[PlayerGameweekActual],
    position: str,
    gameweek_from: int,
    gameweek_to: int,
    ownership_percent: float | None = None,
    attacking_prior_k: float | None = None,
    defensive_prior_k: float | None = None,
    is_penalty_taker: bool = False,
) -> PlayerActualStats | None:
    """Sum one player's raw counts and points-per-component across ``[gameweek_from,
    gameweek_to]``. Returns ``None`` if the player has no recorded gameweek in that range --
    "not on the pitch (yet)" is a real state, not a caller error, and a zero-filled row would
    misread as "played and did nothing".

    Points are converted gameweek by gameweek (:func:`~engine.data.player_history.
    actual_points_for_gameweek`) *before* being summed, not summed from raw totals first -- see
    that function's own docstring for why goals-conceded penalty in particular depends on this
    order.

    The two ``*_prior_k`` values come from :func:`fit_overperformance_priors`, fitted over the
    whole population for this player's position. Omitting them (the default) simply leaves the
    ratios unpopulated, which keeps this function usable on its own for the raw-count summary it
    originally existed to produce.
    """
    in_range = [actual for actual in history if gameweek_from <= actual.gameweek <= gameweek_to]
    if not in_range:
        return None

    per_gameweek_points = [actual_points_for_gameweek(actual, position) for actual in in_range]
    points_breakdown = ComponentBreakdown(
        appearance=sum(p.appearance for p in per_gameweek_points),
        goals=sum(p.goals for p in per_gameweek_points),
        assists=sum(p.assists for p in per_gameweek_points),
        clean_sheet=sum(p.clean_sheet for p in per_gameweek_points),
        goals_conceded=sum(p.goals_conceded for p in per_gameweek_points),
        defensive_contribution=sum(p.defensive_contribution for p in per_gameweek_points),
        saves=sum(p.saves for p in per_gameweek_points),
        bonus=sum(p.bonus for p in per_gameweek_points),
        cards=sum(p.cards for p in per_gameweek_points),
        penalty_misses=sum(p.penalty_misses for p in per_gameweek_points),
        own_goals=sum(p.own_goals for p in per_gameweek_points),
    )
    apps = sum(1 for actual in in_range if actual.minutes > 0)

    return PlayerActualStats(
        gameweek_from=gameweek_from,
        gameweek_to=gameweek_to,
        apps=apps,
        minutes=sum(actual.minutes for actual in in_range),
        goals_scored=sum(actual.goals_scored for actual in in_range),
        assists=sum(actual.assists for actual in in_range),
        clean_sheets=sum(actual.clean_sheets for actual in in_range),
        goals_conceded=sum(actual.goals_conceded for actual in in_range),
        own_goals=sum(actual.own_goals for actual in in_range),
        penalties_missed=sum(actual.penalties_missed for actual in in_range),
        penalties_saved=sum(actual.penalties_saved for actual in in_range),
        saves=sum(actual.saves for actual in in_range),
        bonus=sum(actual.bonus for actual in in_range),
        yellow_cards=sum(actual.yellow_cards for actual in in_range),
        red_cards=sum(actual.red_cards for actual in in_range),
        total_points=sum(actual.total_points for actual in in_range),
        expected_goals=sum(actual.expected_goals for actual in in_range),
        expected_assists=sum(actual.expected_assists for actual in in_range),
        expected_goal_involvements=sum(actual.expected_goal_involvements for actual in in_range),
        expected_goals_conceded=sum(actual.expected_goals_conceded for actual in in_range),
        points_breakdown=points_breakdown,
        ownership_percent=ownership_percent,
        small_sample=apps < SMALL_SAMPLE_APPS_THRESHOLD,
        attacking_ratio=(
            rate_ratio_posterior(
                actual=sum(actual.goals_scored + actual.assists for actual in in_range),
                exposure=sum(actual.expected_goal_involvements for actual in in_range),
                k=attacking_prior_k,
            )
            if attacking_prior_k is not None
            else None
        ),
        defensive_ratio=(
            rate_ratio_posterior(
                actual=sum(actual.clean_sheets for actual in in_range),
                exposure=expected_clean_sheets(in_range),
                k=defensive_prior_k,
            )
            if defensive_prior_k is not None and position in CLEAN_SHEET_RATIO_POSITIONS
            else None
        ),
        is_penalty_taker=is_penalty_taker,
    )


def expected_clean_sheets(records: Sequence[PlayerGameweekActual]) -> float:
    """Expected clean sheets over these matches: the sum of each match's own clean-sheet
    probability, taken as ``exp(-xGC)`` (the Poisson probability the opponent scores zero).

    Summed per match and never derived from mean goals conceded -- see the module docstring for
    the 0,0,0,6 versus 1,1,1,1 case that makes a mean actively misrank. Matches with no minutes
    are skipped: a player on the bench was not exposed to that clean sheet either way.
    """
    return sum(
        math.exp(-record.expected_goals_conceded) for record in records if record.minutes > 0
    )


def fit_overperformance_priors(
    player_history: Mapping[int, Sequence[PlayerGameweekActual]],
    position_by_player: Mapping[int, str],
) -> tuple[dict[str, float], dict[str, float]]:
    """Fit the gamma-Poisson prior strength per position, over every gameweek supplied.

    Returns ``(attacking_k_by_position, defensive_k_by_position)``. Callers pass the *full season
    to date* here rather than the range being displayed: ``k`` describes how much genuine spread
    exists in a position's finishing or clean-sheet luck, which is a property of the population
    and not of whichever window the user happens to be looking at. Estimating it from a narrow
    view would be least reliable exactly when the shrinkage it drives matters most.

    A position with no detectable heterogeneity gets ``inf``, collapsing its ratios to 1.0 --
    see :func:`~engine.rates.fit_rate_ratio_prior` for why that is a finding rather than a
    failure.
    """
    attacking: dict[str, list[tuple[float, float]]] = {}
    defensive: dict[str, list[tuple[float, float]]] = {}

    for player_id, history in player_history.items():
        position = position_by_player.get(player_id)
        if position is None:
            continue
        played = [record for record in history if record.minutes > 0]
        if not played:
            continue

        attacking.setdefault(position, []).append(
            (
                sum(record.goals_scored + record.assists for record in played),
                sum(record.expected_goal_involvements for record in played),
            )
        )
        if position in CLEAN_SHEET_RATIO_POSITIONS:
            defensive.setdefault(position, []).append(
                (
                    sum(record.clean_sheets for record in played),
                    expected_clean_sheets(played),
                )
            )

    def _fit(pairs_by_position: Mapping[str, list[tuple[float, float]]]) -> dict[str, float]:
        return {
            position: fit_rate_ratio_prior(
                [actual for actual, _ in pairs], [exposure for _, exposure in pairs]
            )
            for position, pairs in pairs_by_position.items()
        }

    return _fit(attacking), _fit(defensive)


def build_actual_stats_by_player(
    player_history: Mapping[int, Sequence[PlayerGameweekActual]],
    position_by_player: Mapping[int, str],
    gameweek_from: int,
    gameweek_to: int,
    ownership_by_player: Mapping[int, float | None] | None = None,
    full_season_history: Mapping[int, Sequence[PlayerGameweekActual]] | None = None,
    penalty_takers: frozenset[int] | None = None,
) -> dict[int, PlayerActualStats]:
    """Every player with at least one recorded gameweek in range, summarized -- a player outside
    the range (or with no history at all, e.g. a brand new signing) is simply absent from the
    result, never a zero-filled row.

    ``full_season_history`` is what the overperformance priors are fitted on, independently of
    ``[gameweek_from, gameweek_to]``; omit it and the ratios are simply left unpopulated. Pass
    ``player_history`` itself if the caller genuinely has nothing wider available.
    """
    ownership_by_player = ownership_by_player or {}
    penalty_takers = penalty_takers or frozenset()

    attacking_k: Mapping[str, float] = {}
    defensive_k: Mapping[str, float] = {}
    if full_season_history is not None:
        attacking_k, defensive_k = fit_overperformance_priors(
            full_season_history, position_by_player
        )

    result: dict[int, PlayerActualStats] = {}
    for player_id, history in player_history.items():
        position = position_by_player.get(player_id)
        if position is None:
            continue
        summary = summarize_actual_stats(
            history,
            position,
            gameweek_from,
            gameweek_to,
            ownership_by_player.get(player_id),
            attacking_prior_k=attacking_k.get(position),
            defensive_prior_k=defensive_k.get(position),
            is_penalty_taker=player_id in penalty_takers,
        )
        if summary is not None:
            result[player_id] = summary
    return result
