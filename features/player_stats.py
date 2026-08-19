"""Actual-performance summarization for the Player Stats page (PLAYER_STATS_PLAN D2/D12/D13) --
sums a player's per-gameweek actual counts and points (``engine.data.player_history``) over a
gameweek range picked in the UI. Pure functions over already-loaded ``AppState`` data, no API/HTTP
concerns, matching every other ``features/`` module's layering.

Every other filter on the Player Stats page (search, team, position, price -- D14) is client-side
over one bulk fetch, so nothing here takes those as parameters; the only server-side parameter is
the gameweek range, since it changes which stats get summed rather than just which rows are shown.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.aggregate import ComponentBreakdown
from engine.data.player_history import PlayerGameweekActual, actual_points_for_gameweek

__all__ = [
    "SMALL_SAMPLE_APPS_THRESHOLD",
    "PlayerActualStats",
    "summarize_actual_stats",
    "build_actual_stats_by_player",
]

# D12: fewer than this many gameweeks with minutes played in the selected range is flagged as a
# small sample, so a single good game doesn't read as an established trend.
SMALL_SAMPLE_APPS_THRESHOLD = 3


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
    selected_by_percent: float | None
    small_sample: bool


def summarize_actual_stats(
    history: Sequence[PlayerGameweekActual],
    position: str,
    gameweek_from: int,
    gameweek_to: int,
    selected_by_percent: float | None = None,
) -> PlayerActualStats | None:
    """Sum one player's raw counts and points-per-component across ``[gameweek_from,
    gameweek_to]``. Returns ``None`` if the player has no recorded gameweek in that range --
    "not on the pitch (yet)" is a real state, not a caller error, and a zero-filled row would
    misread as "played and did nothing".

    Points are converted gameweek by gameweek (:func:`~engine.data.player_history.
    actual_points_for_gameweek`) *before* being summed, not summed from raw totals first -- see
    that function's own docstring for why goals-conceded penalty in particular depends on this
    order.
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
        selected_by_percent=selected_by_percent,
        small_sample=apps < SMALL_SAMPLE_APPS_THRESHOLD,
    )


def build_actual_stats_by_player(
    player_history: Mapping[int, Sequence[PlayerGameweekActual]],
    position_by_player: Mapping[int, str],
    gameweek_from: int,
    gameweek_to: int,
    ownership_by_player: Mapping[int, float | None] | None = None,
) -> dict[int, PlayerActualStats]:
    """Every player with at least one recorded gameweek in range, summarized -- a player outside
    the range (or with no history at all, e.g. a brand new signing) is simply absent from the
    result, never a zero-filled row."""
    ownership_by_player = ownership_by_player or {}
    result: dict[int, PlayerActualStats] = {}
    for player_id, history in player_history.items():
        position = position_by_player.get(player_id)
        if position is None:
            continue
        summary = summarize_actual_stats(
            history, position, gameweek_from, gameweek_to, ownership_by_player.get(player_id)
        )
        if summary is not None:
            result[player_id] = summary
    return result
