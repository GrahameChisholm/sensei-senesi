"""Sum all components, gated/scaled by minutes, into a single expected-points number (2.7).

Every component (2.1-2.6) has already applied its own minutes gating/scaling internally where the
functional form calls for it (goals/assists/defensive-contribution/cards scale their own rate by
``expected_minutes / 90``; clean-sheet points are gated by ``p_60_plus``). This module's job is
purely additive: combine already-computed component projections into one
:class:`ComponentBreakdown` per gameweek, and roll several gameweeks up into one
:class:`PlayerProjection` over the planning horizon.

**Why this recomputes each line from the underlying projection fields rather than calling each
component's own ``expected_points``.** A few component modules bundle two logical lines from the
BUILD_PLAN 2.7 decomposition into one convenience property for standalone use (e.g.
:class:`~engine.models.goals.GoalProjection` bundles goal points *and* the penalty-miss deduction;
:class:`~engine.models.clean_sheets.CleanSheetProjection` bundles clean-sheet points *and* the
goals-conceded deduction). The web app's "detail on click" view and debugging both depend on
seeing every line of the BUILD_PLAN 2.7 decomposition separately, so this module always
reconstructs the fine-grained breakdown from each projection's raw fields instead.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.models.assists import AssistProjection
from engine.models.bonus import BonusProjection
from engine.models.cards import CardsProjection
from engine.models.clean_sheets import CleanSheetProjection
from engine.models.defensive_contribution import DefensiveContributionProjection
from engine.models.goals import GoalProjection
from engine.models.minutes import MinutesDistribution
from engine.models.saves import SavesProjection
from engine.scoring import (
    APPEARANCE_POINTS,
    CLEAN_SHEET_POINTS,
    GK,
    GOAL_POINTS,
    GOALS_CONCEDED_POSITIONS,
    PENALTY_MISS_POINTS,
)

DEFAULT_PLANNING_HORIZON = 5


@dataclass(frozen=True)
class ComponentBreakdown:
    """One line per term of the BUILD_PLAN 2.7 additive decomposition, for one player in one
    gameweek. Deduction lines (``goals_conceded``, ``cards``, ``penalty_misses``) are already
    signed non-positive."""

    appearance: float
    goals: float
    assists: float
    clean_sheet: float
    goals_conceded: float
    defensive_contribution: float
    saves: float
    bonus: float
    cards: float
    penalty_misses: float

    @property
    def total(self) -> float:
        return (
            self.appearance
            + self.goals
            + self.assists
            + self.clean_sheet
            + self.goals_conceded
            + self.defensive_contribution
            + self.saves
            + self.bonus
            + self.cards
            + self.penalty_misses
        )


def aggregate_gameweek(
    position: str,
    minutes: MinutesDistribution,
    goals: GoalProjection,
    assists: AssistProjection,
    clean_sheet: CleanSheetProjection,
    bonus: BonusProjection,
    cards: CardsProjection,
    defensive_contribution: DefensiveContributionProjection | None = None,
    saves: SavesProjection | None = None,
) -> ComponentBreakdown:
    """Combine one gameweek's worth of component projections into a single breakdown.

    ``defensive_contribution`` is required for every position except GK (not modelled there —
    BUILD_PLAN 2.5) and ``saves`` is required for GK only (BUILD_PLAN 2.6) — passing the wrong one
    for a position is almost always a wiring bug upstream, so this raises rather than silently
    zeroing it.
    """
    if position == GK:
        if defensive_contribution is not None:
            raise ValueError("defensive contribution is not modelled for GK")
        if saves is None:
            raise ValueError("saves projection is required for GK")
    else:
        if defensive_contribution is None:
            raise ValueError(f"defensive contribution projection is required for {position}")
        if saves is not None:
            raise ValueError(f"saves projection is not applicable to {position}")

    appearance_points = (
        minutes.p_1_to_59 * APPEARANCE_POINTS["1-59"] + minutes.p_60_plus * APPEARANCE_POINTS["60+"]
    )
    goals_points = goals.expected_goals * GOAL_POINTS[position]
    penalty_miss_points = goals.expected_penalty_misses * PENALTY_MISS_POINTS
    clean_sheet_points = (
        clean_sheet.clean_sheet_probability * CLEAN_SHEET_POINTS[position] * minutes.p_60_plus
    )
    goals_conceded_points = (
        clean_sheet.expected_goals_conceded_penalty if position in GOALS_CONCEDED_POSITIONS else 0.0
    )

    return ComponentBreakdown(
        appearance=appearance_points,
        goals=goals_points,
        assists=assists.expected_points,
        clean_sheet=clean_sheet_points,
        goals_conceded=goals_conceded_points,
        defensive_contribution=(
            defensive_contribution.expected_points if defensive_contribution is not None else 0.0
        ),
        saves=saves.expected_points if saves is not None else 0.0,
        bonus=bonus.expected_points,
        cards=cards.expected_points,
        penalty_misses=penalty_miss_points,
    )


@dataclass(frozen=True)
class PlayerProjection:
    """A player's full projection over the planning horizon — one :class:`ComponentBreakdown` per
    gameweek, keyed by gameweek number, plus the rolled-up horizon total."""

    player_id: int
    position: str
    gameweek_breakdowns: dict[int, ComponentBreakdown]

    def __post_init__(self) -> None:
        if not self.gameweek_breakdowns:
            raise ValueError("gameweek_breakdowns must not be empty")

    @property
    def per_gameweek_points(self) -> dict[int, float]:
        return {gw: breakdown.total for gw, breakdown in self.gameweek_breakdowns.items()}

    @property
    def horizon_total_points(self) -> float:
        return sum(breakdown.total for breakdown in self.gameweek_breakdowns.values())


def aggregate_horizon(
    player_id: int, position: str, gameweek_breakdowns: dict[int, ComponentBreakdown]
) -> PlayerProjection:
    """Roll several gameweeks' breakdowns up into one :class:`PlayerProjection` over the planning
    horizon (BUILD_PLAN 2.7 — 5 gameweeks by default, see :data:`DEFAULT_PLANNING_HORIZON`)."""
    return PlayerProjection(
        player_id=player_id, position=position, gameweek_breakdowns=dict(gameweek_breakdowns)
    )
