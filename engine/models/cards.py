"""Cards and own goals — historical per-90 rates -> expected point deductions (2.6).

Small but real, especially for aggressive defenders and midfielders. Referee assignment is a
genuine, quantifiable, knowable-in-advance factor (referees vary measurably in card strictness,
appointments are announced days ahead), but is **deliberately not modelled for v1** — cards are
one of the smallest components in the engine and the added data-sourcing/maintenance cost isn't
worth the marginal accuracy here (BUILD_PLAN 2.6). Revisit only if backtesting shows cards
materially affecting captaincy or differential calls.

Own goals (ENGINE_IMPROVEMENTS_2.md D.6) are modelled the same way, for the same reason: a small,
real, rare deduction with no principled reason to omit it once the per-90 rate is already sitting
in the ingested data unused. Kept in this module rather than a new file since it's structurally
identical to cards — a historical per-90 rate scaled by minutes exposure — not because it's
conceptually a card.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.scoring import OWN_GOAL_POINTS, RED_CARD_POINTS, YELLOW_CARD_POINTS


@dataclass(frozen=True)
class CardsProjection:
    """A player's cards-component projection for one gameweek."""

    expected_yellow_cards: float
    expected_red_cards: float

    def __post_init__(self) -> None:
        if self.expected_yellow_cards < 0:
            raise ValueError("expected_yellow_cards must be non-negative")
        if self.expected_red_cards < 0:
            raise ValueError("expected_red_cards must be non-negative")

    @property
    def expected_points(self) -> float:
        return (
            self.expected_yellow_cards * YELLOW_CARD_POINTS
            + self.expected_red_cards * RED_CARD_POINTS
        )


def project_cards(
    yellow_card_rate_per_90: float,
    red_card_rate_per_90: float,
    expected_minutes: float = 90.0,
) -> CardsProjection:
    """Scale each historical per-90 card rate by expected minutes — a substitute on for 20
    minutes has correspondingly less exposure to picking up a card."""
    if yellow_card_rate_per_90 < 0 or red_card_rate_per_90 < 0:
        raise ValueError("card rates must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    minutes_scaling = expected_minutes / 90.0
    return CardsProjection(
        expected_yellow_cards=yellow_card_rate_per_90 * minutes_scaling,
        expected_red_cards=red_card_rate_per_90 * minutes_scaling,
    )


@dataclass(frozen=True)
class OwnGoalProjection:
    """A player's own-goals-component projection for one gameweek (ENGINE_IMPROVEMENTS_2.md D.6)."""

    expected_own_goals: float

    def __post_init__(self) -> None:
        if self.expected_own_goals < 0:
            raise ValueError("expected_own_goals must be non-negative")

    @property
    def expected_points(self) -> float:
        return self.expected_own_goals * OWN_GOAL_POINTS


def project_own_goals(
    own_goal_rate_per_90: float,
    expected_minutes: float = 90.0,
) -> OwnGoalProjection:
    """Scale the historical own-goal per-90 rate by expected minutes, same as cards — real but
    rare enough that no opponent/fixture adjustment is worth the added complexity (BUILD_PLAN
    2.6's own reasoning for cards applies identically here)."""
    if own_goal_rate_per_90 < 0:
        raise ValueError("own_goal_rate_per_90 must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    minutes_scaling = expected_minutes / 90.0
    return OwnGoalProjection(expected_own_goals=own_goal_rate_per_90 * minutes_scaling)
