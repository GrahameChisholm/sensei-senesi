"""Cards model — historical yellow/red rates -> expected point deductions (2.6).

Small but real, especially for aggressive defenders and midfielders. Referee assignment is a
genuine, quantifiable, knowable-in-advance factor (referees vary measurably in card strictness,
appointments are announced days ahead), but is **deliberately not modelled for v1** — cards are
one of the smallest components in the engine and the added data-sourcing/maintenance cost isn't
worth the marginal accuracy here (BUILD_PLAN 2.6). Revisit only if backtesting shows cards
materially affecting captaincy or differential calls.
"""

from __future__ import annotations

from dataclasses import dataclass

from engine.scoring import RED_CARD_POINTS, YELLOW_CARD_POINTS


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
