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

from engine.rates import shrink_toward_prior
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
    *,
    individual_weight: float | None = None,
    league_avg_yellow_card_rate_per_90: float | None = None,
    league_avg_red_card_rate_per_90: float | None = None,
    yellow_shrinkage_k: float = 0.0,
    red_shrinkage_k: float = 0.0,
) -> CardsProjection:
    """Scale each historical per-90 card rate by expected minutes — a substitute on for 20
    minutes has correspondingly less exposure to picking up a card.

    Shrinkage toward each rate's own league-average-by-position prior
    (ENGINE_IMPROVEMENTS_3.md A.3) only kicks in when the caller supplies ``individual_weight``
    and the relevant ``league_avg_*_card_rate_per_90`` and shrinkage-``k`` — omitting them uses
    each rate unmodified, matching every existing caller and the same opt-in shape
    ``engine.models.goals.project_goals``/``engine.models.assists.project_assists`` already use.
    Red and yellow shrink independently (typically with a much larger ``red_shrinkage_k``, since a
    red card is a near-unique event whose per-90 rate is meaningless from a single observation —
    a real point-in-time pull showed an unshrunk red-card rate as high as 22.9 per 90 minutes from
    one dismissal in a 3-minute cameo).
    """
    if yellow_card_rate_per_90 < 0 or red_card_rate_per_90 < 0:
        raise ValueError("card rates must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    effective_yellow_rate = yellow_card_rate_per_90
    if (
        individual_weight is not None
        and league_avg_yellow_card_rate_per_90 is not None
        and yellow_shrinkage_k > 0
    ):
        effective_yellow_rate = shrink_toward_prior(
            yellow_card_rate_per_90,
            individual_weight,
            league_avg_yellow_card_rate_per_90,
            yellow_shrinkage_k,
        )
    effective_red_rate = red_card_rate_per_90
    if (
        individual_weight is not None
        and league_avg_red_card_rate_per_90 is not None
        and red_shrinkage_k > 0
    ):
        effective_red_rate = shrink_toward_prior(
            red_card_rate_per_90,
            individual_weight,
            league_avg_red_card_rate_per_90,
            red_shrinkage_k,
        )
    minutes_scaling = expected_minutes / 90.0
    return CardsProjection(
        expected_yellow_cards=effective_yellow_rate * minutes_scaling,
        expected_red_cards=effective_red_rate * minutes_scaling,
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
    *,
    individual_weight: float | None = None,
    league_avg_own_goal_rate_per_90: float | None = None,
    shrinkage_k: float = 0.0,
) -> OwnGoalProjection:
    """Scale the historical own-goal per-90 rate by expected minutes, same as cards — real but
    rare enough that no opponent/fixture adjustment is worth the added complexity (BUILD_PLAN
    2.6's own reasoning for cards applies identically here).

    Shrinkage toward the league-average-by-position prior (ENGINE_IMPROVEMENTS_3.md A.3), same
    opt-in shape as :func:`project_cards` — own goals are rarer even than red cards, so a thin
    individual sample should lean almost entirely on the prior."""
    if own_goal_rate_per_90 < 0:
        raise ValueError("own_goal_rate_per_90 must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    effective_rate = own_goal_rate_per_90
    if (
        individual_weight is not None
        and league_avg_own_goal_rate_per_90 is not None
        and shrinkage_k > 0
    ):
        effective_rate = shrink_toward_prior(
            own_goal_rate_per_90, individual_weight, league_avg_own_goal_rate_per_90, shrinkage_k
        )
    minutes_scaling = expected_minutes / 90.0
    return OwnGoalProjection(expected_own_goals=effective_rate * minutes_scaling)
