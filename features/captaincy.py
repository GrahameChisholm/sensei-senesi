"""Captaincy: rank the full player pool by EV/floor/ceiling, highlight owned/eligible options
(BUILD_PLAN Phase 4).

Ranks every player in ``projections``, not just the user's own 15 — seeing how your best captain
option stacks up against the whole league is useful context for gauging differentials and
spotting transfer targets, even though the armband itself can only ever go to a player you
actually own and start (an FPL structural rule, not a design choice here). "Eligible" mirrors the
same definition BUILD_PLAN 3.2's captaincy hit-rate backtest uses — your actual starting XI, not
the full 15, since bench players were never really in contention for the armband.

**No single headline pick.** Deliberately kept simple rather than mini-league-rank-aware: this
module surfaces the highest-EV, floor (safe), and ceiling (punt) picks side by side and lets the
caller/UI choose, rather than the engine picking one "the" recommendation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from engine.aggregate import ComponentBreakdown
from engine.projections import PlayerGameweekProjection
from features.team_state import MyTeamState

__all__ = [
    "CaptaincyOption",
    "CaptaincyRecommendation",
    "rank_captaincy_pool",
]

# How many of a breakdown's nonzero positive-contribution lines to name in the reasoning string —
# kept small so the reasoning stays a one-line human summary, not a full breakdown dump (the web
# app's "detail on click" view is where the full breakdown belongs, per BUILD_PLAN 2.7).
_REASONING_TOP_N_COMPONENTS = 2


def _top_components(
    breakdown: ComponentBreakdown, n: int = _REASONING_TOP_N_COMPONENTS
) -> list[tuple[str, float]]:
    lines = [
        ("appearance", breakdown.appearance),
        ("goals", breakdown.goals),
        ("assists", breakdown.assists),
        ("clean sheet", breakdown.clean_sheet),
        ("defensive contribution", breakdown.defensive_contribution),
        ("saves", breakdown.saves),
        ("bonus", breakdown.bonus),
    ]
    positive = [(name, value) for name, value in lines if value > 0]
    positive.sort(key=lambda item: item[1], reverse=True)
    return positive[:n]


def _reasoning(projection: PlayerGameweekProjection) -> str:
    top = _top_components(projection.breakdown)
    driven_by = ", ".join(f"{name} {value:.1f}" for name, value in top)
    driven_by = driven_by or "no material positive contribution"
    summary = (
        f"{projection.expected_points:.1f} EV (GW{projection.gameweek}), driven by {driven_by}"
    )
    if projection.simulation is not None:
        sim = projection.simulation
        summary += (
            f"; floor {sim.floor:.1f}, ceiling {sim.ceiling:.1f}, P(10+) {sim.prob_big_haul:.0%}"
        )
    return summary


@dataclass(frozen=True)
class CaptaincyOption:
    """One player's captaincy candidacy for one gameweek. ``expected_points`` is always populated
    (the deterministic component breakdown, BUILD_PLAN 2.7); ``floor``/``ceiling``/
    ``prob_big_haul`` are ``None`` when this projection wasn't run through the full Monte Carlo
    simulation (2.9) for this gameweek."""

    player_id: int
    position: str
    expected_points: float
    floor: float | None
    ceiling: float | None
    prob_big_haul: float | None
    is_owned: bool
    is_eligible: bool  # in the starting XI -- the only players who could actually get the armband
    reasoning: str


@dataclass(frozen=True)
class CaptaincyRecommendation:
    """The full pool ranked by EV, plus the three side-by-side eligible picks (BUILD_PLAN 4: "no
    single headline pick"). Any of the three picks is ``None`` if no eligible player in the pool
    has a computed value for it (e.g. no starting-XI player was simulated, for floor/ceiling)."""

    ranked_pool: tuple[CaptaincyOption, ...]  # every player in `projections`, sorted by EV desc
    top_ev_pick: CaptaincyOption | None
    safe_pick: CaptaincyOption | None  # eligible, highest floor
    punt_pick: CaptaincyOption | None  # eligible, highest ceiling


def rank_captaincy_pool(
    my_team: MyTeamState,
    projections: Sequence[PlayerGameweekProjection],
) -> CaptaincyRecommendation:
    """Build the full-pool captaincy ranking plus the three eligible-only picks.

    ``projections`` is one :class:`~engine.projections.PlayerGameweekProjection` per player in
    whatever pool is being considered (the full league, or any subset a caller wants ranked) —
    this function doesn't fetch or filter the pool itself, only ranks and annotates it.
    """
    if not projections:
        raise ValueError("projections must not be empty")

    owned = set(my_team.player_ids)
    eligible = set(my_team.starting_xi)

    options = [
        CaptaincyOption(
            player_id=projection.player_id,
            position=projection.position,
            expected_points=projection.expected_points,
            floor=projection.simulation.floor if projection.simulation is not None else None,
            ceiling=projection.simulation.ceiling if projection.simulation is not None else None,
            prob_big_haul=(
                projection.simulation.prob_big_haul if projection.simulation is not None else None
            ),
            is_owned=projection.player_id in owned,
            is_eligible=projection.player_id in eligible,
            reasoning=_reasoning(projection),
        )
        for projection in projections
    ]

    ranked_pool = tuple(sorted(options, key=lambda option: option.expected_points, reverse=True))
    eligible_options = [option for option in ranked_pool if option.is_eligible]

    top_ev_pick = max(eligible_options, key=lambda option: option.expected_points, default=None)
    safe_pick = max(
        (option for option in eligible_options if option.floor is not None),
        key=lambda option: option.floor,
        default=None,
    )
    punt_pick = max(
        (option for option in eligible_options if option.ceiling is not None),
        key=lambda option: option.ceiling,
        default=None,
    )

    return CaptaincyRecommendation(
        ranked_pool=ranked_pool,
        top_ev_pick=top_ev_pick,
        safe_pick=safe_pick,
        punt_pick=punt_pick,
    )
