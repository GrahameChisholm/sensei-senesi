"""Chips: per-chip value-now-vs-waiting evaluators over the shared planning horizon (BUILD_PLAN
Phase 4).

**Bench Boost / Triple Captain** compare this week's value against the *best week available
anywhere in the planning horizon* — not a historical rolling average. The actual question a
use-it-once chip needs answered is "is this the peak week within reach, or is patience worth
more," which a rolling-average-against-your-own-history comparison doesn't answer.

**Free Hit** reduces to the same "best week in the horizon" shape, just with a different value
signal: blank/double exposure across the squad rather than points. It reuses
``features.fixtures.fixture_counts_by_gameweek`` rather than re-deriving blank/double detection
(BUILD_PLAN 4: "don't duplicate blank/double-gameweek logic").

**Wildcard** doesn't fit that per-gameweek shape — its effect persists across the rest of the
half-season rather than being tied to one target gameweek, so it's evaluated as a single "how far
below optimal is my current squad, right now" drift metric instead, reusing
``features.transfers.find_transfer_candidates`` as a proxy (see :func:`evaluate_wildcard`'s
docstring for why this is a lower-bound heuristic, not the true optimal-squad gap).

**2026/27 chip allowance (BUILD_PLAN 4).** Eight chips total, one full set per half of the
season; only one playable per gameweek. Free Hit specifically can't be played in gameweek 1, and
if used in gameweek 19 the second Free Hit can't be played in gameweek 20 — that second rule
depends on full-season chip-usage history this module doesn't track, so it's the caller's
responsibility to gate on, not this evaluator's (see :data:`FREE_HIT_BLOCKED_GAMEWEEKS`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.projections import PlayerHorizonProjection
from features.fixtures import TeamFixture, fixture_counts_by_gameweek
from features.team_state import MyTeamState
from features.transfers import find_transfer_candidates

__all__ = [
    "FREE_HIT_BLOCKED_GAMEWEEKS",
    "WILDCARD_UPLIFT_THRESHOLD",
    "ChipEvaluation",
    "WildcardEvaluation",
    "bench_points_by_gameweek",
    "best_eligible_points_by_gameweek",
    "blank_exposure_by_gameweek",
    "evaluate_bench_boost",
    "evaluate_triple_captain",
    "evaluate_free_hit",
    "evaluate_wildcard",
]

# Free Hit can't be played in gameweek 1 (BUILD_PLAN 4 -- no prior fixture swing exists to weigh
# it against). The GW19-used -> GW20-blocked interaction is NOT encoded here: it depends on
# whether *this specific team* already used its first-half Free Hit in gameweek 19, which needs
# full chip-usage history beyond a single evaluator call -- the caller must gate on that itself.
FREE_HIT_BLOCKED_GAMEWEEKS = (1,)

# Materiality bar (points, summed across the horizon) for recommending Wildcard now rather than
# holding -- deliberately set above a single transfer's -4 hit cost (roughly 2x it) so the chip is
# only recommended when it clearly beats just taking a hit or two. Not fitted; revisit with real
# backtesting evidence once enough live gameweeks accumulate.
WILDCARD_UPLIFT_THRESHOLD = 8.0


@dataclass(frozen=True)
class ChipEvaluation:
    """One chip's value-now-vs-best-week-in-horizon verdict. ``value_now``/``best_value`` share
    whatever unit that chip's evaluator uses (points for Bench Boost/Triple Captain, a blank-
    exposure count for Free Hit) -- only meaningful compared within the same chip's own values."""

    chip: str
    target_gameweek: int
    value_now: float
    best_gameweek: int
    best_value: float
    recommendation: str  # "play_now" | "wait"
    reasoning: str


def _evaluate_value_now_vs_best(
    chip: str, values_by_gameweek: Mapping[int, float], target_gameweek: int, value_label: str
) -> ChipEvaluation:
    if not values_by_gameweek:
        raise ValueError("values_by_gameweek must not be empty")
    if target_gameweek not in values_by_gameweek:
        raise ValueError(f"target_gameweek {target_gameweek} not in values_by_gameweek")

    best_gameweek = max(values_by_gameweek, key=lambda gw: values_by_gameweek[gw])
    best_value = values_by_gameweek[best_gameweek]
    value_now = values_by_gameweek[target_gameweek]

    if best_gameweek == target_gameweek:
        recommendation = "play_now"
        reasoning = (
            f"{value_label} peaks at GW{target_gameweek} ({value_now:.1f}) within the horizon"
        )
    else:
        recommendation = "wait"
        reasoning = (
            f"{value_label} at GW{target_gameweek} is {value_now:.1f}, but GW{best_gameweek} "
            f"looks stronger ({best_value:.1f}) within the horizon"
        )

    return ChipEvaluation(
        chip=chip,
        target_gameweek=target_gameweek,
        value_now=value_now,
        best_gameweek=best_gameweek,
        best_value=best_value,
        recommendation=recommendation,
        reasoning=reasoning,
    )


def bench_points_by_gameweek(
    my_team: MyTeamState, projections: Mapping[int, PlayerHorizonProjection]
) -> dict[int, float]:
    """Sum of the bench's expected points, per gameweek in the horizon -- the value Bench Boost
    unlocks that gameweek. Players missing from ``projections`` (e.g. an unmatched crosswalk
    entry) are silently skipped rather than raising, since a missing single bench player shouldn't
    block the whole evaluation."""
    totals: dict[int, float] = {}
    for player_id in my_team.bench_order:
        horizon = projections.get(player_id)
        if horizon is None:
            continue
        for gameweek, gw_projection in horizon.gameweeks.items():
            totals[gameweek] = totals.get(gameweek, 0.0) + gw_projection.expected_points
    return totals


def best_eligible_points_by_gameweek(
    my_team: MyTeamState, projections: Mapping[int, PlayerHorizonProjection]
) -> dict[int, float]:
    """The highest expected-points figure among the starting XI, per gameweek -- the marginal
    gain Triple Captain adds on top of the normal 2x multiplier (captaincy.py's own top-EV pick
    for that gameweek, recomputed per gameweek across the horizon rather than duplicating
    captaincy.py's ranking machinery for a single number)."""
    bests: dict[int, float] = {}
    for player_id in my_team.starting_xi:
        horizon = projections.get(player_id)
        if horizon is None:
            continue
        for gameweek, gw_projection in horizon.gameweeks.items():
            bests[gameweek] = max(bests.get(gameweek, 0.0), gw_projection.expected_points)
    return bests


def blank_exposure_by_gameweek(
    my_team: MyTeamState,
    team_id_by_player: Mapping[int, int],
    fixtures: Sequence[TeamFixture],
    gameweeks: Sequence[int],
) -> dict[int, int]:
    """Count of squad players whose team has zero fixtures, per gameweek -- the exposure Free Hit
    exists to patch over. Reuses :func:`features.fixtures.fixture_counts_by_gameweek` per distinct
    squad team rather than re-deriving blank detection here (BUILD_PLAN 4)."""
    counts = dict.fromkeys(gameweeks, 0)
    squad_team_ids = {
        team_id_by_player[player_id]
        for player_id in my_team.player_ids
        if player_id in team_id_by_player
    }
    for team_id in squad_team_ids:
        team_fixture_counts = fixture_counts_by_gameweek(fixtures, team_id, gameweeks)
        squad_size_for_team = sum(
            1 for player_id in my_team.player_ids if team_id_by_player.get(player_id) == team_id
        )
        for gameweek, fixture_count in team_fixture_counts.items():
            if fixture_count == 0:
                counts[gameweek] += squad_size_for_team
    return counts


def evaluate_bench_boost(
    my_team: MyTeamState,
    projections: Mapping[int, PlayerHorizonProjection],
    target_gameweek: int,
) -> ChipEvaluation:
    values = bench_points_by_gameweek(my_team, projections)
    return _evaluate_value_now_vs_best("bench_boost", values, target_gameweek, "bench EV")


def evaluate_triple_captain(
    my_team: MyTeamState,
    projections: Mapping[int, PlayerHorizonProjection],
    target_gameweek: int,
) -> ChipEvaluation:
    values = best_eligible_points_by_gameweek(my_team, projections)
    return _evaluate_value_now_vs_best("triple_captain", values, target_gameweek, "best captain EV")


def evaluate_free_hit(
    my_team: MyTeamState,
    team_id_by_player: Mapping[int, int],
    fixtures: Sequence[TeamFixture],
    gameweeks: Sequence[int],
    target_gameweek: int,
) -> ChipEvaluation:
    if target_gameweek in FREE_HIT_BLOCKED_GAMEWEEKS:
        raise ValueError(f"Free Hit cannot be played in gameweek {target_gameweek}")
    values = blank_exposure_by_gameweek(my_team, team_id_by_player, fixtures, gameweeks)
    return _evaluate_value_now_vs_best(
        "free_hit", {gw: float(v) for gw, v in values.items()}, target_gameweek, "blank exposure"
    )


@dataclass(frozen=True)
class WildcardEvaluation:
    """Wildcard's "squad value drifting below optimal" signal (BUILD_PLAN 4) -- a single snapshot
    rather than a per-gameweek pick, since a wildcard's effect persists across the rest of the
    half-season instead of being tied to one target gameweek's fixture swing."""

    squad_uplift: float
    upgradeable_slots: int
    recommendation: str  # "play_now" | "hold"
    reasoning: str


def evaluate_wildcard(
    my_team: MyTeamState,
    current_projections: Mapping[int, PlayerHorizonProjection],
    candidate_pool: Mapping[int, PlayerHorizonProjection],
    buy_prices: Mapping[int, int],
    uplift_threshold: float = WILDCARD_UPLIFT_THRESHOLD,
) -> WildcardEvaluation:
    """Approximate "how far below optimal is my current squad" by summing, for each squad slot,
    the best available position-matched single-swap upgrade (reusing
    :func:`features.transfers.find_transfer_candidates` as a proxy, hit-free since Wildcard
    removes the -4 cost).

    **This is a lower-bound heuristic, not the true optimal-squad gap.** A real wildcard
    reshuffles the entire budget across all 15 slots at once with no per-slot budget constraint;
    finding the true optimum is a much larger combinatorial search (formation- and budget-
    constrained over the full player pool) that's out of v1 scope, matching transfers.py's own
    documented greedy-not-combinatorial cut.
    """
    plan = find_transfer_candidates(my_team, current_projections, candidate_pool, buy_prices)

    best_gain_by_sell_id: dict[int, float] = {}
    for candidate in plan.affordable_candidates:
        if candidate.points_gain <= 0:
            continue
        current_best = best_gain_by_sell_id.get(candidate.sell_player_id, 0.0)
        if candidate.points_gain > current_best:
            best_gain_by_sell_id[candidate.sell_player_id] = candidate.points_gain

    squad_uplift = sum(best_gain_by_sell_id.values())
    recommendation = "play_now" if squad_uplift >= uplift_threshold else "hold"
    reasoning = (
        f"Best available hit-free upgrades sum to +{squad_uplift:.1f} pts over the horizon "
        f"across {len(best_gain_by_sell_id)} squad slot(s)"
    )
    reasoning += (
        f" -- above the {uplift_threshold:.1f} pt materiality bar"
        if recommendation == "play_now"
        else f" -- below the {uplift_threshold:.1f} pt materiality bar, hold for now"
    )

    return WildcardEvaluation(
        squad_uplift=squad_uplift,
        upgradeable_slots=len(best_gain_by_sell_id),
        recommendation=recommendation,
        reasoning=reasoning,
    )
