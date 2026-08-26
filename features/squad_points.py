"""One function behind the header's predicted-points number, every pitch card's per-gameweek
value, and every chip preview — a pure read over a squad plus its projections. No squad-editing
concept lives here at all, this module only ever answers "how many points does this specific
squad, captain, and chip choice score," which is exactly why previewing a chip needs no separate
step: it's just a different ``chip`` argument to the same read-only call.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.projections import PlayerHorizonProjection
from features.team_state import MyTeamState

__all__ = ["CHIP_BENCH_BOOST", "CHIP_TRIPLE_CAPTAIN", "CHIPS", "SquadPoints", "projected_points"]

CHIP_BENCH_BOOST = "bench_boost"
CHIP_TRIPLE_CAPTAIN = "triple_captain"
# The only two chips left in the sandbox model -- plain, always-available toggles on how points
# are displayed, with no scarcity and no usage tracking (Wildcard/Free Hit no longer exist).
CHIPS = (CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN)


@dataclass(frozen=True)
class SquadPoints:
    """One squad's predicted total under one chip choice (or none). ``per_player``/
    ``per_gameweek`` are already post-multiplier (captain doubling/tripling, bench boost's extra
    four players) — exactly what the pitch cards and header both render directly."""

    total: float
    starting_xi_points: float
    bench_points: float
    captain_bonus: float
    per_player: dict[int, float]
    per_gameweek: dict[int, float]
    missing_player_ids: tuple[int, ...]


def _expected_points(
    player_id: int, gameweek: int, projections: Mapping[int, PlayerHorizonProjection]
) -> float | None:
    horizon = projections.get(player_id)
    if horizon is None or gameweek not in horizon.gameweeks:
        return None
    return horizon.gameweeks[gameweek].expected_points


def projected_points(
    state: MyTeamState,
    projections: Mapping[int, PlayerHorizonProjection],
    gameweeks: Sequence[int],
    chip: str | None = None,
) -> SquadPoints:
    """Sum of ``state``'s expected points across ``gameweeks``, captain doubled (or tripled under
    Triple Captain) and the bench added at full weight under Bench Boost. ``chip`` is a plain,
    always-available toggle with no scarcity or usage tracking — pass ``None``, ``"bench_boost"``,
    or ``"triple_captain"``; anything else raises ``ValueError``. Chip **stacking** needs no
    explicit rejection: this function only ever takes one ``chip`` argument, so it can't even be
    expressed.

    A player missing a projection for a requested gameweek (a blank gameweek, or a cold-start gap)
    contributes nothing for that gameweek and is reported in ``missing_player_ids`` — never
    silently counted as a real zero-point forecast.
    """
    if chip is not None and chip not in CHIPS:
        raise ValueError(f"unknown chip: {chip!r}")

    captain_multiplier = 3.0 if chip == CHIP_TRIPLE_CAPTAIN else 2.0
    include_bench = chip == CHIP_BENCH_BOOST

    missing_ids: set[int] = set()
    per_player: dict[int, float] = {}
    per_gameweek: dict[int, float] = dict.fromkeys(gameweeks, 0.0)

    def base_total(player_id: int) -> float:
        total = 0.0
        for gameweek in gameweeks:
            points = _expected_points(player_id, gameweek, projections)
            if points is None:
                missing_ids.add(player_id)
                continue
            total += points
            per_gameweek[gameweek] += points
        return total

    for player_id in state.starting_xi:
        per_player[player_id] = base_total(player_id)

    captain_base = per_player.get(state.captain_id, 0.0)
    captain_bonus = captain_base * (captain_multiplier - 1.0)
    if state.captain_id in per_player:
        per_player[state.captain_id] += captain_bonus
    for gameweek in gameweeks:
        points = _expected_points(state.captain_id, gameweek, projections)
        if points is not None:
            per_gameweek[gameweek] += points * (captain_multiplier - 1.0)

    starting_xi_points = sum(per_player[pid] for pid in state.starting_xi)

    bench_points = 0.0
    if include_bench:
        for player_id in state.bench_order:
            player_total = base_total(player_id)
            per_player[player_id] = player_total
            bench_points += player_total

    total = starting_xi_points + bench_points

    return SquadPoints(
        total=total,
        starting_xi_points=starting_xi_points,
        bench_points=bench_points,
        captain_bonus=captain_bonus,
        per_player=per_player,
        per_gameweek=per_gameweek,
        missing_player_ids=tuple(sorted(missing_ids)),
    )
