"""Season Replay's "how many points did I actually score" — the real-results mirror of
``features.squad_points.projected_points``. That module answers "what does this squad expect to
score" from a :class:`~engine.projections.PlayerHorizonProjection`; this module answers "what did
this squad *actually* score" from a real recorded ``{player_id: {minutes, total_points}}`` result
for one already-played gameweek, which is what ``POST /squad/advance`` needs when the user clicks
through a replayed season.

Reuses ``simulator.formation.apply_autosubs`` for the real FPL autosub rule rather than
reimplementing it — the one thing genuinely new here is the interaction between chip, autosubs,
captain fallback, and the hit cost already charged at confirm time, none of which
``features.squad_points`` needs (it only ever scores a *projection*, never a real result, so it has
no autosubs and no hit-cost concept).

Deliberately does **not** reuse ``simulator.run_simulation``'s own ``_score_squad``/
``_effective_captain`` — that module has two known bugs worth not inheriting: it computes but never
deducts ``hit_cost``, and it double-counts an autosubbed-in player under Bench Boost (autosubs run
unconditionally, then the *original* bench is summed on top). Both are fixed here: Bench Boost
skips autosubs entirely (all 15 score exactly as picked, matching real FPL), and ``hit_cost`` is
subtracted directly.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from features.squad_points import CHIP_BENCH_BOOST, CHIP_TRIPLE_CAPTAIN
from features.team_state import CHIPS, MyTeamState
from simulator.formation import apply_autosubs

__all__ = ["ActualGameweekResult", "score_actual_gameweek"]


@dataclass(frozen=True)
class ActualGameweekResult:
    """One gameweek's real, already-decided outcome for a committed squad — everything the Season
    Replay UI needs to show in its post-Advance reveal and season log."""

    gameweek: int
    chip_played: str | None
    effective_xi: tuple[int, ...]  # who actually counted -- 11 after autosubs, or all 15 under BB
    effective_captain_id: int
    hit_cost: int
    points: float  # already net of hit_cost


def score_actual_gameweek(
    gameweek: int,
    team_state: MyTeamState,
    chip: str | None,
    hit_cost: int,
    minutes_by_player: Mapping[int, int],
    points_by_player: Mapping[int, float],
) -> ActualGameweekResult:
    """Score ``team_state`` against one gameweek's real recorded ``minutes``/``total_points``.

    Under Bench Boost, no autosubs run and all 15 squad players count at face value (real FPL
    behaviour); otherwise :func:`simulator.formation.apply_autosubs` produces the effective 11.
    The captain's multiplier (x2, or x3 under Triple Captain) falls back to the vice-captain if the
    captain recorded 0 minutes — FPL's own single-level fallback, matching
    ``simulator.run_simulation``'s existing convention. A player absent from either mapping (no
    result recorded — a blank gameweek) contributes 0, never an error, since a real committed squad
    can legally contain a player with no fixture that week.
    """
    if chip is not None and chip not in CHIPS:
        raise ValueError(f"unknown chip: {chip!r}")
    if hit_cost < 0:
        raise ValueError("hit_cost must be non-negative")

    if chip == CHIP_BENCH_BOOST:
        effective_xi = team_state.starting_xi + team_state.bench_order
    else:
        effective_xi = apply_autosubs(
            team_state.starting_xi, team_state.bench_order, team_state.squad, minutes_by_player
        )

    captain_played = minutes_by_player.get(team_state.captain_id, 0) > 0
    effective_captain_id = team_state.captain_id if captain_played else team_state.vice_captain_id
    multiplier = 3.0 if chip == CHIP_TRIPLE_CAPTAIN else 2.0

    base_points = sum(points_by_player.get(pid, 0.0) for pid in effective_xi)
    captain_bonus = points_by_player.get(effective_captain_id, 0.0) * (multiplier - 1.0)
    points = base_points + captain_bonus - hit_cost

    return ActualGameweekResult(
        gameweek=gameweek,
        chip_played=chip,
        effective_xi=effective_xi,
        effective_captain_id=effective_captain_id,
        hit_cost=hit_cost,
        points=points,
    )
