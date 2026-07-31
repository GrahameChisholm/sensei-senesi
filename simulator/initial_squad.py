"""Builds a legal, budget- and club-constrained squad from a real player-price pool (``planning/
SEASON_SIMULATOR.md`` component 4) — used both to seed the simulator's GW1 squad and to rebuild it
wholesale on a Wildcard/Free Hit week (Wildcard/Free Hit have no per-slot budget constraint, unlike
a normal transfer, so both reduce to "pick the best legal 15/temporary-11-plus-bench for this
budget" rather than the one-swap-at-a-time comparator ``features/transfers.py`` uses).

**Why a real specific manager's actual 2025/26 GW1 squad isn't used.** It is no longer retrievable
at all: FPL's entry endpoints only ever serve the *current* season's picks, confirmed live (checked
2026-07-31, after the 2025/26 season had already rolled over to 2026/27) —
``GET /entry/{id}/event/1/picks/`` 404s, and ``GET /entry/{id}/transfers/`` returns an empty list,
for any entry ID. Only each entry's lifetime *season total* (``GET /entry/{id}/history/``'s
``past`` list) survives the rollover — usable as an external benchmark, never as a source for
reconstructing the actual week-by-week squad. This module's ranking-by-hindsight-season-points
approach is the same "template squad" device ``backtest/run_season.py``'s
``build_stand_in_squad_starting_xi`` already uses and documents for the same reason (no real
historical "my team" exists to replay), not a new philosophical stance invented for this module.
"""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from engine.scoring import DEF, FWD, GK, MID
from features.team_state import SquadPlayer

__all__ = ["DEFAULT_SQUAD_SHAPE", "DEFAULT_BUDGET", "DEFAULT_MAX_PER_CLUB", "build_squad"]

DEFAULT_SQUAD_SHAPE: Mapping[str, int] = {GK: 2, DEF: 5, MID: 5, FWD: 3}
DEFAULT_BUDGET = 1000  # tenths of a million, i.e. GBP100.0m
DEFAULT_MAX_PER_CLUB = 3


def build_squad(
    player_pool: pd.DataFrame,
    budget: int = DEFAULT_BUDGET,
    shape: Mapping[str, int] = DEFAULT_SQUAD_SHAPE,
    max_per_club: int = DEFAULT_MAX_PER_CLUB,
    value_col: str = "value_score",
    price_col: str = "price",
    club_col: str = "team",
    player_id_col: str = "player_id",
    position_col: str = "position",
) -> tuple[SquadPlayer, ...]:
    """Greedy, budget- and club-constrained squad selection: for each position (scarcest-count
    first — so an expensive early pick in a big bucket can never starve a later, smaller one),
    rank candidates by ``value_col`` descending and take the best affordable one not already at
    ``max_per_club`` for its club, checking a remaining-slots affordability floor (the cheapest
    price seen anywhere in the whole pool, times every slot not yet filled) before committing to
    each pick.

    A v1 greedy heuristic — matching every other decision policy in this codebase
    (``features/transfers.py``'s greedy one-swap comparator, in particular), not a full knapsack
    solver. ``value_col`` is caller-supplied on purpose: seeding GW1 ranks by hindsight season
    total points (see this module's own docstring), while a Wildcard/Free Hit rebuild mid-season
    ranks by the engine's own real projected points instead — the same selection algorithm serves
    both, only the ranking signal changes.
    """
    pool = player_pool[[player_id_col, position_col, price_col, club_col, value_col]].dropna(
        subset=[position_col, price_col, club_col, value_col]
    )
    pool = pool[pool[price_col] > 0]
    if pool.empty:
        raise ValueError("player_pool has no eligible (priced) players")
    cheapest_price_overall = float(pool[price_col].min())
    total_slots = sum(shape.values())

    club_counts: dict[int, int] = {}
    selected: list[SquadPlayer] = []

    for position in sorted(shape, key=lambda pos: shape[pos]):
        needed = shape[position]
        candidates = pool[pool[position_col] == position].sort_values(value_col, ascending=False)
        filled = 0
        for row in candidates.itertuples():
            if filled >= needed:
                break
            price = int(getattr(row, price_col))
            club = getattr(row, club_col)
            player_id = int(getattr(row, player_id_col))
            if club_counts.get(club, 0) >= max_per_club:
                continue
            remaining_slots_after = total_slots - len(selected) - 1
            spent_so_far = sum(p.purchase_price for p in selected)
            if (budget - spent_so_far - price) < remaining_slots_after * cheapest_price_overall:
                continue
            selected.append(
                SquadPlayer(
                    player_id=player_id,
                    position=position,
                    purchase_price=price,
                    current_price=price,
                )
            )
            club_counts[club] = club_counts.get(club, 0) + 1
            filled += 1
        if filled < needed:
            raise ValueError(
                f"could not fill {needed} {position} slot(s) within budget/club constraints "
                f"(filled {filled})"
            )

    return tuple(selected)
