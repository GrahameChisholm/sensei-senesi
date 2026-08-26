"""Build a real squad from FPL manager (entry) API responses.

Feeds directly off :class:`~engine.data.fpl_client.FPLClient`'s ``get_entry``/``get_entry_picks``
methods plus the bootstrap ``elements`` table already fetched for every other live purpose. In the
sandbox model there is no purchase price, free-transfer count, or chip usage to reconstruct — a
player's price is always just their current price, so this only ever needs the manager's current
picks and today's prices.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd

from engine.scoring import ELEMENT_TYPE_TO_POSITION
from features.squad_rules import build_team_state
from features.team_state import MyTeamState, SquadPlayer

__all__ = ["build_my_team_state"]


def build_my_team_state(
    picks: Mapping[str, Any],
    elements: pd.DataFrame,
    team_id_by_player: Mapping[int, int],
) -> MyTeamState:
    """Assemble a real :class:`~features.team_state.MyTeamState` from
    :class:`~engine.data.fpl_client.FPLClient`'s ``get_entry_picks`` response and the bootstrap
    ``elements`` table. Keeps the manager's actual starting XI/bench/captain/vice exactly as FPL
    reports them, and skips the classic £100m budget check
    (:func:`~features.squad_rules.build_team_state`'s ``check_budget=False``) since a real
    squad's current value can legitimately exceed it through price-rise profit — the caller is
    responsible for computing a personal budget ceiling from the result if it needs one.
    """
    now_cost_by_id = dict(zip(elements["id"], elements["now_cost"], strict=True))
    element_type_by_id = dict(zip(elements["id"], elements["element_type"], strict=True))

    squad = tuple(
        SquadPlayer(
            player_id=int(pick["element"]),
            position=ELEMENT_TYPE_TO_POSITION[int(element_type_by_id[int(pick["element"])])],
            price=int(now_cost_by_id[int(pick["element"])]),
        )
        for pick in picks["picks"]
    )
    starting_xi = tuple(
        int(pick["element"]) for pick in picks["picks"] if int(pick["position"]) <= 11
    )
    bench_order = tuple(
        int(pick["element"])
        for pick in sorted(
            (p for p in picks["picks"] if int(p["position"]) > 11),
            key=lambda p: int(p["position"]),
        )
    )
    captain_id = next(int(p["element"]) for p in picks["picks"] if p["is_captain"])
    vice_captain_id = next(int(p["element"]) for p in picks["picks"] if p["is_vice_captain"])

    return build_team_state(
        squad=squad,
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        team_id_by_player=team_id_by_player,
        check_budget=False,
    )
