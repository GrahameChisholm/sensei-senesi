"""Tests for engine.data.team_state_builder — assembling a real MyTeamState from FPL manager
(entry) API responses. All synthetic; no network.

Positions (element_type: 1=GK, 2=DEF, 3=MID, 4=FWD): id 1 + 12 are GK, ids 2-5 + 13 are DEF, ids
6-9 + 14 are MID, ids 10-11 + 15 are FWD -- a legal 2/5/5/3 squad with a legal starting XI
(1 GK, 4 DEF, 4 MID, 2 FWD, slots 1-11) and bench (slots 12-15).
"""

from __future__ import annotations

import pandas as pd

from engine.data.team_state_builder import build_my_team_state

_ELEMENT_TYPE_BY_ID = {
    1: 1,
    12: 1,  # GK
    2: 2,
    3: 2,
    4: 2,
    5: 2,
    13: 2,  # DEF
    6: 3,
    7: 3,
    8: 3,
    9: 3,
    14: 3,  # MID
    10: 4,
    11: 4,
    15: 4,  # FWD
}


def _pick(
    element: int, position: int, is_captain: bool = False, is_vice_captain: bool = False
) -> dict:
    return {
        "element": element,
        "position": position,
        "multiplier": 2 if is_captain else 1,
        "is_captain": is_captain,
        "is_vice_captain": is_vice_captain,
    }


def _picks_payload() -> dict:
    picks = [_pick(i, i, is_captain=(i == 6), is_vice_captain=(i == 7)) for i in range(1, 12)]
    picks += [_pick(i, i) for i in range(12, 16)]
    return {"active_chip": None, "picks": picks}


def _elements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"id": i, "element_type": _ELEMENT_TYPE_BY_ID[i], "now_cost": 50 + i}
            for i in range(1, 16)
        ]
    )


def _team_id_by_player() -> dict[int, int]:
    return {i: i for i in range(1, 16)}


def test_build_my_team_state_assembles_a_valid_squad():
    state = build_my_team_state(
        picks=_picks_payload(), elements=_elements(), team_id_by_player=_team_id_by_player()
    )

    assert len(state.squad) == 15
    assert state.starting_xi == tuple(range(1, 12))
    assert set(state.bench_order) == {12, 13, 14, 15}
    assert state.captain_id == 6
    assert state.vice_captain_id == 7


def test_build_my_team_state_uses_current_price_for_every_player():
    state = build_my_team_state(
        picks=_picks_payload(), elements=_elements(), team_id_by_player=_team_id_by_player()
    )
    assert state.player(1).price == 51
    assert state.player(15).price == 65


def test_build_my_team_state_allows_a_squad_over_the_classic_budget_ceiling():
    # now_cost sums well past £100m (1000) here -- a real squad's current value can legitimately
    # exceed the classic ceiling through price-rise profit, so this must not raise.
    elements = pd.DataFrame(
        [{"id": i, "element_type": _ELEMENT_TYPE_BY_ID[i], "now_cost": 200} for i in range(1, 16)]
    )
    state = build_my_team_state(
        picks=_picks_payload(), elements=elements, team_id_by_player=_team_id_by_player()
    )
    assert sum(p.price for p in state.squad) == 3000
