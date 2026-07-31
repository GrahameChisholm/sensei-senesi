"""Tests for engine.data.team_state_builder — assembling a real MyTeamState from FPL manager
(entry) API responses (A3). All synthetic; no network."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data.team_state_builder import (
    DEFAULT_FIRST_HALF_LAST_GAMEWEEK,
    build_my_team_state,
    compute_chips_remaining,
    compute_free_transfers,
    compute_purchase_prices,
)
from features.team_state import CHIPS


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
    picks = [_pick(i, i, is_captain=(i == 1), is_vice_captain=(i == 2)) for i in range(1, 12)]
    picks += [_pick(i, i) for i in range(12, 16)]
    return {"active_chip": None, "picks": picks}


def _elements() -> pd.DataFrame:
    # Mostly MID (element_type 3) for simplicity; now_cost varies so purchase-price fallback is
    # distinguishable from the transfer-derived price.
    return pd.DataFrame([{"id": i, "element_type": 3, "now_cost": 50 + i} for i in range(1, 16)])


def test_compute_purchase_prices_uses_most_recent_transfer_in():
    picks = [{"element": 1}, {"element": 2}]
    transfers = [
        {"element_in": 1, "element_in_cost": 55, "event": 3},
        {"element_in": 1, "element_in_cost": 60, "event": 7},  # later -> this one wins
    ]
    now_cost = {1: 65, 2: 52}

    prices = compute_purchase_prices(picks, transfers, now_cost)

    assert prices[1] == 60
    assert prices[2] == 52  # never transferred in -- falls back to current price


def test_compute_purchase_prices_raises_on_unknown_player():
    with pytest.raises(KeyError):
        compute_purchase_prices([{"element": 99}], [], {1: 50})


def test_compute_free_transfers_accrues_and_consumes():
    # Baseline 1 -> gw1 (0 transfers): consume none, accrue to 2 -> gw2 (2 transfers): consume
    # both down to 0, accrue to 1 -> gw3 (0 transfers): consume none, accrue to 2.
    history = [{"event_transfers": 0}, {"event_transfers": 2}, {"event_transfers": 0}]
    assert compute_free_transfers(history) == 2


def test_compute_free_transfers_caps_at_max():
    history = [{"event_transfers": 0}] * 10
    assert compute_free_transfers(history, max_free_transfers=5) == 5


def test_compute_free_transfers_defaults_to_one_with_no_history():
    assert compute_free_transfers([]) == 1


def test_compute_chips_remaining_excludes_chips_played_this_half():
    chips_played = [{"name": "wildcard", "event": 5}]
    remaining = compute_chips_remaining(chips_played, current_gameweek=10)
    assert remaining == frozenset(CHIPS) - {"wildcard"}


def test_compute_chips_remaining_resets_after_the_half_boundary():
    chips_played = [{"name": "wildcard", "event": 5}]  # played in the first half
    remaining = compute_chips_remaining(
        chips_played, current_gameweek=20, first_half_last_gameweek=DEFAULT_FIRST_HALF_LAST_GAMEWEEK
    )
    assert remaining == frozenset(CHIPS)  # reset -- wildcard usable again in the second half


def test_build_my_team_state_assembles_a_valid_squad():
    state = build_my_team_state(
        picks=_picks_payload(),
        entry={"last_deadline_bank": 5},
        transfers=[],
        history={"current": [], "chips": []},
        elements=_elements(),
        current_gameweek=1,
    )

    assert len(state.squad) == 15
    assert state.starting_xi == tuple(range(1, 12))
    assert set(state.bench_order) == {12, 13, 14, 15}
    assert state.captain_id == 1
    assert state.vice_captain_id == 2
    assert state.bank == 5
    assert state.free_transfers == 1
    assert state.chips_remaining == frozenset(CHIPS)


def test_build_my_team_state_uses_explicit_free_transfers_override():
    state = build_my_team_state(
        picks=_picks_payload(),
        entry={"last_deadline_bank": 0},
        transfers=[],
        history={"current": [{"event_transfers": 0}] * 3, "chips": []},
        elements=_elements(),
        current_gameweek=1,
        free_transfers=3,
    )

    assert state.free_transfers == 3


def test_build_my_team_state_reflects_a_real_transfer():
    picks = _picks_payload()
    transfers = [{"element_in": 1, "element_in_cost": 999, "event": 2}]

    state = build_my_team_state(
        picks=picks,
        entry={"last_deadline_bank": 0},
        transfers=transfers,
        history={"current": [], "chips": []},
        elements=_elements(),
        current_gameweek=1,
    )

    assert state.player(1).purchase_price == 999
