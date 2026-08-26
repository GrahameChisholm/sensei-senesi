"""Tests for features/team_state.py — the shared, permanently-live sandbox "My Team" state."""

from __future__ import annotations

import pytest

from features.team_state import MyTeamState, SquadPlayer


def _squad_player(player_id: int, position: str = "MID", price: int = 50):
    return SquadPlayer(player_id=player_id, position=position, price=price)


def _full_squad(price_overrides: dict[int, int] | None = None):
    """15 distinct players: 2 GK, 5 DEF, 5 MID, 3 FWD — a legal FPL squad shape, prices flat at
    5.0m unless overridden by player_id via ``price_overrides``."""
    price_overrides = price_overrides or {}
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = []
    for i, position in enumerate(positions, start=1):
        players.append(_squad_player(i, position, price_overrides.get(i, 50)))
    return tuple(players)


def _default_state(**kwargs):
    squad = kwargs.pop("squad", _full_squad())
    defaults = dict(
        squad=squad,
        starting_xi=tuple(range(1, 12)),
        bench_order=tuple(range(12, 16)),
        captain_id=1,
        vice_captain_id=2,
    )
    defaults.update(kwargs)
    return MyTeamState(**defaults)


# --- SquadPlayer ---------------------------------------------------------------------------------


def test_squad_player_rejects_unknown_position():
    with pytest.raises(ValueError):
        _squad_player(1, position="XYZ")


def test_squad_player_rejects_non_positive_price():
    with pytest.raises(ValueError):
        SquadPlayer(player_id=1, position="MID", price=0)


# --- MyTeamState -----------------------------------------------------------------------------


def test_my_team_state_valid_construction_round_trips():
    state = _default_state()
    assert len(state.player_ids) == 15
    assert state.player(1).position == "GK"


def test_my_team_state_rejects_wrong_squad_size():
    with pytest.raises(ValueError):
        _default_state(squad=_full_squad()[:14])


def test_my_team_state_rejects_duplicate_player_ids():
    squad = list(_full_squad())
    squad[1] = _squad_player(1, "DEF")  # duplicate of player_id 1
    with pytest.raises(ValueError):
        _default_state(squad=tuple(squad))


def test_my_team_state_rejects_wrong_starting_xi_size():
    with pytest.raises(ValueError):
        _default_state(starting_xi=tuple(range(1, 11)), bench_order=tuple(range(11, 16)))


def test_my_team_state_rejects_overlap_between_xi_and_bench():
    with pytest.raises(ValueError):
        _default_state(starting_xi=tuple(range(1, 12)), bench_order=tuple(range(11, 15)))


def test_my_team_state_rejects_xi_bench_not_partitioning_squad():
    with pytest.raises(ValueError):
        _default_state(starting_xi=tuple(range(1, 12)), bench_order=(13, 14, 15, 16))


def test_my_team_state_rejects_captain_not_in_starting_xi():
    with pytest.raises(ValueError):
        _default_state(captain_id=12)


def test_my_team_state_rejects_same_captain_and_vice():
    with pytest.raises(ValueError):
        _default_state(vice_captain_id=1)
