"""Tests for features/team_state.py — the shared "My Team" state (BUILD_PLAN Phase 4)."""

from __future__ import annotations

import pytest

from features.team_state import CHIPS, MyTeamState, SquadPlayer, compute_sell_price


def _squad_player(player_id: int, position: str = "MID", purchase: int = 50, current: int = 50):
    return SquadPlayer(
        player_id=player_id, position=position, purchase_price=purchase, current_price=current
    )


def _full_squad(price_overrides: dict[int, tuple[int, int]] | None = None):
    """15 distinct players: 2 GK, 5 DEF, 5 MID, 3 FWD — a legal FPL squad shape, prices flat at
    5.0m unless overridden by player_id via ``price_overrides``."""
    price_overrides = price_overrides or {}
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    players = []
    for i, position in enumerate(positions, start=1):
        purchase, current = price_overrides.get(i, (50, 50))
        players.append(_squad_player(i, position, purchase, current))
    return tuple(players)


def _default_state(**kwargs):
    squad = kwargs.pop("squad", _full_squad())
    defaults = dict(
        squad=squad,
        starting_xi=tuple(range(1, 12)),
        bench_order=tuple(range(12, 16)),
        captain_id=1,
        vice_captain_id=2,
        bank=0,
        free_transfers=1,
        chips_remaining=frozenset(CHIPS),
    )
    defaults.update(kwargs)
    return MyTeamState(**defaults)


# --- compute_sell_price -------------------------------------------------------------------------


def test_compute_sell_price_risen_player_banks_half_profit_rounded_down():
    # profit = 15 tenths -> half = 7.5 -> floor to 7
    assert compute_sell_price(purchase_price=50, current_price=65) == 57


def test_compute_sell_price_even_profit_halves_exactly():
    assert compute_sell_price(purchase_price=50, current_price=60) == 55


def test_compute_sell_price_flat_price_sells_at_current():
    assert compute_sell_price(purchase_price=50, current_price=50) == 50


def test_compute_sell_price_fallen_price_sells_at_current_no_extra_penalty():
    assert compute_sell_price(purchase_price=50, current_price=45) == 45


# --- SquadPlayer ---------------------------------------------------------------------------------


def test_squad_player_sell_price_matches_compute_sell_price():
    player = _squad_player(1, purchase=50, current=65)
    assert player.sell_price == compute_sell_price(50, 65)


def test_squad_player_rejects_unknown_position():
    with pytest.raises(ValueError):
        _squad_player(1, position="XYZ")


def test_squad_player_rejects_non_positive_prices():
    with pytest.raises(ValueError):
        SquadPlayer(player_id=1, position="MID", purchase_price=0, current_price=50)


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


def test_my_team_state_rejects_free_transfers_out_of_range():
    with pytest.raises(ValueError):
        _default_state(free_transfers=6)
    with pytest.raises(ValueError):
        _default_state(free_transfers=-1)


def test_my_team_state_rejects_unknown_chip():
    with pytest.raises(ValueError):
        _default_state(chips_remaining=frozenset({"golden_boot"}))


def test_my_team_state_total_sell_value_uses_sell_price_not_current_price():
    overrides = {1: (50, 65)}  # player 1 risen from 5.0m to 6.5m -> sells at 5.7m
    squad = _full_squad(overrides)
    state = _default_state(squad=squad)
    expected = compute_sell_price(50, 65) + sum(p.current_price for p in squad[1:])
    assert state.total_sell_value == expected
