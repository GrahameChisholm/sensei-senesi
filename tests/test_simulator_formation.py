"""Tests for simulator/formation.py -- starting XI/formation selection and FPL's autosub rule."""

from __future__ import annotations

import pytest

from engine.scoring import DEF, FWD, GK, MID
from features.team_state import SquadPlayer
from simulator.formation import VALID_FORMATIONS, apply_autosubs, select_starting_xi


def _squad_player(player_id: int, position: str) -> SquadPlayer:
    return SquadPlayer(player_id=player_id, position=position, purchase_price=50, current_price=50)


def _standard_squad() -> tuple[SquadPlayer, ...]:
    # 2 GK, 5 DEF, 5 MID, 3 FWD -- a real FPL squad shape.
    return (
        _squad_player(1, GK),
        _squad_player(2, GK),
        _squad_player(3, DEF),
        _squad_player(4, DEF),
        _squad_player(5, DEF),
        _squad_player(6, DEF),
        _squad_player(7, DEF),
        _squad_player(8, MID),
        _squad_player(9, MID),
        _squad_player(10, MID),
        _squad_player(11, MID),
        _squad_player(12, MID),
        _squad_player(13, FWD),
        _squad_player(14, FWD),
        _squad_player(15, FWD),
    )


def test_valid_formations_all_sum_to_ten_outfield():
    for d, m, f in VALID_FORMATIONS:
        assert d + m + f == 10
        assert 3 <= d <= 5
        assert 2 <= m <= 5
        assert 1 <= f <= 3


def test_select_starting_xi_picks_higher_ev_goalkeeper():
    squad = _standard_squad()
    ev = {pid: 5.0 for pid in range(1, 16)}
    ev[2] = 10.0  # GK 2 clearly better
    starting_xi, bench_order = select_starting_xi(squad, ev)
    assert 2 in starting_xi
    assert 1 not in starting_xi
    assert 1 in bench_order
    assert bench_order[-1] == 1  # reserve GK always last


def test_select_starting_xi_is_exactly_eleven_and_partitions_squad():
    squad = _standard_squad()
    ev = {pid: float(pid) for pid in range(1, 16)}
    starting_xi, bench_order = select_starting_xi(squad, ev)
    assert len(starting_xi) == 11
    assert len(bench_order) == 4
    assert set(starting_xi) | set(bench_order) == {p.player_id for p in squad}
    assert not set(starting_xi) & set(bench_order)


def test_select_starting_xi_prefers_higher_ev_outfield_combo():
    squad = _standard_squad()
    # Make one DEF and one MID much higher-EV than the rest -- formation choice should favor
    # including both high-value players over an arbitrary split.
    ev = {pid: 2.0 for pid in range(1, 16)}
    ev[1] = 5.0
    ev[3] = 20.0  # top DEF
    ev[8] = 20.0  # top MID
    starting_xi, _ = select_starting_xi(squad, ev)
    assert 3 in starting_xi
    assert 8 in starting_xi


def test_select_starting_xi_raises_with_no_goalkeeper():
    squad = tuple(p for p in _standard_squad() if p.position != GK)
    with pytest.raises(ValueError, match="goalkeeper"):
        select_starting_xi(squad, {pid: 1.0 for pid in range(3, 16)})


def test_apply_autosubs_brings_on_bench_player_for_zero_minute_starter():
    squad = _standard_squad()
    ev = {pid: float(pid) for pid in range(1, 16)}
    starting_xi, bench_order = select_starting_xi(squad, ev)
    # Force the lowest-EV starter (first FWD in formation, or whichever landed in XI) to blank.
    zero_minute_player = starting_xi[-1]
    minutes = dict.fromkeys(range(1, 16), 90)
    minutes[zero_minute_player] = 0
    effective_xi = apply_autosubs(starting_xi, bench_order, squad, minutes)
    assert zero_minute_player not in effective_xi
    assert len(effective_xi) == 11


def test_apply_autosubs_does_not_break_def_floor():
    squad = _standard_squad()
    ev = {pid: float(pid) for pid in range(1, 16)}
    starting_xi, bench_order = select_starting_xi(squad, ev)
    # Blank every DEF starter at once -- autosubs must never drop below 3 DEF in the final XI.
    minutes = dict.fromkeys(range(1, 16), 90)
    for pid in starting_xi:
        if pid in {3, 4, 5, 6, 7}:
            minutes[pid] = 0
    effective_xi = apply_autosubs(starting_xi, bench_order, squad, minutes)
    def_count = sum(1 for pid in effective_xi if pid in {3, 4, 5, 6, 7})
    assert def_count >= 3 or def_count == sum(1 for pid in starting_xi if pid in {3, 4, 5, 6, 7})


def test_apply_autosubs_reserve_gk_only_replaces_starting_gk():
    squad = _standard_squad()
    ev = {pid: float(pid) for pid in range(1, 16)}
    ev[2] = 10.0  # GK 2 starts, GK 1 benched
    starting_xi, bench_order = select_starting_xi(squad, ev)
    minutes = dict.fromkeys(range(1, 16), 90)
    # An outfield starter blanks, but the reserve GK must not come on for them.
    outfield_starter = next(pid for pid in starting_xi if pid != 2)
    minutes[outfield_starter] = 0
    effective_xi = apply_autosubs(starting_xi, bench_order, squad, minutes)
    assert 1 not in effective_xi  # reserve GK stays put; an outfield reserve came on instead
