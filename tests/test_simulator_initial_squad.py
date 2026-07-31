"""Tests for simulator/initial_squad.py -- greedy, budget/club-constrained squad selection."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.scoring import DEF, FWD, GK, MID
from simulator.initial_squad import DEFAULT_SQUAD_SHAPE, build_squad


def _make_pool(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def _cheap_pool(position: str, count: int, club_prefix: str, start_id: int) -> list[dict]:
    return [
        {
            "player_id": start_id + i,
            "position": position,
            "price": 40,
            "team": f"{club_prefix}{i}",
            "value_score": 10.0,
        }
        for i in range(count)
    ]


def test_build_squad_respects_shape():
    rows = (
        _cheap_pool(GK, 4, "gk_club", 1)
        + _cheap_pool(DEF, 8, "def_club", 100)
        + _cheap_pool(MID, 8, "mid_club", 200)
        + _cheap_pool(FWD, 6, "fwd_club", 300)
    )
    pool = _make_pool(rows)
    squad = build_squad(pool, budget=1000)
    counts = {
        position: sum(1 for p in squad if p.position == position)
        for position in DEFAULT_SQUAD_SHAPE
    }
    assert counts == dict(DEFAULT_SQUAD_SHAPE)
    assert len(squad) == 15


def test_build_squad_picks_highest_value_within_budget():
    rows = _cheap_pool(GK, 4, "gk_club", 1) + _cheap_pool(DEF, 8, "def_club", 100)
    rows += _cheap_pool(MID, 8, "mid_club", 200) + _cheap_pool(FWD, 6, "fwd_club", 300)
    # Make one GK clearly the best value -- must be selected.
    rows[0]["value_score"] = 999.0
    pool = _make_pool(rows)
    squad = build_squad(pool, budget=1000)
    gk_ids = {p.player_id for p in squad if p.position == GK}
    assert rows[0]["player_id"] in gk_ids


def test_build_squad_respects_max_per_club():
    # Five DEF from one club (all rated highest) plus enough from elsewhere to fill the rest --
    # must not take more than max_per_club from the one club, even though they're top-rated.
    rows = [
        {"player_id": i, "position": DEF, "price": 40, "team": "SameClub", "value_score": 999.0}
        for i in range(1, 6)
    ]
    rows += _cheap_pool(DEF, 5, "other_club", 50)
    rows += _cheap_pool(GK, 4, "gk_club", 100)
    rows += _cheap_pool(MID, 8, "mid_club", 200)
    rows += _cheap_pool(FWD, 6, "fwd_club", 300)
    pool = _make_pool(rows)
    squad = build_squad(pool, budget=1000, max_per_club=3)
    def_from_same_club = [p for p in squad if p.position == DEF and p.player_id in range(1, 6)]
    assert len(def_from_same_club) <= 3
    assert len([p for p in squad if p.position == DEF]) == 5


def test_build_squad_raises_when_position_cannot_be_filled():
    rows = _cheap_pool(GK, 1, "gk_club", 1)  # need 2 GK, only 1 available
    rows += _cheap_pool(DEF, 8, "def_club", 100)
    rows += _cheap_pool(MID, 8, "mid_club", 200)
    rows += _cheap_pool(FWD, 6, "fwd_club", 300)
    pool = _make_pool(rows)
    with pytest.raises(ValueError, match="could not fill"):
        build_squad(pool, budget=1000)


def test_build_squad_respects_total_budget():
    rows = _cheap_pool(GK, 4, "gk_club", 1)
    rows += _cheap_pool(DEF, 8, "def_club", 100)
    rows += _cheap_pool(MID, 8, "mid_club", 200)
    rows += _cheap_pool(FWD, 6, "fwd_club", 300)
    pool = _make_pool(rows)
    squad = build_squad(pool, budget=1000)
    assert sum(p.purchase_price for p in squad) <= 1000
