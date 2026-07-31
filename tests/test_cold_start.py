"""Tests for engine.data.cold_start -- position x price baseline projections for players with no
reconstructable engine history (D5).
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data.cold_start import PRICE_BUCKET_WIDTH, baseline_projection, fit_cold_start_priors
from engine.projections import PlayerGameweekProjection

_BASE_ROW = {
    "clean_sheets": 0,
    "goals_conceded": 1,
    "defensive_contribution": 0,
    "saves": 0,
    "bonus": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "penalties_missed": 0,
    "own_goals": 0,
}


def _row(
    position: str, value: int, minutes: int, goals_scored: int, assists: int, **overrides
) -> dict:
    row = dict(_BASE_ROW)
    row.update(
        {
            "position": position,
            "value": value,
            "minutes": minutes,
            "goals_scored": goals_scored,
            "assists": assists,
        }
    )
    row.update(overrides)
    return row


def _mid_rows() -> list[dict]:
    rows = []
    # Cheap bucket (£4.5m): plays but rarely scores.
    for _ in range(10):
        rows.append(_row("MID", value=45, minutes=70, goals_scored=0, assists=0))
    # Mid bucket (£6.5m): starts, occasional goal.
    for _ in range(10):
        rows.append(_row("MID", value=65, minutes=85, goals_scored=0, assists=1))
    for _ in range(2):
        rows.append(_row("MID", value=65, minutes=85, goals_scored=1, assists=0))
    # Premium bucket (£9.5m): nailed starter, regular returns.
    for _ in range(10):
        rows.append(_row("MID", value=95, minutes=90, goals_scored=1, assists=1))
    return rows


def _gk_rows() -> list[dict]:
    rows = []
    for _ in range(8):
        rows.append(
            _row("GK", value=50, minutes=90, goals_scored=0, assists=0, saves=3, clean_sheets=1)
        )
    return rows


def _def_rows() -> list[dict]:
    rows = []
    for _ in range(8):
        rows.append(
            _row(
                "DEF",
                value=45,
                minutes=90,
                goals_scored=0,
                assists=0,
                defensive_contribution=2,
                clean_sheets=1,
            )
        )
    return rows


def _prior_merged_gw() -> pd.DataFrame:
    return pd.DataFrame(_mid_rows() + _gk_rows() + _def_rows())


class TestFitColdStartPriors:
    def test_missing_required_column_raises(self):
        broken = _prior_merged_gw().drop(columns=["saves"])
        with pytest.raises(ValueError, match="missing expected column"):
            fit_cold_start_priors(broken)

    def test_buckets_and_position_fallback_both_populated(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        assert ("MID", 45) in priors.by_position_and_bucket
        assert ("MID", 65) in priors.by_position_and_bucket
        assert ("MID", 95) in priors.by_position_and_bucket
        assert "MID" in priors.by_position
        assert "GK" in priors.by_position
        assert "DEF" in priors.by_position


class TestBaselineProjection:
    def test_output_is_a_real_player_gameweek_projection(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        projection = baseline_projection(9001, "MID", 65, gameweek=1, priors=priors)
        assert isinstance(projection, PlayerGameweekProjection)
        assert projection.player_id == 9001
        assert projection.position == "MID"
        assert projection.gameweek == 1
        assert projection.simulation is None

    def test_breakdown_total_is_internally_consistent(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        projection = baseline_projection(9001, "MID", 65, gameweek=1, priors=priors)
        b = projection.breakdown
        assert projection.expected_points == pytest.approx(b.total)

    def test_gk_gets_no_defensive_contribution(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        projection = baseline_projection(9002, "GK", 50, gameweek=1, priors=priors)
        assert projection.breakdown.defensive_contribution == 0.0

    def test_gk_gets_real_saves_points(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        projection = baseline_projection(9002, "GK", 50, gameweek=1, priors=priors)
        assert projection.breakdown.saves > 0.0

    def test_def_gets_no_saves_but_real_defensive_contribution(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        projection = baseline_projection(9003, "DEF", 45, gameweek=1, priors=priors)
        assert projection.breakdown.saves == 0.0
        assert projection.breakdown.defensive_contribution > 0.0

    def test_ep_monotonic_in_price_within_a_position(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        cheap = baseline_projection(1, "MID", 45, gameweek=1, priors=priors)
        mid = baseline_projection(2, "MID", 65, gameweek=1, priors=priors)
        premium = baseline_projection(3, "MID", 95, gameweek=1, priors=priors)
        assert cheap.expected_points < mid.expected_points < premium.expected_points

    def test_missing_bucket_falls_back_to_position_average(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        # £20.0m MID never existed in the fitted data -- must fall back, not raise.
        projection = baseline_projection(4, "MID", 200, gameweek=1, priors=priors)
        position_only = priors.by_position["MID"]
        assert projection.expected_points == pytest.approx(position_only.breakdown.total)

    def test_unknown_position_raises(self):
        priors = fit_cold_start_priors(_prior_merged_gw())
        with pytest.raises(ValueError, match="no cold-start prior available"):
            baseline_projection(5, "FWD", 60, gameweek=1, priors=priors)

    def test_price_bucket_width_is_half_a_million(self):
        assert PRICE_BUCKET_WIDTH == 5
