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
    # defensive_contribution is a raw action count (tackles/interceptions/blocks/clearances), not
    # an FPL points value -- 12 clears DEF's own threshold of 10 (engine.scoring
    # .DEFENSIVE_CONTRIBUTION_THRESHOLD), so every row here should earn the flat bonus.
    rows = []
    for _ in range(8):
        rows.append(
            _row(
                "DEF",
                value=45,
                minutes=90,
                goals_scored=0,
                assists=0,
                defensive_contribution=12,
                clean_sheets=1,
            )
        )
    return rows


def _prior_merged_gw() -> pd.DataFrame:
    return pd.DataFrame(_mid_rows() + _gk_rows() + _def_rows())


def _club_ranked_rows() -> list[dict]:
    """Three players per club, all landing in the same £5.5m-£5.9m MID price bucket (55-59 in
    tenths of a million; ``_price_bucket`` floors all three to 55), so every within-club rank tier
    (1st, 2nd, 3rd-or-lower most expensive) is populated within one bucket. A clear real signal
    separates them: the highest-priced player at each club plays full matches and returns
    regularly, the cheapest of the three barely gets on the pitch. This is what lets
    fit_cold_start_priors learn the within-club rank differentiator from real data rather than a
    hand-picked multiplier.
    """
    rows = []
    for team in ("Hull", "Coventry"):
        for _ in range(15):
            rows.append(
                _row(
                    "MID",
                    value=59,
                    minutes=88,
                    goals_scored=1,
                    assists=1,
                    team=team,
                    element=f"{team}-first",
                )
            )
        for _ in range(15):
            rows.append(
                _row(
                    "MID",
                    value=57,
                    minutes=60,
                    goals_scored=0,
                    assists=0,
                    team=team,
                    element=f"{team}-second",
                )
            )
        for _ in range(15):
            rows.append(
                _row(
                    "MID",
                    value=55,
                    minutes=5,
                    goals_scored=0,
                    assists=0,
                    team=team,
                    element=f"{team}-third",
                )
            )
    return rows


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


class TestWithinClubRankDifferentiation:
    """T-C: 190 real cold-start players collapsed into only 18 distinct expected-points values
    because the prior varied only by (position, price bucket). These tests fit from a frame with
    real club identity and assert two players sharing a position and price bucket, but different
    within-club price rank, no longer collapse onto the same projection.
    """

    def test_priors_fit_a_rank_tier_cut_when_club_identity_is_present(self):
        priors = fit_cold_start_priors(pd.DataFrame(_club_ranked_rows()))
        assert ("MID", 55, "1") in priors.by_position_bucket_and_rank
        assert ("MID", 55, "2") in priors.by_position_bucket_and_rank
        assert ("MID", 55, "3+") in priors.by_position_bucket_and_rank

    def test_no_rank_tier_cut_when_club_identity_is_absent(self):
        # The existing club-less fixtures (no "team"/"element" columns) must keep behaving exactly
        # as before this differentiator existed.
        priors = fit_cold_start_priors(_prior_merged_gw())
        assert priors.by_position_bucket_and_rank == {}

    def test_same_bucket_different_rank_gives_different_projections(self):
        priors = fit_cold_start_priors(pd.DataFrame(_club_ranked_rows()))
        # 59 and 55 both floor to price bucket 55 (PRICE_BUCKET_WIDTH == 5) -- only the within-club
        # rank differs between these two calls.
        assert 59 // PRICE_BUCKET_WIDTH == 55 // PRICE_BUCKET_WIDTH
        first_choice = baseline_projection(
            101, "MID", 59, gameweek=1, priors=priors, within_club_position_rank=1
        )
        third_choice = baseline_projection(
            102, "MID", 55, gameweek=1, priors=priors, within_club_position_rank=3
        )
        assert first_choice.expected_points != third_choice.expected_points
        assert first_choice.expected_points > third_choice.expected_points

    def test_rank_ignored_when_combination_was_never_observed_falls_back(self):
        priors = fit_cold_start_priors(pd.DataFrame(_club_ranked_rows()))
        # No prior-season rows for a MID priced at £20.0m at any rank -- must fall back to the
        # position-only average rather than raising, exactly like the bucket fallback already does.
        projection = baseline_projection(
            103, "MID", 200, gameweek=1, priors=priors, within_club_position_rank=1
        )
        position_only = priors.by_position["MID"]
        assert projection.expected_points == pytest.approx(position_only.breakdown.total)

    def test_omitting_rank_uses_the_flatter_bucket_prior(self):
        priors = fit_cold_start_priors(pd.DataFrame(_club_ranked_rows()))
        no_rank = baseline_projection(104, "MID", 57, gameweek=1, priors=priors)
        bucket_only = priors.by_position_and_bucket[("MID", 55)]
        assert no_rank.expected_points == pytest.approx(bucket_only.breakdown.total)
