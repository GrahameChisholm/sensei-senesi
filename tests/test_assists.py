"""Tests for engine/models/assists.py — xA-based scoring rate (2.3)."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.models.assists import (
    DEFAULT_ASSIST_SHARE_OF_TEAM_XG,
    AssistProjection,
    expected_assist_rate,
    fit_assist_share_of_team_xg,
    prior_assist_rate_from_team_xg,
    project_assists,
    shrunk_player_xa_per_90,
)
from engine.scoring import ASSIST_POINTS


def test_expected_assist_rate_matches_formula():
    rate = expected_assist_rate(
        player_xa_per_90=0.3,
        opponent_xga_per_90=1.6,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
    )
    assert rate == pytest.approx(0.3 * (1.6 / 1.4))


def test_expected_assist_rate_rejects_non_positive_league_average():
    with pytest.raises(ValueError):
        expected_assist_rate(0.3, 1.4, 0.0, 90.0)


def test_expected_assist_rate_rejects_negative_inputs():
    with pytest.raises(ValueError):
        expected_assist_rate(-0.1, 1.4, 1.4, 90.0)


def test_expected_assist_rate_rejects_nan_inputs():
    with pytest.raises(ValueError):
        expected_assist_rate(float("nan"), 1.4, 1.4, 90.0)
    with pytest.raises(ValueError):
        expected_assist_rate(0.3, float("nan"), 1.4, 90.0)
    with pytest.raises(ValueError):
        expected_assist_rate(0.3, 1.4, 1.4, float("nan"))


def test_prior_assist_rate_from_team_xg_scales_with_share():
    prior = prior_assist_rate_from_team_xg(team_xg_per_90=1.5, assist_share=0.1)
    assert prior == pytest.approx(0.15)


def test_prior_assist_rate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        prior_assist_rate_from_team_xg(-1.0)
    with pytest.raises(ValueError):
        prior_assist_rate_from_team_xg(1.0, assist_share=1.5)


def test_shrunk_player_xa_thin_sample_moves_toward_team_prior():
    shrunk = shrunk_player_xa_per_90(
        player_xa_per_90=0.9,
        individual_weight=5.0,
        team_xg_per_90=1.0,
        shrinkage_k=50.0,
        assist_share=0.1,
    )
    # Prior = 0.1; thin individual sample should pull the rate well below the raw 0.9.
    assert shrunk < 0.9
    assert shrunk > 0.1


def test_shrunk_player_xa_thick_sample_stays_near_individual():
    shrunk = shrunk_player_xa_per_90(
        player_xa_per_90=0.9,
        individual_weight=5000.0,
        team_xg_per_90=1.0,
        shrinkage_k=50.0,
        assist_share=0.1,
    )
    assert shrunk == pytest.approx(0.9, abs=1e-2)


def test_assist_projection_expected_points_uses_flat_rate():
    projection = AssistProjection(assist_rate=0.25)
    assert projection.expected_points == pytest.approx(0.25 * ASSIST_POINTS)


def test_assist_projection_rejects_negative_rate():
    with pytest.raises(ValueError):
        AssistProjection(assist_rate=-0.1)


def test_project_assists_without_shrinkage_uses_raw_rate():
    projection = project_assists(
        player_xa_per_90=0.3,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
    )
    assert projection.assist_rate == pytest.approx(0.3)


def test_project_assists_with_shrinkage_pulls_toward_team_prior():
    without_shrinkage = project_assists(
        player_xa_per_90=0.9,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
    )
    with_shrinkage = project_assists(
        player_xa_per_90=0.9,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
        individual_weight=5.0,
        team_xg_per_90=1.0,
        shrinkage_k=50.0,
    )
    assert with_shrinkage.assist_rate < without_shrinkage.assist_rate


def test_project_assists_missing_shrinkage_args_falls_back_to_raw_rate():
    # shrinkage_k omitted (defaults to 0) -> no shrinkage applied even though other args given.
    projection = project_assists(
        player_xa_per_90=0.9,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
        individual_weight=5.0,
        team_xg_per_90=1.0,
    )
    assert projection.assist_rate == pytest.approx(0.9)


def test_fit_assist_share_of_team_xg_inverts_the_prior_formula():
    # 3 rows, each implying an opportunity of team_xg_per_90 * minutes / 90 = 2.0 * 90/90 = 2.0,
    # total opportunity 6.0; total assists 3 -> share = 3/6 = 0.5.
    assists = pd.Series([1, 1, 1])
    team_xg_per_90 = pd.Series([2.0, 2.0, 2.0])
    minutes = pd.Series([90.0, 90.0, 90.0])

    share = fit_assist_share_of_team_xg(assists, team_xg_per_90, minutes, min_rows=1)

    assert share == pytest.approx(0.5)


def test_fit_assist_share_of_team_xg_falls_back_for_thin_sample():
    assists = pd.Series([1, 1])
    team_xg_per_90 = pd.Series([2.0, 2.0])
    minutes = pd.Series([90.0, 90.0])

    share = fit_assist_share_of_team_xg(assists, team_xg_per_90, minutes, min_rows=100)

    assert share == DEFAULT_ASSIST_SHARE_OF_TEAM_XG


def test_fit_assist_share_of_team_xg_falls_back_for_zero_opportunity():
    assists = pd.Series([0, 0])
    team_xg_per_90 = pd.Series([0.0, 0.0])
    minutes = pd.Series([90.0, 90.0])

    share = fit_assist_share_of_team_xg(assists, team_xg_per_90, minutes, min_rows=1)

    assert share == DEFAULT_ASSIST_SHARE_OF_TEAM_XG


def test_project_assists_uses_a_custom_assist_share_when_shrinking():
    # A lower assist_share_of_team_xg pulls the shrunk rate toward a lower prior -- this is the
    # per-position lever ENGINE_IMPROVEMENTS_4.md wires in to fix a real FWD over-prediction bias
    # a single flat share caused once shrinkage was strong enough to fix aggregate calibration.
    flat_share = project_assists(
        player_xa_per_90=0.9,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
        individual_weight=5.0,
        team_xg_per_90=1.0,
        shrinkage_k=50.0,
    )
    low_share = project_assists(
        player_xa_per_90=0.9,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
        individual_weight=5.0,
        team_xg_per_90=1.0,
        shrinkage_k=50.0,
        assist_share_of_team_xg=0.02,
    )
    assert low_share.assist_rate < flat_share.assist_rate
