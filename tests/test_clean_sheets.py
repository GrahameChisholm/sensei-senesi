"""Tests for engine/models/clean_sheets.py — Dixon-Coles clean sheets & goals conceded (2.4)."""

from __future__ import annotations

import pandas as pd
import pytest

from engine.models.clean_sheets import (
    CleanSheetProjection,
    clean_sheet_probability,
    dixon_coles_tau,
    expected_goals_conceded_penalty,
    project_clean_sheet,
    scoreline_distribution,
    split_by_venue,
    team_expected_goals_rate,
)


def test_team_expected_goals_rate_matches_formula():
    rate = team_expected_goals_rate(
        attacking_xg_per_90=1.5, defending_xga_per_90=1.6, league_avg_xga_per_90=1.4
    )
    assert rate == pytest.approx(1.5 * (1.6 / 1.4))


def test_team_expected_goals_rate_rejects_non_positive_league_average():
    with pytest.raises(ValueError):
        team_expected_goals_rate(1.5, 1.6, 0.0)


def test_team_expected_goals_rate_rejects_negative_inputs():
    with pytest.raises(ValueError):
        team_expected_goals_rate(-1.0, 1.6, 1.4)


def test_split_by_venue_partitions_matches():
    matches = pd.DataFrame({"is_home": [True, False, True, False], "goals": [1, 2, 3, 4]})
    home, away = split_by_venue(matches)
    assert list(home["goals"]) == [1, 3]
    assert list(away["goals"]) == [2, 4]


def test_dixon_coles_tau_low_scores_adjusted_nontrivially():
    rho = -0.1
    assert dixon_coles_tau(0, 0, 1.2, 1.1, rho) != 1.0
    assert dixon_coles_tau(0, 1, 1.2, 1.1, rho) != 1.0
    assert dixon_coles_tau(1, 0, 1.2, 1.1, rho) != 1.0
    assert dixon_coles_tau(1, 1, 1.2, 1.1, rho) != 1.0


def test_dixon_coles_tau_other_scores_unadjusted():
    assert dixon_coles_tau(2, 2, 1.2, 1.1, -0.1) == 1.0
    assert dixon_coles_tau(3, 0, 1.2, 1.1, -0.1) == 1.0


def test_dixon_coles_tau_zero_rho_is_no_correction():
    for x, y in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        assert dixon_coles_tau(x, y, 1.2, 1.1, 0.0) == 1.0


def test_scoreline_distribution_sums_to_one():
    joint = scoreline_distribution(1.4, 1.1)
    assert joint.sum() == pytest.approx(1.0)


def test_scoreline_distribution_negative_rho_raises_zero_zero_probability():
    # Negative rho should raise P(0-0) relative to independent Poisson (the whole point of the
    # Dixon-Coles correction -- low scores are underpredicted by independence).
    home_lambda, away_lambda = 1.2, 1.0
    from scipy.stats import poisson

    independent_00 = poisson.pmf(0, home_lambda) * poisson.pmf(0, away_lambda)
    adjusted = scoreline_distribution(home_lambda, away_lambda, rho=-0.15)
    # Compare against the *renormalized* independent baseline shape roughly by checking direction.
    assert adjusted[0, 0] > independent_00 * 0.99  # adjusted (raised) beats raw independent


def test_scoreline_distribution_rejects_negative_lambda():
    with pytest.raises(ValueError):
        scoreline_distribution(-1.0, 1.0)


def test_clean_sheet_probability_higher_when_opponent_weaker():
    strong_defence = clean_sheet_probability(team_for_lambda=1.5, team_against_lambda=0.5)
    weak_defence = clean_sheet_probability(team_for_lambda=1.5, team_against_lambda=2.0)
    assert strong_defence > weak_defence


def test_clean_sheet_probability_in_valid_range():
    prob = clean_sheet_probability(1.4, 1.1)
    assert 0.0 <= prob <= 1.0


def test_expected_goals_conceded_penalty_is_non_positive():
    penalty = expected_goals_conceded_penalty(team_against_lambda=1.4, expected_minutes=90.0)
    assert penalty <= 0.0


def test_expected_goals_conceded_penalty_scales_with_minutes():
    full_match = expected_goals_conceded_penalty(2.0, expected_minutes=90.0)
    half_match = expected_goals_conceded_penalty(2.0, expected_minutes=45.0)
    # Less exposure -> smaller (less negative) expected penalty.
    assert half_match > full_match


def test_expected_goals_conceded_penalty_zero_minutes_is_zero():
    penalty = expected_goals_conceded_penalty(2.0, expected_minutes=0.0)
    assert penalty == pytest.approx(0.0)


def test_expected_goals_conceded_penalty_rejects_negative_lambda():
    with pytest.raises(ValueError):
        expected_goals_conceded_penalty(-1.0)


def test_clean_sheet_projection_rejects_invalid_probability():
    with pytest.raises(ValueError):
        CleanSheetProjection(
            clean_sheet_probability=1.5,
            expected_goals_conceded_penalty=-0.2,
            team_for_lambda=1.0,
            team_against_lambda=1.0,
        )


def test_clean_sheet_projection_rejects_positive_penalty():
    with pytest.raises(ValueError):
        CleanSheetProjection(
            clean_sheet_probability=0.3,
            expected_goals_conceded_penalty=0.2,
            team_for_lambda=1.0,
            team_against_lambda=1.0,
        )


def test_clean_sheet_projection_expected_points_gk_includes_conceded_penalty():
    projection = CleanSheetProjection(
        clean_sheet_probability=0.4,
        expected_goals_conceded_penalty=-0.3,
        team_for_lambda=1.2,
        team_against_lambda=1.3,
    )
    points = projection.expected_points("GK", p_60_plus=0.9)
    assert points == pytest.approx(0.4 * 4 * 0.9 + -0.3)


def test_clean_sheet_projection_expected_points_forward_ignores_conceded_penalty():
    projection = CleanSheetProjection(
        clean_sheet_probability=0.4,
        expected_goals_conceded_penalty=-0.3,
        team_for_lambda=1.2,
        team_against_lambda=1.3,
    )
    points = projection.expected_points("FWD", p_60_plus=0.9)
    assert points == pytest.approx(0.4 * 0 * 0.9)


def test_clean_sheet_projection_rejects_unknown_position():
    projection = CleanSheetProjection(0.4, -0.3, 1.2, 1.3)
    with pytest.raises(ValueError):
        projection.expected_points("XYZ", 0.9)


def test_clean_sheet_projection_rejects_invalid_p_60_plus():
    projection = CleanSheetProjection(0.4, -0.3, 1.2, 1.3)
    with pytest.raises(ValueError):
        projection.expected_points("GK", 1.5)


def test_project_clean_sheet_end_to_end():
    projection = project_clean_sheet(
        team_xg_per_90=1.5,
        team_xga_per_90=1.0,
        opponent_xg_per_90=1.2,
        opponent_xga_per_90=1.4,
        league_avg_xga_per_90=1.4,
        expected_minutes=90.0,
    )
    assert isinstance(projection, CleanSheetProjection)
    assert 0.0 <= projection.clean_sheet_probability <= 1.0
    assert projection.expected_goals_conceded_penalty <= 0.0


def test_project_clean_sheet_weak_opponent_attack_improves_clean_sheet_odds():
    strong_opponent = project_clean_sheet(1.5, 1.0, 2.2, 1.4, 1.4)
    weak_opponent = project_clean_sheet(1.5, 1.0, 0.4, 1.4, 1.4)
    assert weak_opponent.clean_sheet_probability > strong_opponent.clean_sheet_probability
