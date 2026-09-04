"""Tests for features/fixtures.py — custom per-team fixture difficulty rating (BUILD_PLAN Phase
4)."""

from __future__ import annotations

import pytest

from features.fixtures import (
    FixtureDifficulty,
    TeamFixture,
    TeamRates,
    fixture_counts_by_gameweek,
    fixture_difficulty,
    fixture_expected_goals,
    league_average_rate,
    project_fixture_difficulties,
    project_fixture_expected_goals,
    team_horizon_difficulty,
)

LEAGUE_AVG_XG = 1.4
LEAGUE_AVG_XGA = 1.4


def test_fixture_difficulty_uses_opponent_away_rate_when_this_team_is_home():
    # This team is home, so the opponent is away -> opponent's AWAY rates apply.
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=10, is_home=True)
    opponent_rates = TeamRates(
        home_xg_per_90=2.0, away_xg_per_90=1.0, home_xga_per_90=1.5, away_xga_per_90=0.7
    )
    result = fixture_difficulty(fixture, opponent_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    assert result.attack_factor == pytest.approx(0.7 / LEAGUE_AVG_XGA)
    assert result.defense_factor == pytest.approx(1.0 / LEAGUE_AVG_XG)


def test_fixture_difficulty_uses_opponent_home_rate_when_this_team_is_away():
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=10, is_home=False)
    opponent_rates = TeamRates(
        home_xg_per_90=2.0, away_xg_per_90=1.0, home_xga_per_90=1.5, away_xga_per_90=0.7
    )
    result = fixture_difficulty(fixture, opponent_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    assert result.attack_factor == pytest.approx(1.5 / LEAGUE_AVG_XGA)
    assert result.defense_factor == pytest.approx(2.0 / LEAGUE_AVG_XG)


def test_fixture_difficulty_rejects_non_positive_league_averages():
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=10, is_home=True)
    rates = TeamRates(1.0, 1.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fixture_difficulty(fixture, rates, 0.0, LEAGUE_AVG_XGA)
    with pytest.raises(ValueError):
        fixture_difficulty(fixture, rates, LEAGUE_AVG_XG, -1.0)


def test_team_rates_rejects_negative_rate():
    with pytest.raises(ValueError):
        TeamRates(home_xg_per_90=-0.1, away_xg_per_90=1.0, home_xga_per_90=1.0, away_xga_per_90=1.0)


def test_leaky_weak_opponent_is_easy_attack_hard_defense_rating():
    # Opponent concedes a lot (high xGA) and scores a lot (high xG) -- an "end to end" leaky side:
    # easy fixture for goals/assists (low attack_rating) AND hard for a clean sheet (high
    # defense_rating).
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=1, is_home=True)
    opponent_rates = TeamRates(
        home_xg_per_90=1.0, away_xg_per_90=2.0, home_xga_per_90=1.0, away_xga_per_90=2.0
    )
    result = fixture_difficulty(fixture, opponent_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    assert result.attack_rating <= 2  # opponent's away xGA (2.0) >> league avg -> easy
    assert result.defense_rating >= 4  # opponent's away xG (2.0) >> league avg -> hard clean sheet


def test_solid_opponent_is_hard_attack_easy_defense_rating():
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=1, is_home=True)
    opponent_rates = TeamRates(
        home_xg_per_90=1.0, away_xg_per_90=0.6, home_xga_per_90=1.0, away_xga_per_90=0.6
    )
    result = fixture_difficulty(fixture, opponent_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    assert result.attack_rating >= 4  # opponent's away xGA (0.6) << league avg -> hard to score on
    assert result.defense_rating <= 2  # opponent's away xG (0.6) << league avg -> easy clean sheet


def test_overall_rating_averages_attack_and_defense():
    fd = FixtureDifficulty(
        team_id=1,
        opponent_id=2,
        gameweek=1,
        is_home=True,
        attack_factor=1.0,
        defense_factor=1.0,
        attack_rating=2,
        defense_rating=4,
    )
    assert fd.overall_rating == pytest.approx(3.0)


def test_project_fixture_difficulties_looks_up_each_fixtures_own_opponent():
    fixtures = [
        TeamFixture(team_id=1, opponent_id=2, gameweek=1, is_home=True),
        TeamFixture(team_id=1, opponent_id=3, gameweek=2, is_home=False),
    ]
    team_rates = {
        2: TeamRates(1.0, 1.0, 1.0, 1.0),
        3: TeamRates(2.0, 2.0, 2.0, 2.0),
    }
    results = project_fixture_difficulties(fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    assert len(results) == 2
    assert results[0].opponent_id == 2
    assert results[1].opponent_id == 3
    assert results[1].attack_factor == pytest.approx(2.0 / LEAGUE_AVG_XGA)


def test_team_horizon_difficulty_averages_factors_across_fixtures():
    fixtures = [
        TeamFixture(team_id=1, opponent_id=2, gameweek=1, is_home=True),
        TeamFixture(team_id=1, opponent_id=3, gameweek=2, is_home=True),
    ]
    team_rates = {
        2: TeamRates(1.0, 1.4, 1.0, 1.4),
        3: TeamRates(1.0, 1.4, 1.0, 1.4),
    }
    fds = project_fixture_difficulties(fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    horizon = team_horizon_difficulty(fds)
    assert horizon.team_id == 1
    assert horizon.gameweeks == (1, 2)
    assert horizon.fixture_count == 2
    assert horizon.mean_attack_factor == pytest.approx(1.0)
    assert horizon.mean_defense_factor == pytest.approx(1.0)


def test_team_horizon_difficulty_counts_double_gameweek_fixtures():
    fixtures = [
        TeamFixture(team_id=1, opponent_id=2, gameweek=5, is_home=True),
        TeamFixture(team_id=1, opponent_id=3, gameweek=5, is_home=False),
    ]
    team_rates = {2: TeamRates(1.0, 1.0, 1.0, 1.0), 3: TeamRates(1.0, 1.0, 1.0, 1.0)}
    fds = project_fixture_difficulties(fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA)
    horizon = team_horizon_difficulty(fds)
    assert horizon.fixture_count == 2
    assert horizon.gameweeks == (5, 5)


def test_team_horizon_difficulty_rejects_empty_input():
    with pytest.raises(ValueError):
        team_horizon_difficulty([])


def test_team_horizon_difficulty_rejects_mixed_teams():
    fd1 = fixture_difficulty(
        TeamFixture(1, 2, 1, True), TeamRates(1.0, 1.0, 1.0, 1.0), LEAGUE_AVG_XG, LEAGUE_AVG_XGA
    )
    fd2 = fixture_difficulty(
        TeamFixture(9, 2, 1, True), TeamRates(1.0, 1.0, 1.0, 1.0), LEAGUE_AVG_XG, LEAGUE_AVG_XGA
    )
    with pytest.raises(ValueError):
        team_horizon_difficulty([fd1, fd2])


def test_fixture_expected_goals_uses_venue_appropriate_splits_when_home():
    # Home team uses its own home split; the away opponent uses its own away split.
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=10, is_home=True)
    team_rates = TeamRates(
        home_xg_per_90=1.6, away_xg_per_90=1.2, home_xga_per_90=1.0, away_xga_per_90=1.3
    )
    opponent_rates = TeamRates(
        home_xg_per_90=2.0, away_xg_per_90=1.0, home_xga_per_90=1.5, away_xga_per_90=0.7
    )
    result = fixture_expected_goals(fixture, team_rates, opponent_rates, LEAGUE_AVG_XGA)
    assert result.expected_goals_for == pytest.approx(1.6 * (0.7 / LEAGUE_AVG_XGA))
    assert result.expected_goals_against == pytest.approx(1.0 * (1.0 / LEAGUE_AVG_XGA))


def test_fixture_expected_goals_uses_venue_appropriate_splits_when_away():
    # Away team uses its own away split; the home opponent uses its own home split.
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=10, is_home=False)
    team_rates = TeamRates(
        home_xg_per_90=1.6, away_xg_per_90=1.2, home_xga_per_90=1.0, away_xga_per_90=1.3
    )
    opponent_rates = TeamRates(
        home_xg_per_90=2.0, away_xg_per_90=1.0, home_xga_per_90=1.5, away_xga_per_90=0.7
    )
    result = fixture_expected_goals(fixture, team_rates, opponent_rates, LEAGUE_AVG_XGA)
    assert result.expected_goals_for == pytest.approx(1.2 * (1.5 / LEAGUE_AVG_XGA))
    assert result.expected_goals_against == pytest.approx(2.0 * (1.3 / LEAGUE_AVG_XGA))


def test_fixture_expected_goals_is_lower_against_a_stronger_defence():
    # Same attacking team, two candidate opponents: a strong defence (low xGA) and a weak one
    # (high xGA). This is the "Arsenal expects fewer goals against Man City than Coventry" check.
    fixture = TeamFixture(team_id=1, opponent_id=2, gameweek=10, is_home=True)
    team_rates = TeamRates(
        home_xg_per_90=2.2, away_xg_per_90=1.8, home_xga_per_90=1.0, away_xga_per_90=1.2
    )
    strong_defence = TeamRates(
        home_xg_per_90=1.5, away_xg_per_90=1.5, home_xga_per_90=0.6, away_xga_per_90=0.6
    )
    weak_defence = TeamRates(
        home_xg_per_90=1.5, away_xg_per_90=1.5, home_xga_per_90=2.0, away_xga_per_90=2.0
    )
    vs_strong = fixture_expected_goals(fixture, team_rates, strong_defence, LEAGUE_AVG_XGA)
    vs_weak = fixture_expected_goals(fixture, team_rates, weak_defence, LEAGUE_AVG_XGA)
    assert vs_strong.expected_goals_for < vs_weak.expected_goals_for


def test_project_fixture_expected_goals_looks_up_each_fixtures_own_team_and_opponent():
    fixtures = [
        TeamFixture(team_id=1, opponent_id=2, gameweek=1, is_home=True),
        TeamFixture(team_id=1, opponent_id=3, gameweek=2, is_home=False),
    ]
    team_rates = {
        1: TeamRates(1.5, 1.5, 1.0, 1.0),
        2: TeamRates(1.0, 1.0, 1.0, 1.0),
        3: TeamRates(2.0, 2.0, 2.0, 2.0),
    }
    results = project_fixture_expected_goals(fixtures, team_rates, LEAGUE_AVG_XGA)
    assert len(results) == 2
    assert results[0].opponent_id == 2
    assert results[1].opponent_id == 3
    assert results[1].expected_goals_for == pytest.approx(1.5 * (2.0 / LEAGUE_AVG_XGA))


def test_league_average_rate_recovers_underlying_shrunk_rate():
    team_rates = {
        1: TeamRates(1.6, 1.2, 1.0, 1.0),
        2: TeamRates(1.0, 1.0, 1.0, 1.0),
    }
    # Team 1's own underlying xg rate is (1.6 + 1.2) / 2 = 1.4; team 2's is 1.0.
    assert league_average_rate(team_rates, "home_xg_per_90", "away_xg_per_90") == pytest.approx(
        (1.4 + 1.0) / 2
    )


def test_fixture_counts_by_gameweek_marks_blanks_and_doubles():
    fixtures = [
        TeamFixture(team_id=1, opponent_id=2, gameweek=1, is_home=True),
        TeamFixture(team_id=1, opponent_id=3, gameweek=3, is_home=False),
        TeamFixture(team_id=1, opponent_id=4, gameweek=3, is_home=True),
        TeamFixture(team_id=9, opponent_id=2, gameweek=2, is_home=True),  # different team
    ]
    counts = fixture_counts_by_gameweek(fixtures, team_id=1, gameweeks=[1, 2, 3])
    assert counts == {1: 1, 2: 0, 3: 2}
