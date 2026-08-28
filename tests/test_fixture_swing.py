"""Tests for features/fixture_swing.py — per-team fixture-difficulty swing between a near and a
far gameweek window."""

from __future__ import annotations

from features.fixture_swing import compute_team_swing, rank_team_swings
from features.fixtures import TeamFixture, TeamRates

LEAGUE_AVG_XG = 1.4
LEAGUE_AVG_XGA = 1.4

NEAR_GAMEWEEKS = [1, 2, 3]
FAR_GAMEWEEKS = [4, 5, 6, 7, 8]


def _fixtures_against(team_id: int, opponent_id: int, gameweeks: list[int]) -> list[TeamFixture]:
    return [
        TeamFixture(team_id=team_id, opponent_id=opponent_id, gameweek=gw, is_home=True)
        for gw in gameweeks
    ]


def test_compute_team_swing_is_positive_for_an_improving_attacking_run():
    # Near opponent defends well (low xGA, hard to score against); far opponent is leaky (easy).
    # Both opponents concede the league-average xG, so the defense dimension stays flat -- this
    # isolates the attack swing.
    fixtures = _fixtures_against(1, 2, NEAR_GAMEWEEKS) + _fixtures_against(1, 3, FAR_GAMEWEEKS)
    team_rates = {
        2: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=0.5, away_xga_per_90=0.5
        ),
        3: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=3.0, away_xga_per_90=3.0
        ),
    }

    swing = compute_team_swing(
        1, fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA, NEAR_GAMEWEEKS, FAR_GAMEWEEKS
    )

    assert swing.near.attack_rating > swing.far.attack_rating
    assert swing.attack_swing > 0
    assert swing.defense_swing == 0


def test_compute_team_swing_is_negative_for_a_worsening_attacking_run():
    # Same two opponents, windows swapped: near is now the easy leaky side, far the hard one.
    fixtures = _fixtures_against(1, 2, NEAR_GAMEWEEKS) + _fixtures_against(1, 3, FAR_GAMEWEEKS)
    team_rates = {
        2: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=3.0, away_xga_per_90=3.0
        ),
        3: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=0.5, away_xga_per_90=0.5
        ),
    }

    swing = compute_team_swing(
        1, fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA, NEAR_GAMEWEEKS, FAR_GAMEWEEKS
    )

    assert swing.near.attack_rating < swing.far.attack_rating
    assert swing.attack_swing < 0
    assert swing.defense_swing == 0


def test_compute_team_swing_is_zero_for_a_flat_run():
    # Same opponent, same rates, in both windows -- nothing to swing toward or away from.
    fixtures = _fixtures_against(1, 2, NEAR_GAMEWEEKS) + _fixtures_against(1, 2, FAR_GAMEWEEKS)
    team_rates = {
        2: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=1.4, away_xga_per_90=1.4
        ),
    }

    swing = compute_team_swing(
        1, fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA, NEAR_GAMEWEEKS, FAR_GAMEWEEKS
    )

    assert swing.near.attack_rating == swing.far.attack_rating
    assert swing.near.defense_rating == swing.far.defense_rating
    assert swing.attack_swing == 0
    assert swing.defense_swing == 0


def test_compute_team_swing_returns_none_swings_for_a_genuinely_blank_window():
    # This team has no fixture at all in the far window (e.g. a mid-season blank) -- a real,
    # expected fixture-list state (api.fixtures_view's own convention for the same case), not an
    # error, so the far side (and the swings, which need both ends) come back None rather than
    # raising.
    fixtures = _fixtures_against(1, 2, NEAR_GAMEWEEKS)
    team_rates = {
        2: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=1.4, away_xga_per_90=1.4
        ),
    }

    swing = compute_team_swing(
        1, fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA, NEAR_GAMEWEEKS, FAR_GAMEWEEKS
    )

    assert swing.near is not None
    assert swing.far is None
    assert swing.attack_swing is None
    assert swing.defense_swing is None


def test_rank_team_swings_returns_one_entry_per_team_unsorted():
    fixtures = _fixtures_against(1, 3, NEAR_GAMEWEEKS + FAR_GAMEWEEKS) + _fixtures_against(
        2, 3, NEAR_GAMEWEEKS + FAR_GAMEWEEKS
    )
    team_rates = {
        3: TeamRates(
            home_xg_per_90=1.4, away_xg_per_90=1.4, home_xga_per_90=1.4, away_xga_per_90=1.4
        ),
    }

    swings = rank_team_swings(
        [1, 2], fixtures, team_rates, LEAGUE_AVG_XG, LEAGUE_AVG_XGA, NEAR_GAMEWEEKS, FAR_GAMEWEEKS
    )

    assert [s.team_id for s in swings] == [1, 2]
