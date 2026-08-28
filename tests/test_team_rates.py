"""Tests for engine/data/team_rates.py — the shared team-rate snapshot math moved out of
backtest/run_season.py (ENGINE_IMPROVEMENTS_3.md A.1/B1), plus the live-app entry point
build_current_team_rates built on top of it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from engine.data.team_rates import (
    TeamRateSnapshot,
    _league_venue_multipliers,
    _team_prior_match_count,
    _team_rate_asof,
    _team_rate_asof_shrunk,
    _team_venue_multipliers,
    build_current_team_rates,
)


def test_team_prior_match_count_counts_strictly_prior_matches():
    history = pd.DataFrame(
        {"date": pd.to_datetime(["2025-08-01", "2025-08-08", "2025-08-15"], utc=True)}
    )
    assert _team_prior_match_count(history, pd.Timestamp("2025-08-10", tz="UTC")) == 2
    assert _team_prior_match_count(history, pd.Timestamp("2025-07-01", tz="UTC")) == 0


def test_team_prior_match_count_empty_history_is_zero():
    empty = pd.DataFrame(columns=["date"])
    assert _team_prior_match_count(empty, pd.Timestamp("2025-08-10", tz="UTC")) == 0


def test_team_rate_asof_shrunk_pulls_a_thin_sample_toward_the_league_average():
    # ENGINE_IMPROVEMENTS_3.md A.1: one match of a wildly unusual rate should not be taken at face
    # value the way the previous per-team venue split effectively did.
    history = pd.DataFrame(
        {
            "date": pd.to_datetime(["2025-08-01"], utc=True),
            "xG": [4.0],
            "xGA": [4.0],
            "minutes": [90.0],
            "is_home": [True],
        }
    )
    before = pd.Timestamp("2025-08-15", tz="UTC")
    raw = _team_rate_asof(history, "xG", before)
    assert raw == pytest.approx(4.0)

    shrunk = _team_rate_asof_shrunk(history, "xG", before, league_avg=1.4, shrinkage_k=4.0)
    # n_prior=1, k=4 -> weight mostly on the prior: (1*4.0 + 4*1.4) / 5 = 1.92
    assert shrunk == pytest.approx((1 * 4.0 + 4 * 1.4) / 5)
    assert 1.4 < shrunk < 4.0


def test_team_rate_asof_shrunk_barely_moves_a_deep_sample():
    history = pd.DataFrame(
        {
            "date": pd.to_datetime([f"2025-{m:02d}-01" for m in range(1, 13)], utc=True),
            "xG": [1.5] * 12,
            "xGA": [1.5] * 12,
            "minutes": [90.0] * 12,
            "is_home": [True] * 12,
        }
    )
    before = pd.Timestamp("2026-01-01", tz="UTC")
    shrunk = _team_rate_asof_shrunk(history, "xG", before, league_avg=1.0, shrinkage_k=4.0)
    # n_prior=12, k=4 -> (12*1.5 + 4*1.0) / 16 = 1.375: most of the weight stays on the
    # individual rate, unlike the single-match case above where the prior dominates.
    assert shrunk == pytest.approx((12 * 1.5 + 4 * 1.0) / 16)
    assert shrunk > 1.375 - 1e-9 and shrunk < 1.5


def test_team_rate_asof_shrunk_falls_back_to_league_average_for_a_promoted_club():
    # A newly-promoted club has zero top-flight matches, so its own raw rate is undefined. Rather
    # than propagating that NaN (which would drop every opponent's fixture row for the club's
    # first ever match via the required-columns dropna downstream), full shrinkage lands on the
    # league-average rate, matching what shrink_toward_prior already does for a zero-weight
    # individual rate everywhere else it's used.
    empty = pd.DataFrame(columns=["date", "xG", "xGA", "minutes", "is_home"])
    before = pd.Timestamp("2025-08-15", tz="UTC")
    assert _team_rate_asof_shrunk(empty, "xG", before, league_avg=1.4) == pytest.approx(1.4)


def test_team_rate_asof_shrunk_returns_nan_when_league_average_itself_is_undefined():
    empty = pd.DataFrame(columns=["date", "xG", "xGA", "minutes", "is_home"])
    before = pd.Timestamp("2025-08-15", tz="UTC")
    assert pd.isna(_team_rate_asof_shrunk(empty, "xG", before, league_avg=float("nan")))


def test_league_venue_multipliers_reflects_a_real_home_advantage():
    # Every team scores 2.0 at home and 1.0 away -- a real, sizeable, league-wide venue effect.
    rows = []
    for _team_id in range(20):
        for _match in range(5):
            rows.append(
                {
                    "date": pd.Timestamp("2025-08-01", tz="UTC"),
                    "xG": 2.0,
                    "xGA": 1.0,
                    "is_home": True,
                }
            )
            rows.append(
                {
                    "date": pd.Timestamp("2025-08-01", tz="UTC"),
                    "xG": 1.0,
                    "xGA": 2.0,
                    "is_home": False,
                }
            )
    team_histories = {str(i): pd.DataFrame(rows) for i in range(20)}
    xg_mult, xga_mult = _league_venue_multipliers(
        team_histories, pd.Timestamp("2025-09-01", tz="UTC")
    )
    assert xg_mult == pytest.approx(2.0 / 1.5)
    assert xga_mult == pytest.approx(1.0 / 1.5)


def test_league_venue_multipliers_neutral_below_the_minimum_match_count():
    rows = [{"date": pd.Timestamp("2025-08-01", tz="UTC"), "xG": 2.0, "xGA": 1.0, "is_home": True}]
    team_histories = {"1": pd.DataFrame(rows)}
    xg_mult, xga_mult = _league_venue_multipliers(
        team_histories, pd.Timestamp("2025-08-15", tz="UTC")
    )
    assert (xg_mult, xga_mult) == (1.0, 1.0)


def test_league_venue_multipliers_neutral_when_no_team_histories():
    xg_mult, xga_mult = _league_venue_multipliers({}, pd.Timestamp("2025-08-15", tz="UTC"))
    assert (xg_mult, xga_mult) == (1.0, 1.0)


def _venue_history(n_home: int, home_xg: float, away_xg: float) -> pd.DataFrame:
    rows = []
    for _match in range(n_home):
        rows.append(
            {
                "date": pd.Timestamp("2025-08-01", tz="UTC"),
                "xG": home_xg,
                "xGA": 1.0,
                "is_home": True,
            }
        )
        rows.append(
            {
                "date": pd.Timestamp("2025-08-01", tz="UTC"),
                "xG": away_xg,
                "xGA": 1.0,
                "is_home": False,
            }
        )
    return pd.DataFrame(rows)


def test_team_venue_multipliers_pulls_a_thin_sample_toward_the_league_multiplier():
    # B1: a team with only 2 home matches showing an extreme home advantage should end up much
    # closer to the league-wide multiplier than to its own noisy estimate.
    history = _venue_history(n_home=2, home_xg=3.0, away_xg=1.0)  # own raw xg_mult = 3.0/2.0 = 1.5
    before = pd.Timestamp("2025-09-01", tz="UTC")

    xg_mult, _ = _team_venue_multipliers(history, before, (1.1, 0.9), shrinkage_k=10.0)

    # n=2 against k=10 -> (2*1.5 + 10*1.1) / 12 = 1.1667, i.e. a sixth of the way toward its own.
    assert xg_mult == pytest.approx((2 * 1.5 + 10 * 1.1) / 12)
    assert abs(xg_mult - 1.1) < abs(xg_mult - 1.5)


def test_team_venue_multipliers_trusts_a_deep_sample():
    history = _venue_history(n_home=40, home_xg=3.0, away_xg=1.0)
    before = pd.Timestamp("2025-09-01", tz="UTC")

    xg_mult, _ = _team_venue_multipliers(history, before, (1.1, 0.9), shrinkage_k=10.0)

    assert xg_mult == pytest.approx((40 * 1.5 + 10 * 1.1) / 50)
    assert abs(xg_mult - 1.5) < abs(xg_mult - 1.1)


def test_team_venue_multipliers_falls_back_to_the_league_pair_without_both_venues():
    before = pd.Timestamp("2025-09-01", tz="UTC")
    league = (1.1, 0.9)

    assert _team_venue_multipliers(pd.DataFrame(), before, league) == league
    home_only = pd.DataFrame(
        [{"date": pd.Timestamp("2025-08-01", tz="UTC"), "xG": 2.0, "xGA": 1.0, "is_home": True}]
    )
    # Every prior match at home -- no away baseline to form a ratio against.
    assert _team_venue_multipliers(home_only, before, league) == league


def _flat_history(n_matches: int, xg: float, xga: float) -> pd.DataFrame:
    """A team history with the same xG/xGA in every match, split evenly home/away so the venue
    multiplier math has no effect and the resulting rate is easy to hand-compute."""
    rows = []
    for i in range(n_matches):
        rows.append(
            {
                "date": pd.Timestamp("2025-08-01", tz="UTC") + pd.Timedelta(days=7 * i),
                "xG": xg,
                "xGA": xga,
                "minutes": 90.0,
                "is_home": i % 2 == 0,
            }
        )
    return pd.DataFrame(rows)


def test_build_current_team_rates_empty_histories_returns_empty():
    assert build_current_team_rates({}, pd.Timestamp("2026-01-01", tz="UTC")) == {}


def test_build_current_team_rates_shrinks_established_teams_and_backfills_a_promoted_club():
    as_of = pd.Timestamp("2026-01-01", tz="UTC")
    team_histories = {
        "Team A": _flat_history(10, xg=3.0, xga=1.0),
        "Team C": _flat_history(10, xg=1.0, xga=1.0),
        "Team B": pd.DataFrame(columns=["date", "xG", "xGA", "minutes", "is_home"]),
    }

    snapshots = build_current_team_rates(team_histories, as_of)

    assert set(snapshots) == {"Team A", "Team B", "Team C"}
    assert all(isinstance(snapshot, TeamRateSnapshot) for snapshot in snapshots.values())

    # Both teams' 20 combined matches fall well short of the league-wide venue-multiplier minimum
    # (40), so every multiplier here is the neutral (1.0, 1.0) fallback and home == away == the
    # shrunk rate for every team -- the simplest case to hand-verify the shrinkage math itself.
    # league_avg_xg = mean(3.0, 1.0) = 2.0 (Team B's NaN raw rate is excluded from the average).
    expected_a_xg = (10 * 3.0 + 4 * 2.0) / 14  # TEAM_RATE_SHRINKAGE_K default = 4.0
    expected_c_xg = (10 * 1.0 + 4 * 2.0) / 14

    a = snapshots["Team A"]
    assert a.home_xg_per_90 == pytest.approx(expected_a_xg)
    assert a.away_xg_per_90 == pytest.approx(expected_a_xg)
    assert a.home_xga_per_90 == pytest.approx(1.0)
    assert a.away_xga_per_90 == pytest.approx(1.0)

    c = snapshots["Team C"]
    assert c.home_xg_per_90 == pytest.approx(expected_c_xg)
    assert c.away_xg_per_90 == pytest.approx(expected_c_xg)

    # Team B has no history at all, so full shrinkage lands it exactly on the league average.
    b = snapshots["Team B"]
    assert b.home_xg_per_90 == pytest.approx(2.0)
    assert b.away_xg_per_90 == pytest.approx(2.0)
    assert b.home_xga_per_90 == pytest.approx(1.0)
    assert b.away_xga_per_90 == pytest.approx(1.0)
