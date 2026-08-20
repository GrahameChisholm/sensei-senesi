"""Tests for engine.data.live_horizon, which builds the "fit once, predict/simulate per gameweek"
multi-gameweek horizon from a live snapshot instead of a replayed historical season.

All synthetic (no network, no real snapshot files) -- ``build_live_horizon_from_feature_inputs``
is exercised directly against a hand-built ``FeatureInputs``, the same disk-free approach
``tests/test_live_adapter.py`` already uses. ``build_live_horizon`` itself (the thin,
disk-touching wrapper) is covered separately by monkeypatching
``snapshot_to_feature_inputs`` -- real snapshot files aren't needed to prove it builds the right
``target_gameweeks`` and delegates correctly.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from engine.data.cross_season import (
    player_code_map,
    prior_season_merged_gw,
    remap_player_histories,
    synthetic_team_rows,
    team_id_map,
)
from engine.data.live_adapter import (
    FeatureInputs,
    build_merged_gw,
    build_player_histories_from_live_snapshot,
)
from engine.data.live_horizon import (
    augment_feature_inputs_with_prior_season,
    build_live_horizon,
    build_live_horizon_from_feature_inputs,
)

TEAM_A, TEAM_B = 10, 20
# GW1..GW8 -- enough real history (5 played rounds) to clear fit_fn's cold-start zone, plus a
# 3-gameweek horizon (GW6, 7, 8) beyond it.
KICKOFFS = pd.date_range("2025-08-16", periods=8, freq="7D", tz="UTC")


def _teams() -> pd.DataFrame:
    # `code` (FPL's season-stable club id) is unused by build_merged_gw/engineer_features but
    # needed by engine.data.cross_season's team_id_map/synthetic_team_rows -- included here rather
    # than in a parallel fixture so the two "live teams" views can never silently drift apart.
    return pd.DataFrame(
        [
            {"id": TEAM_A, "code": 100, "name": "Team A"},
            {"id": TEAM_B, "code": 200, "name": "Team B"},
        ]
    )


def _elements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "code": 1001,
                "team": TEAM_A,
                "element_type": 3,  # MID
                "now_cost": 75,
                "selected_by_percent": "10.0",
                "transfers_in_event": 1000,
                "transfers_out_event": 200,
            },
            {
                "id": 2,
                "code": 1002,
                "team": TEAM_B,
                "element_type": 3,  # MID
                "now_cost": 60,
                "selected_by_percent": "5.0",
                "transfers_in_event": 100,
                "transfers_out_event": 50,
            },
        ]
    )


def _history_row(element: int, round_: int, opponent: int, was_home: bool) -> dict:
    return {
        "element": element,
        "round": round_,
        "opponent_team": opponent,
        "was_home": was_home,
        "kickoff_time": KICKOFFS[round_ - 1].isoformat(),
        "team_h_score": 2,
        "team_a_score": 1,
        "minutes": 90,
        "total_points": 6,
        "goals_scored": 1,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "defensive_contribution": 3,
        "own_goals": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bps": 25,
        "bonus": 1,
        "penalties_missed": 0,
        "starts": 1,
        "value": 70,
        "selected": 50000,
        "transfers_out": 1000,
        "transfers_balance": 500,
    }


def _element_summary_histories(n_played_rounds: int) -> pd.DataFrame:
    rows = []
    for round_ in range(1, n_played_rounds + 1):
        home = round_ % 2 == 1
        rows.append(_history_row(1, round_, TEAM_B, home))
        rows.append(_history_row(2, round_, TEAM_A, not home))
    return pd.DataFrame(rows)


def _target_fixtures(target_gameweeks: list[int]) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame(
                [
                    {
                        "event": gw,
                        "team_h": TEAM_A,
                        "team_a": TEAM_B,
                        "kickoff_time": KICKOFFS[gw - 1].isoformat(),
                    }
                ]
            )
            for gw in target_gameweeks
        ],
        ignore_index=True,
    )


def _understat_player_match(fpl_id: int, round_: int, npxg: float, xa: float) -> dict:
    return {
        "fpl_id": fpl_id,
        "date": KICKOFFS[round_ - 1].isoformat(),
        "season": "2025",
        "time": "90",
        "npxG": str(npxg),
        "xA": str(xa),
        "goals": "1",
        "npg": "1",
    }


def _understat_player_histories(n_played_rounds: int) -> pd.DataFrame:
    rows = []
    for round_ in range(1, n_played_rounds + 1):
        rows.append(_understat_player_match(1, round_, 0.3, 0.2))
        rows.append(_understat_player_match(2, round_, 0.25, 0.15))
    return pd.DataFrame(rows)


def _team_history(n_played_rounds: int, xg: float, xga: float, is_home: bool) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": KICKOFFS[:n_played_rounds],
            "xG": xg,
            "xGA": xga,
            "minutes": 90.0,
            "is_home": is_home,
        }
    )


def _feature_inputs(
    current_gameweek: int, target_gameweeks: list[int], n_played_rounds: int
) -> FeatureInputs:
    merged_gw = build_merged_gw(
        _elements(),
        _teams(),
        _target_fixtures(target_gameweeks),
        _element_summary_histories(n_played_rounds),
        gameweek=current_gameweek,
        target_gameweeks=target_gameweeks,
    )
    team_histories = {
        "Team A": _team_history(n_played_rounds, xg=1.5, xga=0.7, is_home=True),
        "Team B": _team_history(n_played_rounds, xg=0.8, xga=1.4, is_home=False),
    }
    player_histories = build_player_histories_from_live_snapshot(
        _understat_player_histories(n_played_rounds), season_start_year=2025
    )
    return FeatureInputs(
        merged_gw=merged_gw,
        teams=_teams(),
        team_histories=team_histories,
        player_histories=player_histories,
    )


def test_build_live_horizon_from_feature_inputs_covers_every_target_gameweek():
    feature_inputs = _feature_inputs(
        current_gameweek=6, target_gameweeks=[6, 7, 8], n_played_rounds=5
    )

    result = build_live_horizon_from_feature_inputs(
        feature_inputs, current_gameweek=6, target_gameweeks=[6, 7, 8], n_simulation_runs=10, seed=1
    )

    assert set(result.predictions["gameweek"].unique()) == {6, 7, 8}
    assert set(result.projections) == {1, 2}
    for horizon in result.projections.values():
        assert set(horizon.gameweeks) == {6, 7, 8}
        assert horizon.horizon_total_points == pytest.approx(
            sum(p.expected_points for p in horizon.gameweeks.values())
        )
        for projection in horizon.gameweeks.values():
            b = projection.breakdown
            assert b.total == pytest.approx(
                b.appearance
                + b.goals
                + b.assists
                + b.clean_sheet
                + b.goals_conceded
                + b.defensive_contribution
                + b.saves
                + b.bonus
                + b.cards
                + b.penalty_misses
                + b.own_goals
            )


def test_build_live_horizon_from_feature_inputs_raises_when_too_little_history():
    feature_inputs = _feature_inputs(
        current_gameweek=2, target_gameweeks=[2, 3, 4], n_played_rounds=1
    )

    with pytest.raises(ValueError, match="too early in the season"):
        build_live_horizon_from_feature_inputs(
            feature_inputs,
            current_gameweek=2,
            target_gameweeks=[2, 3, 4],
            n_simulation_runs=10,
            seed=1,
        )


def test_build_live_horizon_computes_target_gameweeks_and_delegates(monkeypatch):
    captured_target_gameweeks = {}

    def fake_snapshot_to_feature_inputs(
        season,
        gameweek,
        captured_at,
        understat_season_start_year,
        base_dir,
        total_managers,
        understat_client,
        prior_season_cache_dir,
        n_prior_seasons,
        target_gameweeks=None,
    ):
        captured_target_gameweeks["value"] = target_gameweeks
        return _feature_inputs(
            current_gameweek=gameweek, target_gameweeks=target_gameweeks, n_played_rounds=5
        )

    monkeypatch.setattr(
        "engine.data.live_horizon.snapshot_to_feature_inputs", fake_snapshot_to_feature_inputs
    )

    result = build_live_horizon(
        season="2025-26",
        current_gameweek=6,
        captured_at=datetime(2025, 10, 1, tzinfo=UTC),
        understat_season_start_year=2025,
        n_simulation_runs=10,
        seed=1,
    )

    assert captured_target_gameweeks["value"] == [6, 7, 8]
    assert set(result.predictions["gameweek"].unique()) == {6, 7, 8}


# --- augment_feature_inputs_with_prior_season: the true GW1 cold-start case --------------------
# A season's real opening gameweek: no this-season history at all (n_played_rounds=0). Proves the
# documented crash-avoidance path fires without augmentation, and that prepending re-keyed
# prior-season rows (engine.data.cross_season) actually unblocks it -- not just that the plumbing
# doesn't crash on its own synthetic inputs.

PRIOR_KICKOFFS = pd.date_range("2024-08-17", periods=5, freq="7D", tz="UTC")


def _prior_teams() -> pd.DataFrame:
    # A surviving club's real-world name/short_name never changes season to season -- only its
    # numeric id can be reassigned by promotion/relegation churn -- so these intentionally match
    # `_teams()`'s own names exactly; only `id` differs, exactly like a real cross-season pairing.
    return pd.DataFrame(
        [
            {"id": 910, "code": 100, "name": "Team A", "short_name": "TMA"},
            {"id": 920, "code": 200, "name": "Team B", "short_name": "TMB"},
        ]
    )


def _prior_players_raw() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 901,
                "code": 1001,
                "web_name": "P1",
                "first_name": "P",
                "second_name": "One",
                "team": 910,
                "element_type": 3,
            },
            {
                "id": 902,
                "code": 1002,
                "web_name": "P2",
                "first_name": "P",
                "second_name": "Two",
                "team": 920,
                "element_type": 3,
            },
        ]
    )


def _prior_merged_gw_row(
    element: int, round_: int, opponent: int, team_name: str, was_home: bool
) -> dict:
    return {
        "element": element,
        "round": round_,
        "position": "MID",
        "team": team_name,
        "opponent_team": opponent,
        "was_home": was_home,
        "kickoff_time": PRIOR_KICKOFFS[round_ - 1].isoformat(),
        "team_h_score": 2,
        "team_a_score": 1,
        "value": 70,
        "selected": 50000,
        "transfers_out": 1000,
        "transfers_balance": 500,
        "minutes": 90,
        "total_points": 6,
        "goals_scored": 1,
        "assists": 0,
        "clean_sheets": 0,
        "goals_conceded": 1,
        "defensive_contribution": 3,
        "own_goals": 0,
        "yellow_cards": 0,
        "red_cards": 0,
        "saves": 0,
        "bps": 25,
        "bonus": 1,
        "penalties_missed": 0,
        "starts": 1,
    }


def _prior_merged_gw() -> pd.DataFrame:
    rows = []
    for round_ in range(1, 6):
        home = round_ % 2 == 1
        rows.append(_prior_merged_gw_row(901, round_, 920, "Team A", home))
        rows.append(_prior_merged_gw_row(902, round_, 910, "Team B", not home))
    return pd.DataFrame(rows)


def _prior_player_histories_by_prior_id() -> dict[int, pd.DataFrame]:
    return {
        901: pd.DataFrame(
            {"date": PRIOR_KICKOFFS, "time": 90.0, "npxG": 0.3, "xA": 0.2, "goals": 1.0, "npg": 1.0}
        ),
        902: pd.DataFrame(
            {
                "date": PRIOR_KICKOFFS,
                "time": 90.0,
                "npxG": 0.25,
                "xA": 0.15,
                "goals": 0.0,
                "npg": 0.0,
            }
        ),
    }


def _pooled_team_history(xg: float, xga: float, is_home: bool) -> pd.DataFrame:
    """Stand-in for ``snapshot_to_feature_inputs``'s own already-existing team-rate prior-season
    pooling (its ``understat_client``/``prior_season_cache_dir`` params) -- spans both the prior
    season's kickoff dates (``PRIOR_KICKOFFS``) and this season's (``KICKOFFS``), matching what a
    real pooled team-rate history actually looks like, so the "as of this kickoff" team-rate
    lookup has something strictly before *every* row in the test, prior-season training rows
    included -- not just the current-season ones ``_team_history`` alone would cover."""
    dates = list(PRIOR_KICKOFFS) + list(KICKOFFS[:5])
    return pd.DataFrame({"date": dates, "xG": xg, "xGA": xga, "minutes": 90.0, "is_home": is_home})


def _gw1_feature_inputs_with_team_rates_already_pooled() -> FeatureInputs:
    """A true season-opener live snapshot: zero in-season rows in ``merged_gw``/``player_histories``
    (nothing has been played yet), but ``team_histories`` already has real matches — a separate,
    already-solved concern (see :func:`_pooled_team_history`) this test deliberately holds constant
    so it isolates exactly what :func:`augment_feature_inputs_with_prior_season` is meant to
    additionally fix."""
    merged_gw = build_merged_gw(
        _elements(),
        _teams(),
        _target_fixtures([1, 2, 3]),
        _element_summary_histories(0),
        gameweek=1,
        target_gameweeks=[1, 2, 3],
    )
    team_histories = {
        "Team A": _pooled_team_history(xg=1.5, xga=0.7, is_home=True),
        "Team B": _pooled_team_history(xg=0.8, xga=1.4, is_home=False),
    }
    player_histories = build_player_histories_from_live_snapshot(
        _understat_player_histories(0), season_start_year=2025
    )
    return FeatureInputs(
        merged_gw=merged_gw,
        teams=_teams(),
        team_histories=team_histories,
        player_histories=player_histories,
    )


def test_augment_feature_inputs_with_prior_season_unblocks_a_true_gw1():
    feature_inputs = _gw1_feature_inputs_with_team_rates_already_pooled()

    # Without augmentation: exactly the documented crash-avoidance path, not an opaque sklearn
    # error.
    with pytest.raises(ValueError, match="too early in the season"):
        build_live_horizon_from_feature_inputs(
            feature_inputs,
            current_gameweek=1,
            target_gameweeks=[1, 2, 3],
            n_simulation_runs=10,
            seed=1,
        )

    code_map = player_code_map(_prior_players_raw(), _elements())
    team_map = team_id_map(_prior_teams(), _teams())
    synthetic_teams = synthetic_team_rows(_prior_teams(), _teams())  # empty: nobody relegated here
    prior_merged_gw = prior_season_merged_gw(_prior_merged_gw(), code_map, team_map, {})
    prior_histories = remap_player_histories(_prior_player_histories_by_prior_id(), code_map)

    augmented = augment_feature_inputs_with_prior_season(
        feature_inputs, prior_merged_gw, synthetic_teams, prior_histories
    )

    result = build_live_horizon_from_feature_inputs(
        augmented, current_gameweek=1, target_gameweeks=[1, 2, 3], n_simulation_runs=10, seed=1
    )

    assert set(result.predictions["gameweek"].unique()) == {1, 2, 3}
    assert set(result.projections) == {1, 2}
    for horizon in result.projections.values():
        assert set(horizon.gameweeks) == {1, 2, 3}
