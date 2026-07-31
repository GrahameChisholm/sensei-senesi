"""Tests for scripts/weekly_refresh.py's real (A1/A2) hooks — make_build_pool_projections,
build_player_horizon_projections, build_app_state_from_predictions. No network: the snapshot these
tests read is written directly via engine.data.snapshots.capture_snapshot with synthetic tables,
matching tests/test_live_adapter.py's fixture shapes.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from engine.data.snapshots import capture_snapshot
from scripts.weekly_refresh import (
    build_app_state_from_predictions,
    build_player_horizon_projections,
    make_build_app_state,
    make_build_pool_projections,
)

TEAM_A, TEAM_B = 10, 20
SEASON = "2025-26"
CAPTURED_AT = datetime(2025, 9, 19, 9, 0, tzinfo=UTC)
TARGET_GAMEWEEK = 3


def _teams() -> pd.DataFrame:
    return pd.DataFrame([{"id": TEAM_A, "name": "Team A"}, {"id": TEAM_B, "name": "Team B"}])


def _elements() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "id": 1,
                "team": TEAM_A,
                "element_type": 3,
                "now_cost": 75,
                "selected_by_percent": "10.0",
                "transfers_in_event": 1000,
                "transfers_out_event": 200,
                "chance_of_playing_next_round": None,
                "status": "a",
            },
            {
                "id": 2,
                "team": TEAM_B,
                "element_type": 3,
                "now_cost": 60,
                "selected_by_percent": "5.0",
                "transfers_in_event": 100,
                "transfers_out_event": 50,
                "chance_of_playing_next_round": 75.0,
                "status": "d",
            },
        ]
    )


def _fixtures() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event": TARGET_GAMEWEEK,
                "team_h": TEAM_A,
                "team_a": TEAM_B,
                "kickoff_time": "2025-09-20T14:00:00Z",
            }
        ]
    )


def _history_row(element: int, round_: int, opponent: int, was_home: bool) -> dict:
    return {
        "element": element,
        "round": round_,
        "opponent_team": opponent,
        "was_home": was_home,
        "kickoff_time": f"2025-08-{9 + round_ * 7:02d}T14:00:00Z",
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


def _element_summary_histories() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _history_row(1, 1, TEAM_B, True),
            _history_row(1, 2, TEAM_B, False),
            _history_row(2, 1, TEAM_A, False),
            _history_row(2, 2, TEAM_A, True),
        ]
    )


def _understat_player_match(fpl_id: int, date: str) -> dict:
    return {
        "fpl_id": fpl_id,
        "date": date,
        "season": "2025",
        "time": "90",
        "npxG": "0.3",
        "xA": "0.2",
        "goals": "1",
        "npg": "1",
    }


def _understat_player_histories() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _understat_player_match(1, "2025-08-16"),
            _understat_player_match(1, "2025-08-23"),
            _understat_player_match(2, "2025-08-16"),
            _understat_player_match(2, "2025-08-23"),
        ]
    )


def _understat_teams_history() -> pd.DataFrame:
    rows = []
    for team_title, h_a in [("Team A", "h"), ("Team B", "a")]:
        for date in ("2025-08-16", "2025-08-23"):
            rows.append({"team_title": team_title, "date": date, "xG": 1.5, "xGA": 1.1, "h_a": h_a})
    return pd.DataFrame(rows)


@pytest.fixture
def snapshot_dir(tmp_path):
    base_dir = tmp_path / "snapshots"
    capture_snapshot(
        season=SEASON,
        gameweek=TARGET_GAMEWEEK,
        sources={
            "fpl": lambda: {
                "elements": _elements(),
                "teams": _teams(),
                "fixtures": _fixtures(),
            },
            "understat": lambda: {
                "players": pd.DataFrame(),
                "teams_history": _understat_teams_history(),
                "dates": pd.DataFrame(),
            },
            "fpl_element_summaries": lambda: {"histories": _element_summary_histories()},
            "understat_player_histories": lambda: {"histories": _understat_player_histories()},
        },
        captured_at=CAPTURED_AT,
        base_dir=base_dir,
    )
    return base_dir


def _manifest(base_dir):
    from engine.data.snapshots import SnapshotManifest

    return SnapshotManifest(
        season=SEASON, gameweek=TARGET_GAMEWEEK, captured_at=CAPTURED_AT, sources={}
    )


def test_make_build_pool_projections_end_to_end(snapshot_dir):
    build = make_build_pool_projections(
        understat_season_start_year=2025, base_dir=snapshot_dir, n_simulation_runs=50, seed=0
    )

    predictions = build(_manifest(snapshot_dir), TARGET_GAMEWEEK)

    assert set(predictions["player_id"]) == {1, 2}
    assert (predictions["gameweek"] == TARGET_GAMEWEEK).all()
    assert predictions["expected_points"].apply(lambda v: v == v).all()  # no NaN
    for col in ("floor", "ceiling", "prob_big_haul"):
        assert col in predictions.columns


@pytest.fixture
def empty_current_season_snapshot_dir(tmp_path):
    # A6: matches a real live pull against Understat's pre-season 2026/27 endpoint -- zero
    # players, zero teams before a season's first match. Every other source is populated
    # normally; only "understat"'s teams_history is empty, the actual real-world shape.
    base_dir = tmp_path / "snapshots"
    capture_snapshot(
        season=SEASON,
        gameweek=TARGET_GAMEWEEK,
        sources={
            "fpl": lambda: {"elements": _elements(), "teams": _teams(), "fixtures": _fixtures()},
            "understat": lambda: {
                "players": pd.DataFrame(),
                "teams_history": pd.DataFrame(),
                "dates": pd.DataFrame(),
            },
            "fpl_element_summaries": lambda: {"histories": _element_summary_histories()},
            "understat_player_histories": lambda: {"histories": _understat_player_histories()},
        },
        captured_at=CAPTURED_AT,
        base_dir=base_dir,
    )
    return base_dir


class _FakePriorSeasonUnderstatClient:
    """A6: duck-typed stand-in covering exactly get_league_data, for
    fetch_understat_multi_season_league_data's own use -- no network."""

    def get_league_data(self, season, league="EPL"):
        return {
            "players": [],
            "teams": {
                "1": {
                    "title": "Team A",
                    "history": [{"date": f"{season}-08-16", "xG": 1.4, "xGA": 1.0, "h_a": "h"}],
                },
                "2": {
                    "title": "Team B",
                    "history": [{"date": f"{season}-08-16", "xG": 1.1, "xGA": 1.3, "h_a": "a"}],
                },
            },
            "dates": [],
        }


def test_make_build_pool_projections_fails_hard_when_current_season_has_no_understat_data_yet(
    empty_current_season_snapshot_dir,
):
    # The real gap this test documents (and the next test's fix resolves): with no prior-season
    # fallback, an empty current-season Understat pull (real at the very start of a season, or
    # engine.data.live_adapter's own docstring) leaves team_xg_per_90 NaN for every row --
    # including the training rows, not just the target gameweek's -- so engineer_features' dropna
    # empties the training set entirely and fit_fn crashes trying to fit sklearn models on zero
    # samples, rather than degrading gracefully. This is the actual failure mode confirmed against
    # a real live pull, not a hypothetical -- see engine.data.live_adapter's own docstring.
    build = make_build_pool_projections(
        understat_season_start_year=2025,
        base_dir=empty_current_season_snapshot_dir,
        n_simulation_runs=10,
        seed=0,
    )

    with pytest.raises(ValueError, match="0 sample"):
        build(_manifest(empty_current_season_snapshot_dir), TARGET_GAMEWEEK)


def test_make_build_pool_projections_uses_prior_seasons_to_fix_the_cold_start(
    empty_current_season_snapshot_dir,
):
    build = make_build_pool_projections(
        understat_season_start_year=2025,
        base_dir=empty_current_season_snapshot_dir,
        n_simulation_runs=10,
        seed=0,
        understat_client=_FakePriorSeasonUnderstatClient(),
    )

    predictions = build(_manifest(empty_current_season_snapshot_dir), TARGET_GAMEWEEK)

    assert set(predictions["player_id"]) == {1, 2}
    assert predictions["expected_points"].apply(lambda v: v == v).all()  # no NaN


def test_make_build_pool_projections_passes_real_live_availability_to_engineer_features(
    snapshot_dir, monkeypatch
):
    # Player 1's bootstrap status/chance are "a"/None (fully fit -> 100.0), player 2's are "d"/75.0
    # -- this is a wiring check on the exact frame passed to engineer_features (whose own
    # live_availability override behavior is separately unit-tested in test_run_season.py), not an
    # assertion on the fitted model's learned sensitivity, which a 2-player/2-gameweek synthetic
    # fit can't be expected to demonstrate reliably.
    import scripts.weekly_refresh as weekly_refresh

    captured = {}
    real_engineer_features = weekly_refresh.engineer_features

    def spy(*args, **kwargs):
        captured["live_availability"] = kwargs.get("live_availability")
        return real_engineer_features(*args, **kwargs)

    monkeypatch.setattr(weekly_refresh, "engineer_features", spy)

    build = make_build_pool_projections(
        understat_season_start_year=2025, base_dir=snapshot_dir, n_simulation_runs=50, seed=0
    )
    build(_manifest(snapshot_dir), TARGET_GAMEWEEK)

    live_availability = captured["live_availability"].set_index("player_id")
    assert live_availability.loc[1, "chance_of_playing_next_round"] == pytest.approx(100.0)
    assert live_availability.loc[1, "status"] == "a"
    assert live_availability.loc[2, "chance_of_playing_next_round"] == pytest.approx(75.0)
    assert live_availability.loc[2, "status"] == "d"


def test_build_player_horizon_projections_attaches_simulation():
    predictions = pd.DataFrame(
        [
            {
                "player_id": 1,
                "position": "MID",
                "gameweek": 3,
                "p_zero": 0.1,
                "p_1_to_59": 0.2,
                "p_60_plus": 0.7,
                "expected_minutes_given_1_to_59": 30.0,
                "expected_minutes_given_60_plus": 85.0,
                "appearance": 2.0,
                "goals": 0.5,
                "assists": 0.3,
                "clean_sheet": 0.1,
                "goals_conceded": 0.0,
                "defensive_contribution": 0.2,
                "saves": 0.0,
                "bonus": 0.4,
                "cards": -0.05,
                "penalty_misses": 0.0,
                "own_goals": 0.0,
                "sim_mean": 4.0,
                "sim_median": 3.5,
                "floor": 1.0,
                "ceiling": 9.0,
                "prob_big_haul": 0.15,
            }
        ]
    )

    horizons = build_player_horizon_projections(predictions)

    assert set(horizons) == {1}
    projection = horizons[1].gameweeks[3]
    assert projection.simulation is not None
    assert projection.simulation.floor == pytest.approx(1.0)
    assert projection.simulation.ceiling == pytest.approx(9.0)
    assert projection.expected_points == pytest.approx(
        2.0 + 0.5 + 0.3 + 0.1 + 0.0 + 0.2 + 0.0 + 0.4 - 0.05 + 0.0 + 0.0
    )


def test_build_player_horizon_projections_no_simulation_when_floor_is_nan():
    predictions = pd.DataFrame(
        [
            {
                "player_id": 1,
                "position": "MID",
                "gameweek": 3,
                "p_zero": 0.1,
                "p_1_to_59": 0.2,
                "p_60_plus": 0.7,
                "expected_minutes_given_1_to_59": 30.0,
                "expected_minutes_given_60_plus": 85.0,
                "appearance": 2.0,
                "goals": 0.0,
                "assists": 0.0,
                "clean_sheet": 0.0,
                "goals_conceded": 0.0,
                "defensive_contribution": 0.0,
                "saves": 0.0,
                "bonus": 0.0,
                "cards": 0.0,
                "penalty_misses": 0.0,
                "own_goals": 0.0,
                "sim_mean": float("nan"),
                "sim_median": float("nan"),
                "floor": float("nan"),
                "ceiling": float("nan"),
                "prob_big_haul": float("nan"),
            }
        ]
    )

    horizons = build_player_horizon_projections(predictions)

    assert horizons[1].gameweeks[3].simulation is None


def test_build_app_state_from_predictions_assembles_full_state():
    predictions = pd.DataFrame(
        [
            {
                "player_id": 1,
                "position": "MID",
                "gameweek": 3,
                "p_zero": 0.1,
                "p_1_to_59": 0.2,
                "p_60_plus": 0.7,
                "expected_minutes_given_1_to_59": 30.0,
                "expected_minutes_given_60_plus": 85.0,
                "appearance": 2.0,
                "goals": 0.0,
                "assists": 0.0,
                "clean_sheet": 0.0,
                "goals_conceded": 0.0,
                "defensive_contribution": 0.0,
                "saves": 0.0,
                "bonus": 0.0,
                "cards": 0.0,
                "penalty_misses": 0.0,
                "own_goals": 0.0,
                "sim_mean": float("nan"),
                "sim_median": float("nan"),
                "floor": float("nan"),
                "ceiling": float("nan"),
                "prob_big_haul": float("nan"),
            }
        ]
    )

    state = build_app_state_from_predictions(
        predictions,
        my_team=None,
        team_id_by_player={1: TEAM_A},
        buy_prices={1: 75},
        fixtures=[],
        team_rates={},
        league_avg_xg_per_90=1.3,
        league_avg_xga_per_90=1.3,
        horizon_gameweeks=[3],
    )

    assert state.my_team is None
    assert set(state.projections) == {1}
    assert state.team_id_by_player == {1: TEAM_A}
    assert state.horizon_gameweeks == [3]


class _FakeManagerFPLClient:
    """A3: duck-typed stand-in for FPLClient covering exactly the methods
    make_build_app_state calls -- no network, no httpx.MockTransport ceremony needed since
    nothing here touches an HTTP layer at all."""

    def __init__(self, elements: pd.DataFrame, picks: dict, entry: dict, transfers, history):
        self._elements = elements
        self._picks = picks
        self._entry = entry
        self._transfers = transfers
        self._history = history

    def get_bootstrap_static(self):
        return {
            "elements": self._elements.to_dict("records"),
            "teams": [],
            "element_types": [],
            "events": [],
        }

    def get_entry(self, entry_id):
        return self._entry

    def get_entry_picks(self, entry_id, gameweek):
        return self._picks

    def get_entry_transfers(self, entry_id):
        return self._transfers

    def get_entry_history(self, entry_id):
        return self._history


def _manager_pick(element, position, is_captain=False, is_vice_captain=False):
    return {
        "element": element,
        "position": position,
        "multiplier": 2 if is_captain else 1,
        "is_captain": is_captain,
        "is_vice_captain": is_vice_captain,
    }


def test_make_build_app_state_combines_real_squad_and_real_projections():
    elements = pd.DataFrame(
        [
            {
                "id": i,
                "element_type": 3,
                "now_cost": 50 + i,
                "team": TEAM_A,
                "web_name": f"Player {i}",
            }
            for i in range(1, 16)
        ]
    )
    picks = {
        "picks": [
            _manager_pick(i, i, is_captain=(i == 1), is_vice_captain=(i == 2)) for i in range(1, 12)
        ]
        + [_manager_pick(i, i) for i in range(12, 16)]
    }
    fpl_client = _FakeManagerFPLClient(
        elements=elements,
        picks=picks,
        entry={"last_deadline_bank": 3},
        transfers=[],
        history={"current": [], "chips": []},
    )
    predictions = pd.DataFrame(
        [
            {
                "player_id": 1,
                "position": "MID",
                "gameweek": 3,
                "p_zero": 0.1,
                "p_1_to_59": 0.2,
                "p_60_plus": 0.7,
                "expected_minutes_given_1_to_59": 30.0,
                "expected_minutes_given_60_plus": 85.0,
                "appearance": 2.0,
                "goals": 0.0,
                "assists": 0.0,
                "clean_sheet": 0.0,
                "goals_conceded": 0.0,
                "defensive_contribution": 0.0,
                "saves": 0.0,
                "bonus": 0.0,
                "cards": 0.0,
                "penalty_misses": 0.0,
                "own_goals": 0.0,
                "sim_mean": float("nan"),
                "sim_median": float("nan"),
                "floor": float("nan"),
                "ceiling": float("nan"),
                "prob_big_haul": float("nan"),
            }
        ]
    )

    build = make_build_app_state(
        fpl_client=fpl_client,
        entry_id=123,
        current_gameweek=3,
        horizon_gameweeks=[3],
        fixtures=[],
        team_rates={},
        league_avg_xg_per_90=1.3,
        league_avg_xga_per_90=1.3,
    )
    state = build(predictions, CAPTURED_AT)

    assert len(state.my_team.squad) == 15
    assert state.my_team.captain_id == 1
    assert state.my_team.bank == 3
    assert state.team_id_by_player[1] == TEAM_A
    assert state.buy_prices[1] == 51
    assert state.player_names[1] == "Player 1"
    assert state.generated_at == CAPTURED_AT
    assert set(state.projections) == {1}
