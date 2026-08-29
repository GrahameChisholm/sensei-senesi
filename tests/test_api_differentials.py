"""Tests for GET /players/differentials (DIFFERENTIALS_PLAN Phase 3) -- the thin FastAPI wiring
over features.differentials/api.differentials_panel. Metric/classification correctness is already
covered at the features/ level (test_differentials.py); these tests check that the endpoint wires
the request to the right functions, resolves the window against the app's own current gameweek,
applies the ownership/squad filters, and returns the right shape.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.state as state_module
from api.mini_league_panel import reset_snapshot_cache
from api.settings import AppSettingsData
from api.squad_state import SquadState
from engine.aggregate import ComponentBreakdown
from engine.data.fpl_client import FPLClientError
from engine.data.player_history import PlayerGameweekActual
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import FWD, MID
from features.team_state import SquadPlayer


def _minutes() -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )


def _breakdown() -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=2.0,
        goals=0.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )


def _horizon(player_id: int, position: str, gameweeks=(7, 8, 9)):
    return project_player_horizon(
        player_id,
        position,
        {
            gw: project_player_gameweek(player_id, position, gw, _minutes(), _breakdown())
            for gw in gameweeks
        },
    )


def _actual(gameweek: int, **overrides) -> PlayerGameweekActual:
    base = dict(
        gameweek=gameweek,
        minutes=90,
        goals_scored=0,
        assists=0,
        clean_sheets=0,
        goals_conceded=0,
        own_goals=0,
        penalties_saved=0,
        penalties_missed=0,
        saves=0,
        yellow_cards=0,
        red_cards=0,
        bonus=0,
        defensive_contribution=0,
        total_points=4,
        expected_goals=0.0,
        expected_assists=0.0,
        expected_goal_involvements=0.0,
        expected_goals_conceded=0.0,
        starts=1,
    )
    base.update(overrides)
    return PlayerGameweekActual(**base)


def _full_squad_containing(player_id: int, position: str):
    """15 distinct players (2 GK, 5 DEF, 5 MID, 3 FWD, a legal FPL shape), with ``player_id`` at
    whichever slot matches ``position`` so it is genuinely a member of the resulting squad."""
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3
    slot = positions.index(position)
    ids = list(range(1001, 1016))
    ids[slot] = player_id
    players = tuple(
        SquadPlayer(player_id=pid, position=pos, price=50)
        for pid, pos in zip(ids, positions, strict=True)
    )
    return players


def _fixture_app_state(gameweek: int = 7):
    from api.state import AppState

    players = {
        1: {
            "web_name": "Steady",
            "team_id": 100,
            "position": MID,
            "price": 55,
            "selected_by_percent": 3.0,
        },
        2: {
            "web_name": "Popular",
            "team_id": 101,
            "position": MID,
            "price": 56,
            "selected_by_percent": 25.0,
        },
        3: {
            "web_name": "Peer",
            "team_id": 100,
            "position": MID,
            "price": 57,  # same £5.5-5.9m bucket as players 1, 2, 5, 6 (bucket width 5)
            "selected_by_percent": 8.0,
        },
        4: {
            "web_name": "NoHistory",
            "team_id": 101,
            "position": FWD,
            "price": 60,
            "selected_by_percent": 1.0,
        },
        5: {
            "web_name": "Peer2",
            "team_id": 101,
            "position": MID,
            "price": 58,
            "selected_by_percent": 5.0,
        },
        6: {
            "web_name": "Peer3",
            "team_id": 100,
            "position": MID,
            "price": 59,
            "selected_by_percent": 5.0,
        },
    }
    teams = {
        100: {"name": "Team 100", "short_name": "T100"},
        101: {"name": "Team 101", "short_name": "T101"},
    }
    projections = {1: _horizon(1, MID), 2: _horizon(2, MID), 3: _horizon(3, MID)}
    # Player 1: a clear outperformer over 6 played gameweeks (1-6). Player 2: same, but highly
    # owned. Players 3/5/6: average peers, anchoring the MID/£5.5-5.9m bracket median at 4 --
    # 3 of them so the two 9-point outliers can't drag the bucket median up themselves.
    player_history = {
        1: [_actual(gw, total_points=9, goals_scored=1) for gw in range(1, 7)],
        2: [_actual(gw, total_points=9, goals_scored=1) for gw in range(1, 7)],
        3: [_actual(gw, total_points=4) for gw in range(1, 7)],
        5: [_actual(gw, total_points=4) for gw in range(1, 7)],
        6: [_actual(gw, total_points=4) for gw in range(1, 7)],
    }

    return AppState(
        season="2026-27",
        gameweek=gameweek,
        horizon_gameweeks=[7, 8, 9],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        model_version="test",
        projections=projections,
        players=players,
        teams=teams,
        fixtures=[],
        diagnostics={},
        player_history=player_history,
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    state_module.set_app_state(_fixture_app_state())
    reset_snapshot_cache()
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_snapshot_cache()
    state_module.reset_state()


def test_resolves_window_from_the_app_states_current_gameweek(client: TestClient):
    # gameweek=7 -> latest played = 6; default window=6 -> gameweek_from=1, gameweek_to=6.
    response = client.get("/players/differentials")

    assert response.status_code == 200
    window = response.json()["window"]
    assert window == {"gameweek_from": 1, "gameweek_to": 6, "requested_gameweeks": 6}


def test_returns_the_outperforming_player_with_a_positive_surplus(client: TestClient):
    response = client.get("/players/differentials")

    rows = {row["player_id"]: row for row in response.json()["rows"]}
    assert 1 in rows
    assert rows[1]["surplus_vs_bracket"] > 0
    assert rows[1]["name"] == "Steady"


def test_max_ownership_filters_out_the_highly_owned_player(client: TestClient):
    response = client.get("/players/differentials", params={"max_ownership": 10.0})

    player_ids = {row["player_id"] for row in response.json()["rows"]}
    assert 1 in player_ids
    assert 2 not in player_ids  # 25% owned, filtered out


def test_without_max_ownership_the_highly_owned_player_is_still_included(client: TestClient):
    response = client.get("/players/differentials")

    player_ids = {row["player_id"] for row in response.json()["rows"]}
    assert 2 in player_ids


def test_player_with_no_history_in_window_is_absent(client: TestClient):
    response = client.get("/players/differentials")

    player_ids = {row["player_id"] for row in response.json()["rows"]}
    assert 4 not in player_ids


def test_includes_fixture_cells_for_the_full_horizon(client: TestClient):
    response = client.get("/players/differentials")

    row = next(r for r in response.json()["rows"] if r["player_id"] == 1)
    assert [cell["gameweek"] for cell in row["fixtures"]] == [7, 8, 9]


def test_requested_window_larger_than_played_gameweeks_clamps(client: TestClient):
    response = client.get("/players/differentials", params={"window": 20})

    window = response.json()["window"]
    assert window["gameweek_from"] == 1
    assert window["gameweek_to"] == 6
    assert window["requested_gameweeks"] == 20


def test_hide_owned_excludes_a_player_in_the_squad(client: TestClient):
    squad = _full_squad_containing(1, MID)
    squad_ids = [p.player_id for p in squad]
    state = SquadState(
        squad=squad,
        starting_xi=tuple(squad_ids[:11]),
        bench_order=tuple(squad_ids[11:]),
        captain_id=squad_ids[0],
        vice_captain_id=squad_ids[1],
    )
    state_module.set_squad_state(state)

    response = client.get("/players/differentials")

    player_ids = {row["player_id"] for row in response.json()["rows"]}
    assert 1 not in player_ids


def test_hide_owned_false_keeps_the_owned_player(client: TestClient):
    squad = _full_squad_containing(1, MID)
    squad_ids = [p.player_id for p in squad]
    state = SquadState(
        squad=squad,
        starting_xi=tuple(squad_ids[:11]),
        bench_order=tuple(squad_ids[11:]),
        captain_id=squad_ids[0],
        vice_captain_id=squad_ids[1],
    )
    state_module.set_squad_state(state)

    response = client.get("/players/differentials", params={"hide_owned": False})

    player_ids = {row["player_id"] for row in response.json()["rows"]}
    assert 1 in player_ids


def test_no_squad_yet_does_not_error_with_hide_owned(client: TestClient):
    response = client.get("/players/differentials", params={"hide_owned": True})

    assert response.status_code == 200


def test_defaults_to_the_global_lens_when_no_league_is_configured(client: TestClient):
    """MINI_LEAGUE_PLAN M24: the fallback chain's first link -- no fpl_team_id/mini_league_ids
    saved at all, so Differentials must keep working exactly as it always has."""
    response = client.get("/players/differentials")

    assert response.status_code == 200
    body = response.json()
    assert body["ownership_lens"] == "global"
    assert body["picks_gameweek"] is None
    assert body["n_rivals"] is None
    row = next(r for r in body["rows"] if r["player_id"] == 1)
    assert row["league_owner_count"] is None


MY_ENTRY_ID = 555


def _league_standings(rival_picks: dict[int, dict[int, int]]) -> dict:
    """One classic-league standings page for entries ``MY_ENTRY_ID`` plus each key of
    ``rival_picks`` (rival entry_id -> {player_id: multiplier})."""
    results = [
        {"entry": MY_ENTRY_ID, "player_name": "Me", "entry_name": "My Team", "rank": 1, "total": 0}
    ]
    for rival_id in rival_picks:
        results.append(
            {
                "entry": rival_id,
                "player_name": f"Rival {rival_id}",
                "entry_name": f"Team {rival_id}",
                "rank": len(results) + 1,
                "total": 0,
            }
        )
    return {
        "league": {"id": 999, "name": "Test League"},
        "standings": {"has_next": False, "results": results},
    }


class _StubFPLClient:
    def __init__(self, rival_picks: dict[int, dict[int, int]] | None = None, error=None):
        self._rival_picks = rival_picks if rival_picks is not None else {}
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_league_standings(self, league_id, page=1):
        if self._error is not None:
            raise self._error
        return _league_standings(self._rival_picks)

    def get_entry(self, entry_id):
        return {"id": entry_id, "current_event": 6}

    def get_entry_picks(self, entry_id, gameweek):
        picks = self._rival_picks.get(entry_id, {})
        return {
            "picks": [
                {"element": pid, "position": i + 1, "multiplier": m}
                for i, (pid, m) in enumerate(picks.items())
            ]
        }

    def get_entry_history(self, entry_id):
        return {"chips": []}


class TestLeagueOwnershipLens:
    """MINI_LEAGUE_PLAN M24/M28: once a league is configured and its live fetch succeeds, the
    Differentials page switches to that league's own effective ownership."""

    def _configure_league(self, monkeypatch, rival_picks: dict[int, dict[int, int]]):
        state_module.set_app_settings(
            AppSettingsData(fpl_team_id=MY_ENTRY_ID, mini_league_ids=(999,))
        )
        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient(rival_picks))

    def test_switches_to_the_league_lens_once_configured(self, client: TestClient, monkeypatch):
        self._configure_league(monkeypatch, {1: {1: 1}, 2: {1: 1}, 3: {1: 1}})

        response = client.get("/players/differentials")

        assert response.status_code == 200
        body = response.json()
        assert body["ownership_lens"] == "league"
        assert body["picks_gameweek"] == 6
        assert body["n_rivals"] == 3

    def test_league_owner_count_reflects_the_snapshot_not_global_percent(
        self, client: TestClient, monkeypatch
    ):
        # Player 1 is 3% owned globally (would pass any sane max_ownership); under the league lens
        # every one of the three rivals owns him.
        self._configure_league(monkeypatch, {1: {1: 1}, 2: {1: 1}, 3: {1: 1}})

        response = client.get("/players/differentials")

        row = next(r for r in response.json()["rows"] if r["player_id"] == 1)
        assert row["league_owner_count"] == 3
        assert row["league_eo_multiplier"] == pytest.approx(1.0)

    def test_max_league_owners_filters_players_the_global_percentage_would_have_kept(
        self, client: TestClient, monkeypatch
    ):
        # Player 2 is 25% owned globally -- already excluded by max_ownership tests above -- but
        # here the real question is whether *league* ownership drives the filter instead.
        self._configure_league(monkeypatch, {1: {2: 1}, 2: {2: 1}, 3: {2: 1}})

        response = client.get(
            "/players/differentials", params={"max_league_owners": 0, "max_ownership": 100.0}
        )

        player_ids = {row["player_id"] for row in response.json()["rows"]}
        assert 2 not in player_ids  # owned by all 3 rivals under the league lens
        assert 1 in player_ids  # owned by none

    def test_falls_back_to_global_when_the_live_fetch_fails(self, client: TestClient, monkeypatch):
        state_module.set_app_settings(
            AppSettingsData(fpl_team_id=MY_ENTRY_ID, mini_league_ids=(999,))
        )
        monkeypatch.setattr(
            api_main, "FPLClient", lambda: _StubFPLClient(error=FPLClientError("boom"))
        )

        response = client.get("/players/differentials")

        assert response.status_code == 200
        body = response.json()
        assert body["ownership_lens"] == "global"
        assert body["picks_gameweek"] is None

    def test_falls_back_to_global_when_the_league_has_no_other_members(
        self, client: TestClient, monkeypatch
    ):
        self._configure_league(monkeypatch, {})  # only "me" in the league

        response = client.get("/players/differentials")

        assert response.json()["ownership_lens"] == "global"

    def test_league_id_query_param_overrides_the_saved_default(
        self, client: TestClient, monkeypatch
    ):
        state_module.set_app_settings(
            AppSettingsData(fpl_team_id=MY_ENTRY_ID, mini_league_ids=(999,))
        )
        seen_league_ids = []

        class _RecordingStub(_StubFPLClient):
            def get_league_standings(self, league_id, page=1):
                seen_league_ids.append(league_id)
                return super().get_league_standings(league_id, page)

        monkeypatch.setattr(api_main, "FPLClient", lambda: _RecordingStub({1: {1: 1}}))

        client.get("/players/differentials", params={"league_id": 424242})

        assert seen_league_ids == [424242]
