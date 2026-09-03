"""Tests for GET /squad/transfers and POST /squad/transfers/apply (TRANSFER_BANNER) -- the thin
FastAPI wiring over api.transfer_panel plus a live (here, stubbed) league fetch. The ranking math
itself is covered at the features level (test_transfer_planner.py); these tests check the plumbing:
the squad guard, that a missing or failing league degrades to a working points-only suggestion
rather than an error, the response shape, the solve cache, and that applying a plan lands a legal
squad.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.state as state_module
import api.transfer_panel as transfer_panel
from api.mini_league_panel import reset_snapshot_cache
from api.state import AppState
from api.transfer_panel import reset_suggestion_cache
from engine.aggregate import ComponentBreakdown
from engine.data.fpl_client import FPLClientError
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID
from tests.conftest import UPCOMING_DEADLINE

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]
ALL_IDS = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]

# Deliberately better than anything in the starting 15 (4.0 each), one per outfield position, so
# there is always a real upgrade for the solver to find and a suggestion is never empty by
# accident. Same price as everyone else, so budget never blocks the swap.
UPGRADE_IDS = {9001: FWD, 9002: MID, 9003: DEF}
UPGRADE_POINTS = 9.0
BASE_POINTS = 4.0
PRICE = 40

MY_ENTRY_ID = 555
LEAGUE_ID = 999


def _position_for(player_id: int) -> str:
    if player_id in UPGRADE_IDS:
        return UPGRADE_IDS[player_id]
    if player_id in (GK1, GK2):
        return GK
    if player_id in DEF_IDS:
        return DEF
    if player_id in MID_IDS:
        return MID
    return FWD


def _horizon(player_id: int, points: float, gameweeks=(1, 2, 3)):
    position = _position_for(player_id)
    minutes = MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )
    breakdown = ComponentBreakdown(
        appearance=2.0,
        goals=points - 2.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )
    return project_player_horizon(
        player_id,
        position,
        {
            gw: project_player_gameweek(player_id, position, gw, minutes, breakdown)
            for gw in gameweeks
        },
    )


def _fixture_app_state() -> AppState:
    # The 15 spread across 5 clubs (3 each) so MAX_PER_CLUB is never accidentally violated, with
    # the upgrade targets on their own separate club so bringing any of them in is always legal.
    players = {}
    teams = {}
    for index, player_id in enumerate(ALL_IDS):
        team_id = 100 + (index // 3)
        teams.setdefault(team_id, {"name": f"Team {team_id}", "short_name": f"T{team_id}"})
        players[player_id] = {
            "web_name": f"Player{player_id}",
            "team_id": team_id,
            "position": _position_for(player_id),
            "price": PRICE,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
        }
    extra_team_id = max(teams) + 1
    teams[extra_team_id] = {"name": "Team Extra", "short_name": "TEX"}
    for player_id in UPGRADE_IDS:
        players[player_id] = {
            "web_name": f"Player{player_id}",
            "team_id": extra_team_id,
            "position": _position_for(player_id),
            "price": PRICE,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
        }

    projections = {pid: _horizon(pid, BASE_POINTS) for pid in ALL_IDS}
    projections.update({pid: _horizon(pid, UPGRADE_POINTS) for pid in UPGRADE_IDS})

    return AppState(
        season="2026-27",
        gameweek=1,
        horizon_gameweeks=[1, 2, 3],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=UPCOMING_DEADLINE,
        model_version="test",
        projections=projections,
        players=players,
        teams=teams,
        fixtures=[],
        diagnostics={},
    )


@pytest.fixture()
def client(tmp_path):
    state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
    state_module.set_app_state(_fixture_app_state())
    reset_snapshot_cache()
    reset_suggestion_cache()
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    reset_snapshot_cache()
    reset_suggestion_cache()
    state_module.reset_state()


def _build_full_squad(client: TestClient) -> None:
    for player_id in ALL_IDS:
        response = client.post(
            "/squad/players",
            json={"player_id": player_id, "position": _position_for(player_id), "price": PRICE},
        )
        assert response.status_code == 200, response.json()


_STANDINGS = [
    {
        "entry": MY_ENTRY_ID,
        "player_name": "Me",
        "entry_name": "My Team",
        "rank": 2,
        "total": 90,
        "event_total": 0,
    },
    {
        "entry": 1,
        "player_name": "Dave",
        "entry_name": "Dave's Team",
        "rank": 1,
        "total": 120,
        "event_total": 0,
    },
]


class _StubFPLClient:
    """Stands in for the live league fetch. Dave owns the upgrade targets, so bringing one in is a
    template cover and leaving him out is a differential, which is what gives the variance half of
    the ranking something to actually move."""

    def __init__(self, error=None, on_call=None, owns_upgrades=True):
        self._error = error
        self._on_call = on_call
        self._owns_upgrades = owns_upgrades

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_league_standings(self, league_id, page=1):
        if self._on_call is not None:
            self._on_call(page)
        if self._error is not None:
            raise self._error
        return {
            "league": {"id": league_id, "name": "Test League"},
            "standings": {"has_next": False, "results": _STANDINGS},
        }

    def get_entry(self, entry_id):
        return {"id": entry_id, "current_event": 1}

    def get_entry_picks(self, entry_id, gameweek):
        picks = [{"element": pid, "multiplier": 1} for pid in ALL_IDS[:12]]
        if self._owns_upgrades:
            picks += [{"element": pid, "multiplier": 1} for pid in UPGRADE_IDS]
        return {"picks": picks}

    def get_entry_history(self, entry_id):
        return {"chips": []}


def _configure_league(client: TestClient) -> None:
    response = client.post(
        "/mini-league/leagues",
        json={"fpl_team_id": MY_ENTRY_ID, "mini_league_ids": [LEAGUE_ID]},
    )
    assert response.status_code == 200, response.json()


class TestSuggestTransfers:
    def test_requires_a_complete_squad(self, client):
        assert client.get("/squad/transfers").status_code == 400

    def test_suggests_an_upgrade_with_no_league_configured(self, client):
        """The Team page has always worked without a mini-league, so the banner must too: no
        league means no rivals, a null league_id, and a ranking that falls back to expected
        points."""
        _build_full_squad(client)
        body = client.get("/squad/transfers").json()

        assert body["league_id"] is None
        assert body["n_rivals"] == 0
        assert body["variance_preference"] == "neutral"
        assert body["plans"]
        best = body["plans"][0]
        assert best["n_transfers"] == 1
        assert best["in_player_ids"][0] in UPGRADE_IDS
        assert best["expected_points_delta"] > 0

    def test_a_failing_league_fetch_still_returns_a_suggestion(self, client, monkeypatch):
        """A transient FPL problem must degrade the banner to its points-only form, not 400 it.
        This is the opposite of /mini-league/{id}'s own contract, deliberately: that page is about
        the league, this banner merely uses it."""
        _build_full_squad(client)
        _configure_league(client)
        monkeypatch.setattr(
            transfer_panel, "FPLClient", lambda: _StubFPLClient(error=FPLClientError("boom"))
        )

        response = client.get("/squad/transfers")
        assert response.status_code == 200, response.json()
        assert response.json()["league_id"] is None
        assert response.json()["plans"]

    def test_returns_league_scored_plans(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_league(client)
        monkeypatch.setattr(transfer_panel, "FPLClient", lambda: _StubFPLClient())

        body = client.get("/squad/transfers", params={"transfers": 2}).json()
        assert body["league_id"] == LEAGUE_ID
        assert body["league_name"] == "Test League"
        assert body["n_rivals"] == 1
        assert body["max_transfers"] == 2
        assert body["plans"]
        for plan in body["plans"]:
            assert plan["n_transfers"] <= 2
            assert len(plan["moves"]) == plan["n_transfers"]
            assert 1.0 <= plan["expected_final_rank"] <= 2.0

    def test_moves_carry_names_and_ownership(self, client, monkeypatch):
        _build_full_squad(client)
        _configure_league(client)
        monkeypatch.setattr(transfer_panel, "FPLClient", lambda: _StubFPLClient())

        move = client.get("/squad/transfers").json()["plans"][0]["moves"][0]
        assert move["in_name"] == f"Player{move['in_player_id']}"
        assert move["out_name"] == f"Player{move['out_player_id']}"
        assert move["position"] == _position_for(move["in_player_id"])
        assert move["price_delta"] == 0
        assert move["in_eo_multiplier"] == pytest.approx(1.0)

    def test_an_unowned_incoming_player_reads_as_zero_not_unknown(self, client, monkeypatch):
        """Nobody in the league owning a player is the strongest differential signal there is, so
        it must arrive as 0.0. Reporting it as null would be indistinguishable from having no
        league at all, and the banner would drop the tag exactly where it matters most."""
        _build_full_squad(client)
        _configure_league(client)
        monkeypatch.setattr(
            transfer_panel, "FPLClient", lambda: _StubFPLClient(owns_upgrades=False)
        )

        move = client.get("/squad/transfers").json()["plans"][0]["moves"][0]
        assert move["in_player_id"] in UPGRADE_IDS
        assert move["in_eo_multiplier"] == 0.0

    def test_ownership_is_null_only_when_there_is_no_league(self, client):
        _build_full_squad(client)
        move = client.get("/squad/transfers").json()["plans"][0]["moves"][0]
        assert move["in_eo_multiplier"] is None

    def test_marginal_gains_line_up_with_best_by_transfer_count(self, client):
        _build_full_squad(client)
        body = client.get("/squad/transfers", params={"transfers": 3}).json()

        assert len(body["best_by_transfer_count"]) == len(body["marginal_points_gains"])
        running = 0.0
        for plan, gain in zip(
            body["best_by_transfer_count"], body["marginal_points_gains"], strict=True
        ):
            running += gain
            assert plan["expected_points_delta"] == pytest.approx(running)

    def test_rejects_a_transfer_count_outside_the_allowed_range(self, client):
        _build_full_squad(client)
        assert client.get("/squad/transfers", params={"transfers": 0}).status_code == 400
        assert client.get("/squad/transfers", params={"transfers": 9}).status_code == 400

    def test_horizon_widens_the_points_gain(self, client):
        """Three gameweeks of the same upgrade is three times one gameweek of it, which is the
        cheapest available check that the horizon argument reaches the planner at all rather than
        being silently ignored."""
        _build_full_squad(client)
        one = client.get("/squad/transfers").json()
        three = client.get("/squad/transfers", params={"horizon": 3}).json()

        assert one["gameweeks"] == [1]
        assert three["gameweeks"] == [1, 2, 3]
        assert three["plans"][0]["expected_points_delta"] == pytest.approx(
            3 * one["plans"][0]["expected_points_delta"]
        )

    def test_second_identical_request_does_not_refetch_or_resolve(self, client, monkeypatch):
        calls = []
        _build_full_squad(client)
        _configure_league(client)
        monkeypatch.setattr(
            transfer_panel, "FPLClient", lambda: _StubFPLClient(on_call=calls.append)
        )

        first = client.get("/squad/transfers").json()
        second = client.get("/squad/transfers").json()
        assert first == second
        # One league page fetch, since the second suggestion came from the solve cache and the
        # snapshot behind it from the shared TTL cache.
        assert len(calls) == 1

    def test_editing_the_squad_invalidates_the_cached_suggestion(self, client):
        _build_full_squad(client)
        first = client.get("/squad/transfers").json()

        target = first["plans"][0]
        client.post(
            "/squad/transfers/apply",
            json={
                "out_player_ids": target["out_player_ids"],
                "in_player_ids": target["in_player_ids"],
            },
        )
        second = client.get("/squad/transfers").json()
        assert second["plans"] != first["plans"]


class TestApplyTransfers:
    def test_applies_a_suggested_plan(self, client):
        _build_full_squad(client)
        plan = client.get("/squad/transfers", params={"transfers": 2}).json()["plans"][0]

        response = client.post(
            "/squad/transfers/apply",
            json={
                "out_player_ids": plan["out_player_ids"],
                "in_player_ids": plan["in_player_ids"],
            },
        )
        assert response.status_code == 200, response.json()
        squad = response.json()
        assert squad["is_complete"]
        ids = {player["player_id"] for player in squad["squad"]}
        assert set(plan["in_player_ids"]) <= ids
        assert not set(plan["out_player_ids"]) & ids

    def test_rejects_a_player_not_in_the_squad(self, client):
        _build_full_squad(client)
        response = client.post(
            "/squad/transfers/apply",
            json={"out_player_ids": [123456], "in_player_ids": [9001]},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "unknown_player"

    def test_rejects_buying_a_player_already_owned(self, client):
        _build_full_squad(client)
        response = client.post(
            "/squad/transfers/apply",
            json={"out_player_ids": [MID_IDS[0]], "in_player_ids": [MID_IDS[1]]},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "duplicate"

    def test_rejects_mismatched_lengths(self, client):
        _build_full_squad(client)
        response = client.post(
            "/squad/transfers/apply",
            json={"out_player_ids": [MID_IDS[0]], "in_player_ids": [9002, 9003]},
        )
        assert response.status_code == 400

    def test_rejects_a_move_that_breaks_the_position_quota(self, client):
        """Validating the finished 15 rather than each swap in turn means an illegal destination
        is still caught, even though the route to it is never walked."""
        _build_full_squad(client)
        response = client.post(
            "/squad/transfers/apply",
            json={"out_player_ids": [MID_IDS[0]], "in_player_ids": [9001]},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "quota"

    def test_requires_a_complete_squad(self, client):
        response = client.post(
            "/squad/transfers/apply",
            json={"out_player_ids": [MID_IDS[0]], "in_player_ids": [9002]},
        )
        assert response.status_code == 400
