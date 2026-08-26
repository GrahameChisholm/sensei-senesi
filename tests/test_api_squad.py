"""Tests for api.main -- the thin FastAPI wiring over features.squad_rules/squad_optimizer/
squad_points for the one permanently-live sandbox squad. Business-rule correctness (legality,
the ILP solver) is already exhaustively covered at the features/ level (test_squad_rules.py,
test_squad_optimizer.py); these tests check that the API wires requests to the right function,
returns the right shape, and turns a SquadRuleError/SquadOptimizerError into the right status code
-- not the rules themselves.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.state as state_module
from api.squad_state import SquadState
from engine.aggregate import ComponentBreakdown
from engine.data.fpl_client import FPLClientError
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID
from features.team_state import SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]
ALL_IDS = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]


def _position_for(player_id: int) -> str:
    if player_id in (GK1, GK2):
        return GK
    if player_id in DEF_IDS:
        return DEF
    if player_id in MID_IDS:
        return MID
    return FWD


def _horizon(player_id: int, points: float = 4.0, gameweeks=(1, 2, 3)):
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


def _fixture_app_state():
    from api.state import AppState

    # Spread the 15 squad players across 5 clubs (3 each) so MAX_PER_CLUB is never accidentally
    # violated -- team_id = 100 + (index // 3).
    players = {}
    teams = {}
    for index, player_id in enumerate(ALL_IDS):
        team_id = 100 + (index // 3)
        teams.setdefault(team_id, {"name": f"Team {team_id}", "short_name": f"T{team_id}"})
        players[player_id] = {
            "web_name": f"Player{player_id}",
            "full_name": f"Player {player_id}",
            "team_id": team_id,
            "position": _position_for(player_id),
            "price": 40,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
        }
    # A couple of extra transfer-target players not in the initial squad, on their own new club.
    extra_team_id = max(teams) + 1
    teams[extra_team_id] = {"name": "Team Extra", "short_name": "TEX"}
    for extra_id, position in ((9001, FWD), (9002, MID), (9003, DEF)):
        players[extra_id] = {
            "web_name": f"Player{extra_id}",
            "full_name": f"Player {extra_id}",
            "team_id": extra_team_id,
            "position": position,
            "price": 40,
            "status": "a",
            "chance_of_playing_next_round": 100.0,
            "low_confidence": False,
            "source": "engine",
        }

    projections = {pid: _horizon(pid) for pid in [*ALL_IDS, 9001, 9002, 9003]}

    return AppState(
        season="2026-27",
        gameweek=1,
        horizon_gameweeks=[1, 2, 3],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
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
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client
    state_module.reset_state()


def _build_full_squad(client: TestClient) -> dict:
    body = None
    for player_id in ALL_IDS:
        response = client.post(
            "/squad/players",
            json={"player_id": player_id, "position": _position_for(player_id), "price": 40},
        )
        assert response.status_code == 200, response.json()
        body = response.json()
    return body


_ELEMENT_TYPE_BY_POSITION = {GK: 1, DEF: 2, MID: 3, FWD: 4}


def _fpl_picks_payload() -> dict:
    starting_ids = [GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]]
    bench_ids = [DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2]
    picks = [
        {
            "element": pid,
            "position": i,
            "multiplier": 2 if pid == MID_IDS[0] else 1,
            "is_captain": pid == MID_IDS[0],
            "is_vice_captain": pid == MID_IDS[1],
        }
        for i, pid in enumerate(starting_ids, start=1)
    ]
    picks += [
        {
            "element": pid,
            "position": i,
            "multiplier": 1,
            "is_captain": False,
            "is_vice_captain": False,
        }
        for i, pid in enumerate(bench_ids, start=12)
    ]
    return {"active_chip": None, "picks": picks}


def _fpl_elements_payload(now_cost: int = 45) -> list[dict]:
    return [
        {
            "id": pid,
            "now_cost": now_cost,
            "element_type": _ELEMENT_TYPE_BY_POSITION[_position_for(pid)],
        }
        for pid in ALL_IDS
    ]


class TestHealthAndGameweek:
    def test_health(self, client):
        assert client.get("/health").json() == {"status": "ok"}

    def test_gameweek(self, client):
        body = client.get("/gameweek").json()
        assert body["season"] == "2026-27"
        assert body["gameweek"] == 1
        assert body["deadline_passed"] is False

    def test_teams(self, client):
        body = client.get("/teams").json()
        assert len(body) > 0
        assert {"team_id", "name", "short_name"} <= set(body[0])


class TestSquadMutations:
    """The sandbox squad: every mutation applies instantly, no confirm step anywhere."""

    def test_empty_squad_state(self, client):
        body = client.get("/squad").json()
        assert body["is_complete"] is False
        assert body["squad"] == []
        assert body["budget_ceiling"] == 1000

    def test_adding_a_player(self, client):
        response = client.post(
            "/squad/players", json={"player_id": GK1, "position": GK, "price": 40}
        )
        body = response.json()
        assert len(body["squad"]) == 1
        assert body["squad"][0]["player_id"] == GK1
        assert body["is_complete"] is False

    def test_adding_duplicate_player_is_rejected(self, client):
        client.post("/squad/players", json={"player_id": GK1, "position": GK, "price": 40})
        response = client.post(
            "/squad/players", json={"player_id": GK1, "position": GK, "price": 40}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "duplicate"

    def test_adding_beyond_budget_is_rejected(self, client):
        response = client.post(
            "/squad/players", json={"player_id": GK1, "position": GK, "price": 1500}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "budget"

    def test_removing_a_player(self, client):
        client.post("/squad/players", json={"player_id": GK1, "position": GK, "price": 40})
        response = client.delete(f"/squad/players/{GK1}")
        assert response.json()["squad"] == []

    def test_removing_unknown_player_is_rejected(self, client):
        response = client.delete(f"/squad/players/{GK1}")
        assert response.status_code == 400

    def test_squad_becomes_complete_at_fifteen_with_auto_derived_arrangement(self, client):
        body = _build_full_squad(client)
        assert body["is_complete"] is True
        assert len(body["squad"]) == 15
        assert len(body["starting_xi"]) == 11
        assert len(body["bench_order"]) == 4
        assert body["captain_id"] is not None
        assert body["vice_captain_id"] is not None

    def test_removing_a_player_from_a_full_squad_drops_back_to_incomplete(self, client):
        _build_full_squad(client)
        response = client.delete(f"/squad/players/{FWD_IDS[2]}")
        body = response.json()
        assert body["is_complete"] is False
        assert len(body["squad"]) == 14
        assert body["starting_xi"] == []
        assert body["captain_id"] is None

    def test_removing_then_readding_regains_complete_state(self, client):
        _build_full_squad(client)
        client.delete(f"/squad/players/{FWD_IDS[2]}")
        response = client.post(
            "/squad/players", json={"player_id": 9001, "position": FWD, "price": 40}
        )
        body = response.json()
        assert body["is_complete"] is True
        assert len(body["squad"]) == 15
        assert 9001 in {p["player_id"] for p in body["squad"]}

    def test_no_transfer_economy_fields_anywhere(self, client):
        _build_full_squad(client)
        client.delete(f"/squad/players/{FWD_IDS[2]}")
        response = client.post(
            "/squad/players", json={"player_id": 9001, "position": FWD, "price": 999}
        )
        assert response.status_code == 400  # over budget, since 999 alone exceeds the ceiling
        assert "free_transfers" not in client.get("/squad").json()
        assert "bank" not in client.get("/squad").json()

    def test_clear_squad_empties_and_resets_budget_ceiling(self, client):
        _build_full_squad(client)
        response = client.delete("/squad/players")
        body = response.json()
        assert body["squad"] == []
        assert body["is_complete"] is False
        assert body["budget_ceiling"] == 1000

    def test_clearing_an_already_empty_squad_is_a_harmless_no_op(self, client):
        response = client.delete("/squad/players")
        assert response.status_code == 200
        assert response.json()["squad"] == []

    def test_can_rebuild_a_fresh_squad_after_clearing(self, client):
        _build_full_squad(client)
        client.delete("/squad/players")
        body = _build_full_squad(client)
        assert body["is_complete"] is True
        assert len(body["squad"]) == 15

    def test_set_captain(self, client):
        _build_full_squad(client)
        xi = client.get("/squad").json()["starting_xi"]
        response = client.post("/squad/captain", json={"player_id": xi[0], "role": "captain"})
        assert response.json()["captain_id"] == xi[0]

    def test_set_vice_captain(self, client):
        _build_full_squad(client)
        xi = client.get("/squad").json()["starting_xi"]
        response = client.post("/squad/captain", json={"player_id": xi[1], "role": "vice"})
        assert response.json()["vice_captain_id"] == xi[1]

    def test_set_captain_before_complete_squad_is_rejected(self, client):
        response = client.post("/squad/captain", json={"player_id": GK1, "role": "captain"})
        assert response.status_code == 400

    def test_invalid_role_rejected(self, client):
        _build_full_squad(client)
        xi = client.get("/squad").json()["starting_xi"]
        response = client.post("/squad/captain", json={"player_id": xi[0], "role": "sideways"})
        assert response.status_code == 400

    def test_bench_order_sets_a_new_xi_bench_partition(self, client):
        _build_full_squad(client)
        squad = client.get("/squad").json()
        untouchable = {squad["captain_id"], squad["vice_captain_id"]}
        position_by_id = {pid: _position_for(pid) for pid in ALL_IDS}
        starting_out = next(
            pid
            for pid in squad["starting_xi"]
            if pid not in untouchable
            and any(
                bpid not in untouchable and position_by_id[bpid] == position_by_id[pid]
                for bpid in squad["bench_order"]
            )
        )
        bench_in = next(
            bpid
            for bpid in squad["bench_order"]
            if bpid not in untouchable and position_by_id[bpid] == position_by_id[starting_out]
        )
        new_xi = [bench_in if pid == starting_out else pid for pid in squad["starting_xi"]]
        new_bench = [starting_out if pid == bench_in else pid for pid in squad["bench_order"]]
        response = client.post(
            "/squad/bench-order", json={"starting_xi": new_xi, "bench_order": new_bench}
        )
        body = response.json()
        assert bench_in in body["starting_xi"]
        assert starting_out in body["bench_order"]

    def test_bench_order_illegal_partition_is_rejected(self, client):
        _build_full_squad(client)
        squad = client.get("/squad").json()
        response = client.post(
            "/squad/bench-order",
            json={"starting_xi": squad["starting_xi"][:-1], "bench_order": squad["bench_order"]},
        )
        assert response.status_code == 400


class TestImportSquad:
    """POST /squad/import -- fetches a real manager's current picks live via FPLClient
    (monkeypatched here) and promotes them verbatim, keeping only player/position/price."""

    def _stub_client(
        self, monkeypatch, *, entry=None, error=None, now_cost=45, picks_gameweeks_seen=None
    ):
        entry = entry if entry is not None else {"current_event": 1}
        picks = _fpl_picks_payload()
        elements = _fpl_elements_payload(now_cost)

        class _StubFPLClient:
            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

            def get_entry(self, entry_id):
                if error is not None:
                    raise error
                return entry

            def get_entry_picks(self, entry_id, gameweek):
                if picks_gameweeks_seen is not None:
                    picks_gameweeks_seen.append(gameweek)
                return picks

            def get_bootstrap_static(self):
                return {"elements": elements}

        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())

    def test_imports_a_real_squad(self, client, monkeypatch):
        self._stub_client(monkeypatch)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200, response.json()
        body = response.json()
        assert body["is_complete"] is True
        assert len(body["squad"]) == 15

    def test_keeps_the_managers_real_captain_and_xi_verbatim(self, client, monkeypatch):
        self._stub_client(monkeypatch)
        response = client.post("/squad/import", json={"team_id": 123456})
        body = response.json()
        assert body["captain_id"] == MID_IDS[0]
        assert body["vice_captain_id"] == MID_IDS[1]
        assert set(body["starting_xi"]) == {GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]}

    def test_uses_current_price_for_every_player(self, client, monkeypatch):
        self._stub_client(monkeypatch, now_cost=45)
        response = client.post("/squad/import", json={"team_id": 123456})
        body = response.json()
        assert all(p["price"] == 45 for p in body["squad"])

    def test_squad_over_100m_via_price_rises_sets_a_higher_budget_ceiling(
        self, client, monkeypatch
    ):
        # Real squads legitimately drift above £100m of nominal spend as prices rise -- import
        # must not reject that, and the personal ceiling should reflect this squad's real value.
        self._stub_client(monkeypatch, now_cost=100)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200, response.json()
        assert response.json()["budget_ceiling"] == 1500

    def test_squad_under_100m_keeps_the_classic_ceiling(self, client, monkeypatch):
        self._stub_client(monkeypatch, now_cost=45)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.json()["budget_ceiling"] == 1000

    def test_overwrites_an_existing_squad(self, client, monkeypatch):
        _build_full_squad(client)
        self._stub_client(monkeypatch)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200
        assert len(response.json()["squad"]) == 15

    def test_unreachable_team_id_surfaces_as_a_400(self, client, monkeypatch):
        self._stub_client(monkeypatch, error=FPLClientError("team not found"))
        response = client.post("/squad/import", json={"team_id": 999999})
        assert response.status_code == 400
        body = response.json()
        assert "team not found" in body["message"]

    def test_non_positive_team_id_is_rejected(self, client):
        response = client.post("/squad/import", json={"team_id": 0})
        assert response.status_code == 422

    def test_fetches_picks_for_entrys_current_event_not_app_states_gameweek(
        self, client, monkeypatch
    ):
        # A real manager's picks/{gw} 404s until that gameweek's deadline or a saved transfer --
        # entry["current_event"] is the gameweek that actually has a picks record, which can lag
        # behind app_state.gameweek (the app's own target gameweek, currently 1 in this fixture).
        seen = []
        self._stub_client(monkeypatch, entry={"current_event": 7}, picks_gameweeks_seen=seen)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200, response.json()
        assert seen == [7]


class TestOptimiseLineup:
    def test_applies_immediately(self, client):
        _build_full_squad(client)
        response = client.post("/squad/optimise-xi")
        assert response.status_code == 200
        assert len(response.json()["starting_xi"]) == 11

    def test_before_complete_squad_is_rejected(self, client):
        response = client.post("/squad/optimise-xi")
        assert response.status_code == 400


class TestAutoBuild:
    """POST /squad/optimise -- the best-possible-squad ILP solver, always locking whatever's
    currently picked."""

    def test_full_rebuild_from_empty(self, client):
        response = client.post("/squad/optimise", json={})
        assert response.status_code == 200, response.json()
        body = response.json()
        assert len(body["squad"]) == 15
        assert body["is_complete"] is True

    def test_fills_only_remaining_slots_keeping_existing_picks(self, client):
        client.post("/squad/players", json={"player_id": GK1, "position": GK, "price": 40})
        client.post("/squad/players", json={"player_id": DEF_IDS[0], "position": DEF, "price": 40})
        client.post("/squad/players", json={"player_id": MID_IDS[0], "position": MID, "price": 40})
        response = client.post("/squad/optimise", json={})
        body = response.json()
        assert len(body["squad"]) == 15
        squad_ids = {p["player_id"] for p in body["squad"]}
        assert {GK1, DEF_IDS[0], MID_IDS[0]}.issubset(squad_ids)

    def test_default_objective_is_starting_xi(self, client):
        response = client.post("/squad/optimise", json={})
        assert response.status_code == 200

    def test_full_squad_objective_is_accepted(self, client):
        response = client.post("/squad/optimise", json={"objective": "full_squad"})
        assert response.status_code == 200

    def test_invalid_objective_is_rejected(self, client):
        response = client.post("/squad/optimise", json={"objective": "nonsense"})
        assert response.status_code == 400

    def test_locked_selection_violating_club_limit_returns_400(self, client):
        app_state = state_module.get_app_state()
        app_state.players[9001]["team_id"] = 100
        app_state.team_id_by_player[9001] = 100
        illegal_squad = (
            SquadPlayer(GK1, GK, 40),
            SquadPlayer(GK2, GK, 40),
            SquadPlayer(DEF_IDS[0], DEF, 40),
            SquadPlayer(9001, FWD, 40),
        )
        state_module.set_squad_state(SquadState(squad=illegal_squad))
        response = client.post("/squad/optimise", json={})
        assert response.status_code == 400
        assert response.json()["code"] == "club_limit"


class TestSquadPoints:
    def test_before_complete_squad_is_rejected(self, client):
        response = client.get("/squad/points")
        assert response.status_code == 400

    def test_total_reflects_captain_doubling(self, client):
        _build_full_squad(client)
        body = client.get("/squad/points").json()
        assert body["total"] > body["starting_xi_points"] - body["captain_bonus"]

    def test_horizon_param_widens_the_window(self, client):
        _build_full_squad(client)
        one_gw = client.get("/squad/points", params={"horizon": 1}).json()
        three_gw = client.get("/squad/points", params={"horizon": 3}).json()
        assert three_gw["total"] > one_gw["total"]

    def test_bench_boost_chip_preview_counts_bench_points(self, client):
        _build_full_squad(client)
        no_chip = client.get("/squad/points").json()
        bench_boost = client.get("/squad/points", params={"chip": "bench_boost"}).json()
        assert bench_boost["total"] > no_chip["total"]
        assert bench_boost["bench_points"] > 0

    def test_triple_captain_chip_preview_triples_the_captain(self, client):
        _build_full_squad(client)
        no_chip = client.get("/squad/points").json()
        triple = client.get("/squad/points", params={"chip": "triple_captain"}).json()
        assert triple["captain_bonus"] > no_chip["captain_bonus"]

    def test_chip_preview_is_stateless_nothing_is_spent(self, client):
        _build_full_squad(client)
        client.get("/squad/points", params={"chip": "bench_boost"})
        plain = client.get("/squad/points").json()
        bench_boost_again = client.get("/squad/points", params={"chip": "bench_boost"}).json()
        assert plain["total"] < bench_boost_again["total"]

    def test_unknown_chip_is_rejected(self, client):
        _build_full_squad(client)
        response = client.get("/squad/points", params={"chip": "wildcard"})
        assert response.status_code == 400


class TestPlayersPanel:
    def test_list_players_returns_rows(self, client):
        response = client.get("/players")
        assert response.status_code == 200
        body = response.json()
        assert len(body) > 0
        assert len(body[0]["fixtures"]) == 3  # always three, per D11

    def test_filter_by_position(self, client):
        response = client.get("/players", params={"position": GK})
        body = response.json()
        assert all(row["position"] == GK for row in body)

    def test_search_by_name(self, client):
        response = client.get("/players", params={"search": f"Player{GK1}"})
        body = response.json()
        assert any(row["player_id"] == GK1 for row in body)

    def test_player_detail_breakdown(self, client):
        response = client.get(f"/players/{GK1}")
        assert response.status_code == 200
        body = response.json()
        assert body["breakdown"]["total"] == pytest.approx(body["expected_points"])

    def test_unknown_player_404(self, client):
        response = client.get("/players/999999")
        assert response.status_code == 404


class TestPersistenceAcrossRestart:
    def test_squad_survives_a_simulated_restart(self, client, tmp_path):
        _build_full_squad(client)

        # Simulate a restart: clear in-memory singletons but keep the same DB file and app state.
        db_path = state_module._db_path
        app_state = state_module.get_app_state()
        state_module.reset_state(db_path=db_path)
        state_module.set_app_state(app_state)

        body = client.get("/squad").json()
        assert body["is_complete"] is True
        assert len(body["squad"]) == 15

    def test_partial_squad_survives_a_simulated_restart(self, client):
        client.post("/squad/players", json={"player_id": GK1, "position": GK, "price": 40})

        db_path = state_module._db_path
        app_state = state_module.get_app_state()
        state_module.reset_state(db_path=db_path)
        state_module.set_app_state(app_state)

        body = client.get("/squad").json()
        assert body["is_complete"] is False
        assert len(body["squad"]) == 1
        assert body["squad"][0]["player_id"] == GK1
