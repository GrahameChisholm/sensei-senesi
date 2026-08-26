"""Tests for api.main -- the thin FastAPI wiring over features.squad_rules/squad_draft/squad_points.
Business-rule correctness (legality, hit costing, Free Hit reversion) is already exhaustively
covered at the features/ level (test_squad_rules.py, test_squad_draft.py); these tests check that
the API wires requests to the right function, returns the right shape, and turns a SquadRuleError
into a 400 with a RuleViolationOut body -- not the rules themselves.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
import api.state as state_module
from engine.aggregate import ComponentBreakdown
from engine.data.fpl_client import FPLClientError
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID

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


def _build_full_squad(client: TestClient) -> None:
    for player_id in ALL_IDS:
        response = client.post(
            "/squad/build/players",
            json={"player_id": player_id, "position": _position_for(player_id), "price": 40},
        )
        assert response.status_code == 200, response.json()


def _confirm_build(client: TestClient) -> dict:
    starting_xi = [GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]]
    bench_order = [DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2]
    response = client.post(
        "/squad/build/confirm",
        json={
            "player_ids": ALL_IDS,
            "starting_xi": starting_xi,
            "bench_order": bench_order,
            "captain_id": MID_IDS[0],
            "vice_captain_id": MID_IDS[1],
        },
    )
    assert response.status_code == 200, response.json()
    return response.json()


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


class TestBuildMode:
    def test_empty_squad_state(self, client):
        body = client.get("/squad").json()
        assert body["is_complete"] is False
        assert body["build_picks"] == []
        assert body["committed"] is None

    def test_adding_a_player(self, client):
        response = client.post(
            "/squad/build/players", json={"player_id": GK1, "position": GK, "price": 40}
        )
        body = response.json()
        assert len(body["build_picks"]) == 1
        assert body["build_picks"][0]["player_id"] == GK1

    def test_adding_duplicate_player_is_rejected(self, client):
        client.post("/squad/build/players", json={"player_id": GK1, "position": GK, "price": 40})
        response = client.post(
            "/squad/build/players", json={"player_id": GK1, "position": GK, "price": 40}
        )
        assert response.status_code == 400
        assert response.json()["code"] == "duplicate"

    def test_removing_a_player(self, client):
        client.post("/squad/build/players", json={"player_id": GK1, "position": GK, "price": 40})
        response = client.delete(f"/squad/build/players/{GK1}")
        assert response.json()["build_picks"] == []

    def test_confirm_with_illegal_squad_is_rejected(self, client):
        _build_full_squad(client)
        response = client.post(
            "/squad/build/confirm",
            json={
                "player_ids": ALL_IDS,
                "starting_xi": [
                    GK1,
                    GK2,
                    *DEF_IDS[:3],
                    *MID_IDS[:4],
                    *FWD_IDS[:2],
                ],  # 2 GK -- illegal
                "bench_order": [
                    pid
                    for pid in ALL_IDS
                    if pid not in (GK1, GK2, *DEF_IDS[:3], *MID_IDS[:4], *FWD_IDS[:2])
                ],
                "captain_id": MID_IDS[0],
                "vice_captain_id": MID_IDS[1],
            },
        )
        assert response.status_code == 400

    def test_confirm_produces_a_real_committed_squad(self, client):
        _build_full_squad(client)
        body = _confirm_build(client)
        assert body["is_complete"] is True
        assert body["committed"] is not None
        assert body["build_picks"] is None
        assert len(body["committed"]["squad"]) == 15

    def test_cannot_confirm_twice(self, client):
        _build_full_squad(client)
        _confirm_build(client)
        response = client.post(
            "/squad/build/confirm",
            json={
                "player_ids": ALL_IDS,
                "starting_xi": [GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]],
                "bench_order": [DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2],
                "captain_id": MID_IDS[0],
                "vice_captain_id": MID_IDS[1],
            },
        )
        assert response.status_code == 400


class TestWipeSquad:
    """The sandbox reset (overriding TEAM_PAGE_PLAN D21 by explicit product direction) -- discards
    a committed squad and returns to the empty-budget build screen, with no sell-price economics
    involved at all."""

    def _committed_client(self, client) -> TestClient:
        _build_full_squad(client)
        _confirm_build(client)
        return client

    def test_wipes_a_committed_squad_back_to_the_build_screen(self, client):
        self._committed_client(client)
        response = client.post("/squad/wipe")
        body = response.json()
        assert response.status_code == 200
        assert body["is_complete"] is False
        assert body["committed"] is None
        assert body["build_picks"] == []

    def test_full_budget_is_available_after_wiping(self, client):
        self._committed_client(client)
        client.post("/squad/wipe")
        response = client.post(
            "/squad/build/players", json={"player_id": 999, "position": FWD, "price": 130}
        )
        # £13.0m is well beyond any real sell price this squad's players could have produced --
        # only reachable if the wipe actually restored the full £100m budget, not squad value.
        assert response.status_code == 200

    def test_wiping_an_already_empty_squad_is_a_harmless_no_op(self, client):
        response = client.post("/squad/wipe")
        assert response.status_code == 200
        assert response.json()["is_complete"] is False

    def test_discards_any_pending_draft_too(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        assert client.get("/squad").json()["draft"] is not None
        response = client.post("/squad/wipe")
        assert response.json()["draft"] is None

    def test_clears_chip_usage(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        client.post("/squad/draft/chip", json={"chip": "bench_boost"})
        client.post("/squad/draft/confirm")
        assert "bench_boost" not in client.get("/squad").json()["chips_available"]

        client.post("/squad/wipe")
        _build_full_squad(client)
        body = _confirm_build(client)
        assert "bench_boost" in body["chips_available"]

    def test_can_rebuild_and_confirm_a_fresh_squad_after_wiping(self, client):
        self._committed_client(client)
        client.post("/squad/wipe")
        _build_full_squad(client)
        body = _confirm_build(client)
        assert body["is_complete"] is True
        assert len(body["committed"]["squad"]) == 15


class TestImportSquad:
    """POST /squad/import -- fetches a real manager's squad live via FPLClient (monkeypatched
    here) and commits it via features.squad_draft.confirm_imported_squad."""

    def _stub_client(
        self,
        monkeypatch,
        *,
        entry=None,
        transfers=None,
        history=None,
        error=None,
        now_cost=45,
        picks_gameweeks_seen=None,
    ):
        entry = entry if entry is not None else {"last_deadline_bank": 55, "current_event": 1}
        transfers = transfers if transfers is not None else []
        history = history if history is not None else {"current": [], "chips": []}
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

            def get_entry_transfers(self, entry_id):
                return transfers

            def get_entry_history(self, entry_id):
                return history

            def get_bootstrap_static(self):
                return {"elements": elements}

        monkeypatch.setattr(api_main, "FPLClient", lambda: _StubFPLClient())

    def test_imports_a_real_squad(self, client, monkeypatch):
        self._stub_client(
            monkeypatch, history={"current": [], "chips": [{"name": "wildcard", "event": 2}]}
        )
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200, response.json()
        body = response.json()
        assert body["is_complete"] is True
        assert body["committed"]["bank"] == 55
        assert len(body["committed"]["squad"]) == 15
        assert "wildcard" not in body["chips_available"]

    def test_purchase_price_falls_back_to_current_price_with_no_transfer_record(
        self, client, monkeypatch
    ):
        self._stub_client(monkeypatch, now_cost=45)
        response = client.post("/squad/import", json={"team_id": 123456})
        body = response.json()
        assert all(p["purchase_price"] == 45 for p in body["committed"]["squad"])

    def test_squad_over_100m_via_price_rises_is_accepted(self, client, monkeypatch):
        # Real squads legitimately drift above £100m of nominal spend as prices rise; the import
        # path must not reject that the way confirm_initial_squad's from-scratch check would.
        self._stub_client(monkeypatch, now_cost=100)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200, response.json()
        assert response.json()["committed"]["bank"] == 55

    def test_overwrites_an_existing_committed_squad(self, client, monkeypatch):
        _build_full_squad(client)
        _confirm_build(client)
        self._stub_client(monkeypatch)
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200
        assert response.json()["committed"]["bank"] == 55

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
        self._stub_client(
            monkeypatch,
            entry={"last_deadline_bank": 55, "current_event": 7},
            picks_gameweeks_seen=seen,
        )
        response = client.post("/squad/import", json={"team_id": 123456})
        assert response.status_code == 200, response.json()
        assert seen == [7]


class TestEditDraftLifecycle:
    def _committed_client(self, client) -> TestClient:
        _build_full_squad(client)
        _confirm_build(client)
        return client

    def test_cannot_open_draft_before_squad_exists(self, client):
        response = client.post("/squad/draft")
        assert response.status_code == 400

    def test_open_draft(self, client):
        self._committed_client(client)
        response = client.post("/squad/draft")
        body = response.json()
        assert body["draft"] is not None
        assert body["draft"]["transfers_made"] == 0

    def test_substitute_in_draft(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        response = client.post(
            "/squad/draft/substitute", json={"out_id": FWD_IDS[1], "in_id": FWD_IDS[2]}
        )
        body = response.json()
        assert FWD_IDS[2] in body["draft"]["working_state"]["starting_xi"]
        assert body["draft"]["transfers_made"] == 0  # substitution is never a "transfer"

    def test_transfer_in_draft_increments_transfers_made(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        response = client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )
        body = response.json()
        assert body["draft"]["transfers_made"] == 1
        assert any(p["player_id"] == 9001 for p in body["draft"]["working_state"]["squad"])

    def test_mutation_without_open_draft_is_rejected(self, client):
        self._committed_client(client)
        response = client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )
        assert response.status_code == 400
        assert response.json()["code"] == "no_pending_draft"

    def test_reset_team_discards_only_the_draft(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )
        response = client.delete("/squad/draft")
        body = response.json()
        assert body["draft"] is None
        assert body["is_complete"] is True  # committed squad untouched
        assert not any(p["player_id"] == 9001 for p in body["committed"]["squad"])

    def test_confirm_edit_no_hit_before_deadline(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )
        response = client.post("/squad/draft/confirm")
        body = response.json()
        assert body["last_hit_cost"] == 0
        assert body["draft"] is None
        assert any(p["player_id"] == 9001 for p in body["committed"]["squad"])

    def test_set_captain_in_draft(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        response = client.post(
            "/squad/draft/captain", json={"player_id": FWD_IDS[0], "role": "captain"}
        )
        assert response.json()["draft"]["working_state"]["captain_id"] == FWD_IDS[0]

    def test_invalid_role_rejected(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        response = client.post(
            "/squad/draft/captain", json={"player_id": FWD_IDS[0], "role": "sideways"}
        )
        assert response.status_code == 400

    def test_bench_order_permutation(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        current = client.get("/squad").json()["draft"]["working_state"]["bench_order"]
        response = client.post(
            "/squad/draft/bench-order", json={"bench_order": list(reversed(current))}
        )
        assert response.json()["draft"]["working_state"]["bench_order"] == list(reversed(current))


class TestChips:
    def _committed_client(self, client) -> TestClient:
        _build_full_squad(client)
        _confirm_build(client)
        return client

    def test_playing_bench_boost(self, client):
        self._committed_client(client)
        client.post("/squad/draft")
        client.post("/squad/draft/chip", json={"chip": "bench_boost"})
        response = client.post("/squad/draft/confirm")
        body = response.json()
        assert body["active_chip"] == "bench_boost"
        assert "bench_boost" not in body["chips_available"]

    def test_chip_preview_does_not_spend_it(self, client):
        self._committed_client(client)
        points_response = client.get("/squad/points", params={"chip": "bench_boost"})
        assert points_response.status_code == 200
        squad = client.get("/squad").json()
        assert "bench_boost" in squad["chips_available"]  # preview alone never spends it


class TestOptimiseLineup:
    def _committed_client(self, client) -> TestClient:
        _build_full_squad(client)
        _confirm_build(client)
        return client

    def test_applies_immediately_without_a_draft(self, client):
        self._committed_client(client)
        response = client.post("/squad/optimise-xi")
        assert response.status_code == 200
        assert response.json()["draft"] is None  # no draft/confirm needed

    def test_before_any_committed_squad_is_rejected(self, client):
        response = client.post("/squad/optimise-xi")
        assert response.status_code == 400


class TestSquadPoints:
    def test_before_committed_squad_is_rejected(self, client):
        response = client.get("/squad/points")
        assert response.status_code == 400

    def test_total_reflects_captain_doubling(self, client):
        _build_full_squad(client)
        _confirm_build(client)
        body = client.get("/squad/points").json()
        assert body["total"] > body["starting_xi_points"] - body["captain_bonus"]

    def test_horizon_param_widens_the_window(self, client):
        _build_full_squad(client)
        _confirm_build(client)
        one_gw = client.get("/squad/points", params={"horizon": 1}).json()
        three_gw = client.get("/squad/points", params={"horizon": 3}).json()
        assert three_gw["total"] > one_gw["total"]

    def test_source_committed_scores_the_confirmed_squad_even_with_a_draft_open(self, client):
        _build_full_squad(client)
        _confirm_build(client)
        committed_total = client.get("/squad/points", params={"source": "committed"}).json()[
            "total"
        ]

        client.post("/squad/draft")
        client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )
        # The draft changed the squad, but "committed" must still reflect the pre-draft squad.
        still_committed_total = client.get("/squad/points", params={"source": "committed"}).json()[
            "total"
        ]
        assert still_committed_total == pytest.approx(committed_total)

        draft_total = client.get("/squad/points", params={"source": "draft"}).json()["total"]
        assert draft_total == pytest.approx(
            committed_total
        )  # same flat-EP fixture, just a like-for-like swap


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


class TestTransferRecommendation:
    def _committed_client(self, client) -> TestClient:
        _build_full_squad(client)
        _confirm_build(client)
        return client

    def test_before_committed_squad_is_rejected(self, client):
        response = client.get("/transfers/recommended")
        assert response.status_code == 400

    def test_no_recommendation_when_no_positive_ev_swap(self, client):
        # The fixture squad and every transfer target share the same flat projected points, so no
        # swap gains anything net of the hit cost.
        self._committed_client(client)
        response = client.get("/transfers/recommended")
        assert response.status_code == 200
        assert response.json() is None

    def test_recommends_a_clear_positive_ev_swap(self, client):
        self._committed_client(client)
        app_state = state_module.get_app_state()
        app_state.projections[9001] = _horizon(9001, points=10.0)
        response = client.get("/transfers/recommended")
        body = response.json()
        assert body is not None
        assert body["buy_player_id"] == 9001
        assert body["net_points_gain"] > 0

    def test_skips_a_candidate_that_would_breach_club_quota(self, client):
        self._committed_client(client)
        app_state = state_module.get_app_state()
        # Team 100 already holds 3 players (GK1, GK2, DEF_IDS[0]) per the fixture's team
        # assignment. Putting the otherwise-clear-best candidate on that team would push it to 4
        # -- illegal, so it must be filtered out rather than recommended.
        app_state.players[9001]["team_id"] = 100
        app_state.team_id_by_player[9001] = 100
        app_state.projections[9001] = _horizon(9001, points=10.0)
        response = client.get("/transfers/recommended")
        body = response.json()
        assert body is None or body["buy_player_id"] != 9001

    def test_evaluates_against_the_open_draft_not_the_stale_committed_squad(self, client):
        self._committed_client(client)
        app_state = state_module.get_app_state()
        app_state.projections[9001] = _horizon(9001, points=10.0)

        client.post("/squad/draft")
        client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )
        # 9001 is now owned (in the open draft), so it can no longer be recommended as a buy.
        response = client.get("/transfers/recommended")
        body = response.json()
        assert body is None or body["buy_player_id"] != 9001


class TestPersistenceAcrossRestart:
    def test_committed_squad_survives_a_simulated_restart(self, client, tmp_path):
        _build_full_squad(client)
        _confirm_build(client)

        # Simulate a restart: clear in-memory singletons but keep the same DB file and app state.
        db_path = state_module._db_path
        app_state = state_module.get_app_state()
        state_module.reset_state(db_path=db_path)
        state_module.set_app_state(app_state)

        body = client.get("/squad").json()
        assert body["is_complete"] is True
        assert len(body["committed"]["squad"]) == 15

    def test_pending_draft_survives_a_simulated_restart(self, client):
        _build_full_squad(client)
        _confirm_build(client)
        client.post("/squad/draft")
        client.post(
            "/squad/draft/transfer",
            json={"out_id": FWD_IDS[2], "in_id": 9001, "in_price": 40, "in_position": FWD},
        )

        db_path = state_module._db_path
        app_state = state_module.get_app_state()
        state_module.reset_state(db_path=db_path)
        state_module.set_app_state(app_state)

        body = client.get("/squad").json()
        assert body["draft"] is not None
        assert body["draft"]["transfers_made"] == 1

    def test_build_picks_survive_a_simulated_restart(self, client):
        client.post("/squad/build/players", json={"player_id": GK1, "position": GK, "price": 40})

        db_path = state_module._db_path
        app_state = state_module.get_app_state()
        state_module.reset_state(db_path=db_path)
        state_module.set_app_state(app_state)

        body = client.get("/squad").json()
        assert body["is_complete"] is False
        assert len(body["build_picks"]) == 1
        assert body["build_picks"][0]["player_id"] == GK1
