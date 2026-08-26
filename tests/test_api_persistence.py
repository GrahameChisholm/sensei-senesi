"""Tests for api.persistence -- the SquadState <-> SavedSquad JSON round trip."""

from __future__ import annotations

from sqlalchemy.orm import Session

from api.persistence import load_squad_state, save_squad_state
from api.squad_state import SquadState
from engine.data.storage import Base, get_engine
from engine.scoring import DEF, FWD, GK, MID
from features.squad_rules import INITIAL_BUDGET, build_team_state
from features.team_state import SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


def _player(player_id: int, position: str, price: int = 40) -> SquadPlayer:
    return SquadPlayer(player_id=player_id, position=position, price=price)


def _squad() -> tuple[SquadPlayer, ...]:
    players = [_player(GK1, GK), _player(GK2, GK)]
    players += [_player(pid, DEF) for pid in DEF_IDS]
    players += [_player(pid, MID) for pid in MID_IDS]
    players += [_player(pid, FWD) for pid in FWD_IDS]
    return tuple(players)


def _team_id_by_player() -> dict[int, int]:
    all_ids = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]
    return {pid: pid for pid in all_ids}


def _full_squad_state(
    budget_ceiling: int = INITIAL_BUDGET, mini_league_ids: tuple[int, ...] = ()
) -> SquadState:
    team_state = build_team_state(
        squad=_squad(),
        starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
        bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
        team_id_by_player=_team_id_by_player(),
    )
    return SquadState(
        squad=team_state.squad,
        starting_xi=team_state.starting_xi,
        bench_order=team_state.bench_order,
        captain_id=team_state.captain_id,
        vice_captain_id=team_state.vice_captain_id,
        mini_league_ids=mini_league_ids,
        budget_ceiling=budget_ceiling,
    )


def _session(tmp_path) -> Session:
    engine = get_engine(str(tmp_path / "test.sqlite"))
    Base.metadata.create_all(engine)
    return Session(engine)


class TestRoundTrip:
    def test_full_squad_round_trips(self, tmp_path):
        session = _session(tmp_path)
        state = _full_squad_state()
        save_squad_state(session, "2026-27", state)

        loaded = load_squad_state(session, "2026-27")
        assert loaded is not None
        assert loaded.squad == state.squad
        assert loaded.starting_xi == state.starting_xi
        assert loaded.bench_order == state.bench_order
        assert loaded.captain_id == state.captain_id
        assert loaded.vice_captain_id == state.vice_captain_id
        assert loaded.budget_ceiling == state.budget_ceiling

    def test_partial_squad_round_trips(self, tmp_path):
        session = _session(tmp_path)
        state = SquadState(squad=(_player(GK1, GK), _player(GK2, GK)))
        save_squad_state(session, "2026-27", state)

        loaded = load_squad_state(session, "2026-27")
        assert loaded is not None
        assert loaded.squad == state.squad
        assert loaded.starting_xi == ()
        assert loaded.captain_id is None

    def test_empty_squad_round_trips(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", SquadState())

        loaded = load_squad_state(session, "2026-27")
        assert loaded is not None
        assert loaded.squad == ()
        assert loaded.budget_ceiling == INITIAL_BUDGET

    def test_budget_ceiling_above_the_classic_100m_round_trips(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _full_squad_state(budget_ceiling=1500))

        loaded = load_squad_state(session, "2026-27")
        assert loaded.budget_ceiling == 1500

    def test_mini_league_ids_round_trip(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _full_squad_state(mini_league_ids=(111, 222)))

        loaded = load_squad_state(session, "2026-27")
        assert loaded.mini_league_ids == (111, 222)

    def test_no_saved_state_returns_none(self, tmp_path):
        session = _session(tmp_path)
        assert load_squad_state(session, "2026-27") is None

    def test_season_mismatch_returns_none(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _full_squad_state())
        assert load_squad_state(session, "2027-28") is None

    def test_saving_twice_overwrites_the_single_row(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _full_squad_state(budget_ceiling=1000))
        save_squad_state(session, "2026-27", _full_squad_state(budget_ceiling=1200))

        loaded = load_squad_state(session, "2026-27")
        assert loaded.budget_ceiling == 1200

        from sqlalchemy import func, select

        from engine.data.storage import SavedSquad

        count = session.execute(select(func.count()).select_from(SavedSquad)).scalar_one()
        assert count == 1
