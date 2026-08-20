"""Tests for api.persistence -- the CommittedSquad/PendingDraft <-> SavedSquad JSON round trip."""

from __future__ import annotations

from sqlalchemy.orm import Session

from api.persistence import load_squad_state, save_squad_state
from engine.data.storage import Base, get_engine
from engine.scoring import DEF, FWD, GK, MID
from features.chip_calendar import ChipUsage
from features.squad_draft import CommittedSquad, confirm_initial_squad, open_draft
from features.team_state import SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]


def _player(player_id: int, position: str, price: int = 40) -> SquadPlayer:
    return SquadPlayer(
        player_id=player_id, position=position, purchase_price=price, current_price=price
    )


def _squad() -> tuple[SquadPlayer, ...]:
    players = [_player(GK1, GK), _player(GK2, GK)]
    players += [_player(pid, DEF) for pid in DEF_IDS]
    players += [_player(pid, MID) for pid in MID_IDS]
    players += [_player(pid, FWD) for pid in FWD_IDS]
    return tuple(players)


def _team_id_by_player() -> dict[int, int]:
    all_ids = [GK1, GK2, *DEF_IDS, *MID_IDS, *FWD_IDS]
    return {pid: pid for pid in all_ids}


def _committed(gameweek: int = 1) -> CommittedSquad:
    return confirm_initial_squad(
        squad=_squad(),
        starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
        bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
        team_id_by_player=_team_id_by_player(),
        gameweek=gameweek,
    )


def _session(tmp_path) -> Session:
    engine = get_engine(str(tmp_path / "test.sqlite"))
    Base.metadata.create_all(engine)
    return Session(engine)


class TestRoundTrip:
    def test_committed_squad_round_trips(self, tmp_path):
        session = _session(tmp_path)
        committed = _committed()
        save_squad_state(session, "2026-27", committed, None)

        loaded = load_squad_state(session, "2026-27")
        assert loaded is not None
        loaded_committed, loaded_pending = loaded
        assert loaded_committed.team_state == committed.team_state
        assert loaded_committed.chip_usage == committed.chip_usage
        assert loaded_committed.committed_gameweek == committed.committed_gameweek
        assert loaded_pending is None

    def test_pending_draft_round_trips(self, tmp_path):
        session = _session(tmp_path)
        committed = _committed()
        draft = open_draft(committed, gameweek=2)
        save_squad_state(session, "2026-27", committed, draft)

        _, loaded_pending = load_squad_state(session, "2026-27")
        assert loaded_pending == draft

    def test_free_hit_snapshot_round_trips(self, tmp_path):
        session = _session(tmp_path)
        committed = _committed()
        snapshot = committed.team_state
        committed_with_snapshot = CommittedSquad(
            team_state=committed.team_state,
            chip_usage=ChipUsage(first_half_played=frozenset({"free_hit"})),
            active_chip="free_hit",
            active_chip_gameweek=3,
            free_hit_snapshot=snapshot,
            free_hit_snapshot_gameweek=3,
            committed_gameweek=3,
        )
        save_squad_state(session, "2026-27", committed_with_snapshot, None)

        loaded, _ = load_squad_state(session, "2026-27")
        assert loaded.free_hit_snapshot == snapshot
        assert loaded.free_hit_snapshot_gameweek == 3
        assert loaded.active_chip == "free_hit"

    def test_build_mode_squad_with_no_team_state_round_trips(self, tmp_path):
        session = _session(tmp_path)
        committed = CommittedSquad(team_state=None, committed_gameweek=1)
        save_squad_state(session, "2026-27", committed, None)

        loaded, _ = load_squad_state(session, "2026-27")
        assert loaded.team_state is None

    def test_no_saved_state_returns_none(self, tmp_path):
        session = _session(tmp_path)
        assert load_squad_state(session, "2026-27") is None

    def test_season_mismatch_returns_none(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _committed(), None)
        assert load_squad_state(session, "2027-28") is None

    def test_saving_twice_overwrites_the_single_row(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _committed(gameweek=1), None)
        save_squad_state(session, "2026-27", _committed(gameweek=2), None)

        loaded, _ = load_squad_state(session, "2026-27")
        assert loaded.committed_gameweek == 2

        from sqlalchemy import func, select

        from engine.data.storage import SavedSquad

        count = session.execute(select(func.count()).select_from(SavedSquad)).scalar_one()
        assert count == 1

    def test_gameweek_hit_cost_round_trips(self, tmp_path):
        session = _session(tmp_path)
        committed = CommittedSquad(
            team_state=_committed().team_state, committed_gameweek=1, gameweek_hit_cost=8
        )
        save_squad_state(session, "2025-26", committed, None)

        loaded, _ = load_squad_state(session, "2025-26")
        assert loaded.gameweek_hit_cost == 8

    def test_gameweek_hit_cost_defaults_to_zero(self, tmp_path):
        session = _session(tmp_path)
        save_squad_state(session, "2026-27", _committed(), None)

        loaded, _ = load_squad_state(session, "2026-27")
        assert loaded.gameweek_hit_cost == 0
