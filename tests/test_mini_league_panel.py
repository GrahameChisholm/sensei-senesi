"""Tests for api.mini_league_panel -- row assembly over an already-fetched LeagueSnapshot plus
already-loaded AppState (build_mini_league_panel), and the in-process TTL cache
(get_cached_league_snapshot). The underlying math is already covered at the features/ level
(test_mini_league.py); these tests check the wiring: which gameweek is used, which entry is
excluded as "you", and that the cache actually avoids re-fetching.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from api.mini_league_panel import (
    build_mini_league_panel,
    get_cached_league_snapshot,
    reset_snapshot_cache,
)
from api.state import AppState
from engine.aggregate import ComponentBreakdown
from engine.data.league_state_builder import LeagueEntry, LeagueSnapshot
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import DEF, FWD, GK, MID
from features.team_state import MyTeamState, SquadPlayer

GK1, GK2 = 1, 2
DEF_IDS = [11, 12, 13, 14, 15]
MID_IDS = [21, 22, 23, 24, 25]
FWD_IDS = [31, 32, 33]
MY_ENTRY_ID = 500


def _player(player_id: int, position: str, price: int = 40) -> SquadPlayer:
    return SquadPlayer(player_id=player_id, position=position, price=price)


def _squad() -> tuple[SquadPlayer, ...]:
    players = [_player(GK1, GK), _player(GK2, GK)]
    players += [_player(pid, DEF) for pid in DEF_IDS]
    players += [_player(pid, MID) for pid in MID_IDS]
    players += [_player(pid, FWD) for pid in FWD_IDS]
    return tuple(players)


def _team_state(**overrides) -> MyTeamState:
    defaults = dict(
        squad=_squad(),
        starting_xi=(GK1, *DEF_IDS[:4], *MID_IDS[:4], *FWD_IDS[:2]),
        bench_order=(DEF_IDS[4], MID_IDS[4], FWD_IDS[2], GK2),
        captain_id=MID_IDS[0],
        vice_captain_id=MID_IDS[1],
    )
    defaults.update(overrides)
    return MyTeamState(**defaults)


def _minutes() -> MinutesDistribution:
    return MinutesDistribution(
        p_zero=0.1,
        p_1_to_59=0.1,
        p_60_plus=0.8,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )


def _breakdown(total: float) -> ComponentBreakdown:
    return ComponentBreakdown(
        appearance=2.0,
        goals=total - 2.0,
        assists=0.0,
        clean_sheet=0.0,
        goals_conceded=0.0,
        defensive_contribution=0.0,
        saves=0.0,
        bonus=0.0,
        cards=0.0,
        penalty_misses=0.0,
    )


def _flat_projections(player_ids, position: str, gameweeks: list[int], points: float = 4.0) -> dict:
    return {
        pid: project_player_horizon(
            pid,
            position,
            {
                gw: project_player_gameweek(pid, position, gw, _minutes(), _breakdown(points))
                for gw in gameweeks
            },
        )
        for pid in player_ids
    }


def _app_state(gameweek: int = 1, extra_projections: dict | None = None) -> AppState:
    positions = {GK1: GK, GK2: GK}
    positions.update({pid: DEF for pid in DEF_IDS})
    positions.update({pid: MID for pid in MID_IDS})
    positions.update({pid: FWD for pid in FWD_IDS})

    projections: dict = {}
    for position in {GK, DEF, MID, FWD}:
        ids = [pid for pid, pos in positions.items() if pos == position]
        projections.update(_flat_projections(ids, position, [gameweek]))
    if extra_projections:
        projections.update(extra_projections)

    players = {
        pid: {"web_name": f"Player{pid}", "team_id": 100, "position": pos, "price": 40}
        for pid, pos in positions.items()
    }

    return AppState(
        season="2026-27",
        gameweek=gameweek,
        horizon_gameweeks=[gameweek],
        deadline_passed=False,
        generated_at=datetime(2026, 8, 20, 9, 0, tzinfo=UTC),
        deadline_time=datetime(2026, 8, 21, 17, 30, tzinfo=UTC),
        model_version="test",
        projections=projections,
        players=players,
        teams={100: {"name": "Team", "short_name": "TMA"}},
        fixtures=[],
        diagnostics={},
    )


def _rival(
    entry_id: int, picks: dict[int, int], total_points: int = 0, chips: tuple = ()
) -> LeagueEntry:
    return LeagueEntry(
        entry_id=entry_id,
        manager_name=f"Manager {entry_id}",
        team_name=f"Team {entry_id}",
        rank=entry_id,
        total_points=total_points,
        gameweek_points=0,
        picks=picks,
        chips=chips,
    )


def _mirrored_picks(state: MyTeamState) -> dict[int, int]:
    """A rival who owns exactly the given squad at exactly its own multipliers -- an empty
    ``{}`` picks dict instead would implicitly mean "the rival owns none of your 15 players",
    turning every one of them into a spurious differential (see test_mini_league.py's own
    identically-named helper for the full explanation)."""
    picks = {
        player_id: 2 if player_id == state.captain_id else 1 for player_id in state.starting_xi
    }
    picks.update(dict.fromkeys(state.bench_order, 0))
    return picks


def _snapshot(entries: list[LeagueEntry], picks_gameweek: int = 1) -> LeagueSnapshot:
    return LeagueSnapshot(
        league_id=999,
        league_name="Test League",
        picks_gameweek=picks_gameweek,
        entries=tuple(entries),
    )


class TestBuildMiniLeaguePanel:
    def test_raises_when_my_entry_is_not_in_the_snapshot(self):
        snapshot = _snapshot([_rival(1, {})])
        with pytest.raises(ValueError):
            build_mini_league_panel(_app_state(), _team_state(), snapshot, my_entry_id=MY_ENTRY_ID)

    def test_excludes_my_own_entry_from_the_rivals_list(self):
        snapshot = _snapshot([_rival(MY_ENTRY_ID, {}), _rival(1, {}), _rival(2, {})])
        panel = build_mini_league_panel(_app_state(), _team_state(), snapshot, MY_ENTRY_ID)
        assert {rival.entry_id for rival in panel.rivals} == {1, 2}

    def test_my_own_entry_is_excluded_from_effective_ownership_too(self):
        """A player only my own snapshot entry picked must show 0.0 EO -- not counted against
        myself as "the field"."""
        snapshot = _snapshot([_rival(MY_ENTRY_ID, {DEF_IDS[4]: 2}), _rival(1, {})])
        panel = build_mini_league_panel(_app_state(), _team_state(), snapshot, MY_ENTRY_ID)
        exposure = next(e for e in panel.exposures if e.player_id == DEF_IDS[4])
        assert exposure.ownership.eo_multiplier == 0.0

    def test_captain_options_are_scoped_to_the_starting_xi(self):
        state = _team_state()
        snapshot = _snapshot([_rival(MY_ENTRY_ID, {}), _rival(1, {})])
        panel = build_mini_league_panel(_app_state(), state, snapshot, MY_ENTRY_ID)
        assert {option.player_id for option in panel.captain_options} == set(state.starting_xi)

    def test_my_own_rank_and_total_points_come_from_my_snapshot_entry(self):
        snapshot = _snapshot(
            [_rival(MY_ENTRY_ID, {}, total_points=123), _rival(1, {}, total_points=99)]
        )
        panel = build_mini_league_panel(_app_state(), _team_state(), snapshot, MY_ENTRY_ID)
        assert panel.my_total_points == 123
        assert panel.my_rank == MY_ENTRY_ID  # _rival() sets rank == entry_id in this fixture

    def test_gameweek_is_the_apps_own_current_gameweek_not_picks_gameweek(self):
        app_state = _app_state(gameweek=7)
        snapshot = _snapshot([_rival(MY_ENTRY_ID, {}), _rival(1, {})], picks_gameweek=5)
        panel = build_mini_league_panel(app_state, _team_state(), snapshot, MY_ENTRY_ID)
        assert panel.gameweek == 7
        assert panel.picks_gameweek == 5

    def test_posture_uses_my_own_entrys_total_points_from_the_snapshot(self):
        state = _team_state()
        snapshot = _snapshot(
            [
                _rival(MY_ENTRY_ID, {}, total_points=100),
                # A rival who mirrors my own squad exactly -> zero expected gap this gameweek, so
                # the only thing that can drive the projection is the points deficit itself.
                _rival(1, _mirrored_picks(state), total_points=150),
            ]
        )
        panel = build_mini_league_panel(_app_state(), state, snapshot, MY_ENTRY_ID)
        [rival] = panel.rivals
        assert rival.head_to_head.expected_gap == 0.0
        assert rival.posture.projected_final_gap < 0


class _CountingClient:
    def __init__(self, snapshot_builder):
        self.calls = 0
        self._snapshot_builder = snapshot_builder

    def get_league_standings(self, league_id, page=1):
        self.calls += 1
        return self._snapshot_builder(page)

    def get_entry(self, entry_id):
        return {"id": entry_id, "current_event": 1}

    def get_entry_picks(self, entry_id, gameweek):
        return {"picks": []}

    def get_entry_history(self, entry_id):
        return {"chips": []}


def _standings_page(results, has_next=False):
    return {
        "league": {"id": 999, "name": "Test League"},
        "standings": {"has_next": has_next, "results": results},
    }


class TestGetCachedLeagueSnapshot:
    def setup_method(self):
        reset_snapshot_cache()

    def teardown_method(self):
        reset_snapshot_cache()

    def test_second_call_within_ttl_reuses_the_cached_snapshot(self):
        client = _CountingClient(
            lambda page: _standings_page([{"entry": 1, "rank": 1, "total": 0}])
        )
        get_cached_league_snapshot(client, league_id=999)
        get_cached_league_snapshot(client, league_id=999)
        assert client.calls == 1

    def test_refresh_bypasses_the_cache(self):
        client = _CountingClient(
            lambda page: _standings_page([{"entry": 1, "rank": 1, "total": 0}])
        )
        get_cached_league_snapshot(client, league_id=999)
        get_cached_league_snapshot(client, league_id=999, refresh=True)
        assert client.calls == 2

    def test_different_leagues_are_cached_independently(self):
        client = _CountingClient(
            lambda page: _standings_page([{"entry": 1, "rank": 1, "total": 0}])
        )
        get_cached_league_snapshot(client, league_id=999)
        get_cached_league_snapshot(client, league_id=1000)
        assert client.calls == 2
