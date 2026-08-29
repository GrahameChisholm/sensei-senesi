"""Tests for the decision gameweek: which gameweek the app presents as "now".

A gameweek stops being decidable at its deadline, not when its matches finish, since every
mutation this app offers (transfers, captaincy, bench order, chips) has stopped affecting it by
then. So `AppState` works the answer out at read time from real deadlines instead of trusting the
`deadline_passed` flag frozen into a projection cache when it was built, which is stale the moment
that deadline goes.

These tests are all about that resolution and how the API reports it. The cache's own `gameweek`
is still reported separately as `projections_gameweek`, because advancing on a stale cache is a
safety net for the window between a deadline and the next `build_projections` run, not a
replacement for running it.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

import api.state as state_module
from api.state import DEADLINE_BEFORE_FIRST_KICKOFF, AppState
from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import project_player_gameweek, project_player_horizon
from engine.scoring import MID

NOW = datetime.now(UTC)
PAST = NOW - timedelta(days=2)
LATER_PAST = NOW - timedelta(days=1)
FUTURE = NOW + timedelta(days=2)
LATER_FUTURE = NOW + timedelta(days=4)

TEAM_A, TEAM_B = 1, 2


def _horizon(player_id: int, points_by_gameweek: dict[int, float]):
    minutes = MinutesDistribution(
        p_zero=0.0,
        p_1_to_59=0.0,
        p_60_plus=1.0,
        expected_minutes_given_1_to_59=30.0,
        expected_minutes_given_60_plus=90.0,
    )

    def _breakdown(points: float) -> ComponentBreakdown:
        return ComponentBreakdown(
            appearance=points,
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

    return project_player_horizon(
        player_id,
        MID,
        {
            gameweek: project_player_gameweek(player_id, MID, gameweek, minutes, _breakdown(points))
            for gameweek, points in points_by_gameweek.items()
        },
    )


def _app_state(
    *,
    gameweek: int = 1,
    horizon_gameweeks: list[int] | None = None,
    deadline_time: datetime = FUTURE,
    deadline_passed: bool = False,
    deadline_times: dict[int, datetime] | None = None,
    fixtures: list[dict] | None = None,
    projections: dict | None = None,
) -> AppState:
    return AppState(
        season="2026-27",
        gameweek=gameweek,
        horizon_gameweeks=horizon_gameweeks if horizon_gameweeks is not None else [1, 2, 3],
        deadline_passed=deadline_passed,
        generated_at=NOW - timedelta(days=3),
        deadline_time=deadline_time,
        model_version="test",
        projections=projections or {},
        players={},
        teams={TEAM_A: {"name": "Team A", "short_name": "TMA"}},
        fixtures=fixtures or [],
        diagnostics={},
        deadline_times=deadline_times or {},
    )


class TestDecisionGameweek:
    def test_stays_put_while_the_deadline_is_still_ahead(self):
        state = _app_state(deadline_times={1: FUTURE, 2: LATER_FUTURE})

        assert state.decision_gameweek == 1
        assert state.remaining_horizon_gameweeks == [1, 2, 3]

    def test_advances_once_the_deadline_has_gone(self):
        state = _app_state(deadline_times={1: PAST, 2: FUTURE, 3: LATER_FUTURE})

        assert state.decision_gameweek == 2
        assert state.remaining_horizon_gameweeks == [2, 3]

    def test_skips_every_gameweek_that_has_already_locked(self):
        state = _app_state(deadline_times={1: PAST, 2: LATER_PAST, 3: FUTURE})

        assert state.decision_gameweek == 3
        assert state.remaining_horizon_gameweeks == [3]

    def test_clamps_to_the_last_horizon_gameweek_when_the_whole_horizon_has_locked(self):
        # Not a gameweek anyone can act on, it means the cache itself is stale. `is_deadline_passed`
        # is what says so, and it is the only case where it comes back True.
        state = _app_state(deadline_times={1: PAST, 2: PAST, 3: LATER_PAST})

        assert state.decision_gameweek == 3
        assert state.is_deadline_passed() is True

    def test_reports_an_open_deadline_for_a_gameweek_still_ahead(self):
        state = _app_state(deadline_times={1: PAST, 2: FUTURE, 3: LATER_FUTURE})

        assert state.is_deadline_passed() is False

    def test_uses_the_caches_own_deadline_time_for_its_own_gameweek(self):
        # A cache built before `deadline_times` existed still carries the one scalar deadline for
        # the gameweek it was built for, which is enough to know that gameweek has locked.
        state = _app_state(gameweek=1, deadline_time=PAST)

        assert state.decision_gameweek == 2

    def test_falls_back_to_ninety_minutes_before_a_gameweeks_first_kickoff(self):
        # No recorded deadlines, so GW2's has to come off its fixtures. Its first kickoff is only
        # an hour away, and FPL's deadline is 90 minutes before that, so GW2 has already locked.
        state = _app_state(
            gameweek=1,
            deadline_time=PAST,
            fixtures=[
                {
                    "team_id": TEAM_A,
                    "opponent_id": TEAM_B,
                    "gameweek": 2,
                    "is_home": True,
                    "kickoff_time": (NOW + timedelta(hours=3)).isoformat(),
                    "difficulty": 3,
                },
                {
                    "team_id": TEAM_B,
                    "opponent_id": TEAM_A,
                    "gameweek": 2,
                    "is_home": False,
                    "kickoff_time": (NOW + timedelta(hours=1)).isoformat(),
                    "difficulty": 3,
                },
            ],
        )

        assert state.deadline_for(2) == NOW + timedelta(hours=1) - DEADLINE_BEFORE_FIRST_KICKOFF
        assert state.decision_gameweek == 3

    def test_stops_advancing_at_a_gameweek_whose_deadline_is_unknown(self):
        # GW1 has demonstrably locked, GW2 has no deadline anywhere in the cache. Advancing to GW2
        # and stopping is right: skipping it would be asserting it has locked too, on no evidence.
        state = _app_state(gameweek=1, deadline_time=PAST)

        assert state.deadline_for(2) is None
        assert state.decision_gameweek == 2
        assert state.is_deadline_passed() is False

    def test_falls_back_to_the_cache_gameweek_when_the_horizon_is_empty(self):
        state = _app_state(gameweek=7, horizon_gameweeks=[], deadline_time=PAST)

        assert state.decision_gameweek == 7

    def test_expected_points_defaults_to_the_decision_gameweek(self):
        # Optimising an XI for a gameweek that has already locked changes nothing, so the default
        # target follows the decision gameweek rather than the cache's.
        state = _app_state(
            deadline_times={1: PAST, 2: FUTURE, 3: LATER_FUTURE},
            projections={9: _horizon(9, {1: 2.0, 2: 6.0, 3: 9.0})},
        )

        assert state.expected_points() == pytest.approx({9: 6.0})
        assert state.expected_points(1) == pytest.approx({9: 2.0})


class TestGameweekEndpoint:
    def _client(self, tmp_path, state: AppState) -> TestClient:
        state_module.reset_state(db_path=str(tmp_path / "test.sqlite"))
        state_module.set_app_state(state)
        from api.main import app

        return TestClient(app)

    def test_serves_the_decision_gameweek_and_the_cache_gameweek_separately(self, tmp_path):
        state = _app_state(gameweek=1, deadline_times={1: PAST, 2: FUTURE, 3: LATER_FUTURE})

        with self._client(tmp_path, state) as client:
            body = client.get("/gameweek").json()

        assert body["gameweek"] == 2
        assert body["projections_gameweek"] == 1
        assert body["deadline_time"] == FUTURE.isoformat()
        assert body["deadline_passed"] is False
        assert body["horizon_gameweeks"] == [2, 3]

        state_module.reset_state()

    def test_ignores_the_frozen_deadline_passed_flag_from_the_cache(self, tmp_path):
        # The cache says the deadline had gone when it was built. It has not gone for the gameweek
        # being decided now, and that is what the endpoint has to answer about.
        state = _app_state(
            gameweek=1,
            deadline_passed=True,
            deadline_times={1: FUTURE, 2: LATER_FUTURE, 3: LATER_FUTURE},
        )

        with self._client(tmp_path, state) as client:
            body = client.get("/gameweek").json()

        assert body["gameweek"] == 1
        assert body["deadline_passed"] is False

        state_module.reset_state()

    def test_flags_a_horizon_that_has_entirely_locked(self, tmp_path):
        state = _app_state(gameweek=1, deadline_times={1: PAST, 2: PAST, 3: LATER_PAST})

        with self._client(tmp_path, state) as client:
            body = client.get("/gameweek").json()

        assert body["gameweek"] == 3
        assert body["projections_gameweek"] == 1
        assert body["deadline_passed"] is True

        state_module.reset_state()

    def test_fixture_ticker_defaults_to_the_decision_gameweek(self, tmp_path):
        fixtures = [
            {
                "team_id": TEAM_A,
                "opponent_id": TEAM_B,
                "gameweek": gameweek,
                "is_home": True,
                "kickoff_time": (FUTURE + timedelta(days=gameweek)).isoformat(),
                "difficulty": 3,
            }
            for gameweek in range(1, 8)
        ]
        state = _app_state(
            gameweek=1,
            deadline_times={1: PAST, 2: FUTURE, 3: LATER_FUTURE},
            fixtures=fixtures,
        )

        with self._client(tmp_path, state) as client:
            rows = client.get("/fixtures").json()

        row = next(row for row in rows if row["team_id"] == TEAM_A)
        assert [cell["gameweek"] for cell in row["gameweeks"]] == [2, 3, 4, 5, 6]

        state_module.reset_state()
