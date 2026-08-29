"""Tests for api.settings -- the AppSettingsData <-> AppSettings JSON-free round trip, and the
MINI_LEAGUE_PLAN M14 one-time legacy carry-over from SquadState.mini_league_ids."""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from api.settings import AppSettingsData, load_app_settings, save_app_settings
from engine.data.storage import AppSettings, Base, get_engine


def _session(tmp_path) -> Session:
    engine = get_engine(str(tmp_path / "test.sqlite"))
    Base.metadata.create_all(engine)
    return Session(engine)


class TestRoundTrip:
    def test_full_settings_round_trip(self, tmp_path):
        session = _session(tmp_path)
        settings = AppSettingsData(
            fpl_team_id=12345, mini_league_ids=(111, 222), planning_horizon_gameweeks=6
        )
        save_app_settings(session, settings)

        loaded = load_app_settings(session)
        assert loaded == settings

    def test_no_saved_settings_returns_defaults(self, tmp_path):
        session = _session(tmp_path)
        loaded = load_app_settings(session)
        assert loaded == AppSettingsData()

    def test_no_fpl_team_id_round_trips_as_none(self, tmp_path):
        session = _session(tmp_path)
        save_app_settings(session, AppSettingsData(mini_league_ids=(1,)))
        loaded = load_app_settings(session)
        assert loaded.fpl_team_id is None

    def test_no_mini_league_ids_round_trips_as_an_empty_tuple(self, tmp_path):
        session = _session(tmp_path)
        save_app_settings(session, AppSettingsData(fpl_team_id=1))
        loaded = load_app_settings(session)
        assert loaded.mini_league_ids == ()

    def test_saving_twice_overwrites_the_single_row(self, tmp_path):
        session = _session(tmp_path)
        save_app_settings(session, AppSettingsData(fpl_team_id=1))
        save_app_settings(session, AppSettingsData(fpl_team_id=2))

        loaded = load_app_settings(session)
        assert loaded.fpl_team_id == 2

        count = session.execute(select(func.count()).select_from(AppSettings)).scalar_one()
        assert count == 1


class TestLegacyCarryOver:
    def test_a_non_empty_legacy_value_is_carried_over_on_first_load(self, tmp_path):
        session = _session(tmp_path)
        loaded = load_app_settings(session, legacy_mini_league_ids=(111, 222))
        assert loaded.mini_league_ids == (111, 222)

    def test_the_carry_over_is_persisted_so_it_only_ever_happens_once(self, tmp_path):
        session = _session(tmp_path)
        load_app_settings(session, legacy_mini_league_ids=(111, 222))

        # A later load with a *different* (or absent) legacy value must not override the row
        # that's now on disk -- the carry-over already happened.
        loaded = load_app_settings(session, legacy_mini_league_ids=(999,))
        assert loaded.mini_league_ids == (111, 222)

        loaded_again = load_app_settings(session)
        assert loaded_again.mini_league_ids == (111, 222)

    def test_an_empty_legacy_value_does_not_create_a_row(self, tmp_path):
        session = _session(tmp_path)
        load_app_settings(session, legacy_mini_league_ids=())

        count = session.execute(select(func.count()).select_from(AppSettings)).scalar_one()
        assert count == 0

    def test_legacy_value_is_ignored_once_real_settings_have_been_saved(self, tmp_path):
        session = _session(tmp_path)
        save_app_settings(session, AppSettingsData(fpl_team_id=1))

        loaded = load_app_settings(session, legacy_mini_league_ids=(111, 222))
        assert loaded.mini_league_ids == ()
        assert loaded.fpl_team_id == 1
