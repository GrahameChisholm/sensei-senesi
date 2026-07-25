"""Tests for engine.data.storage — schema creation and the ground-truth results constraint."""

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from engine.data.storage import Fixture, GameweekResult, Player, Team, init_db


@pytest.fixture
def engine(tmp_path):
    return init_db(str(tmp_path / "test.sqlite"))


def _seed_team_player_fixture(session: Session):
    team_a = Team(id=1, name="Arsenal", short_name="ARS")
    team_b = Team(id=2, name="Burnley", short_name="BUR")
    session.add_all([team_a, team_b])
    player = Player(
        id=1, first_name="Bukayo", second_name="Saka", web_name="Saka", team_id=1, position="MID"
    )
    session.add(player)
    fixture = Fixture(id=1, event=1, team_h=1, team_a=2, finished=True)
    session.add(fixture)
    session.commit()
    return team_a, player, fixture


def test_init_db_creates_all_tables(engine):
    with Session(engine) as session:
        assert session.query(Team).count() == 0
        assert session.query(Player).count() == 0
        assert session.query(Fixture).count() == 0
        assert session.query(GameweekResult).count() == 0


def test_can_insert_and_read_back_a_gameweek_result(engine):
    with Session(engine) as session:
        _, player, fixture = _seed_team_player_fixture(session)
        result = GameweekResult(
            player_id=player.id,
            event=1,
            fixture_id=fixture.id,
            minutes=90,
            total_points=12,
            goals_scored=1,
            assists=1,
            defensive_contribution=1,
        )
        session.add(result)
        session.commit()

    with Session(engine) as session:
        stored = session.query(GameweekResult).one()
        assert stored.total_points == 12
        assert stored.minutes == 90


def test_gameweek_result_unique_per_player_and_event(engine):
    with Session(engine) as session:
        _, player, fixture = _seed_team_player_fixture(session)
        session.add(
            GameweekResult(
                player_id=player.id, event=1, fixture_id=fixture.id, minutes=90, total_points=5
            )
        )
        session.commit()

        session.add(
            GameweekResult(
                player_id=player.id, event=1, fixture_id=fixture.id, minutes=45, total_points=2
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()


def test_init_db_is_idempotent(tmp_path):
    db_path = str(tmp_path / "idempotent.sqlite")
    init_db(db_path)
    init_db(db_path)  # should not raise on a second call
