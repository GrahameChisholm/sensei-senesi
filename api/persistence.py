"""JSON (de)serialization for the team-selection page's squad state, and the round trip through
``engine.data.storage`` -- deliberately an API-layer concern: ``features/`` stays pure and I/O-free
(this repo's own convention), so serialization lives here, not there.

Two tables back this: ``SavedSquad`` (single row, account-level fields that don't vary week to
week -- ``budget_ceiling``, ``mini_league_ids`` -- plus legacy squad columns kept only as a
one-time migration seed) and ``SavedSquadGameweek`` (one row per horizon gameweek that has ever
been edited, the week-on-week simulator's actual squad snapshots).
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.squad_state import SquadState
from engine.data.storage import SavedSquad, SavedSquadGameweek
from features.team_state import SquadPlayer

__all__ = [
    "save_squad_state",
    "load_squad_state",
    "save_squad_account_fields",
    "save_squad_gameweek",
    "load_squad_gameweek",
    "latest_squad_gameweek_at_or_before",
    "delete_squad_gameweeks_after",
]


def _squad_player_to_dict(player: SquadPlayer) -> dict:
    return {"player_id": player.player_id, "position": player.position, "price": player.price}


def _squad_player_from_dict(data: dict) -> SquadPlayer:
    return SquadPlayer(player_id=data["player_id"], position=data["position"], price=data["price"])


def save_squad_state(session: Session, season: str, state: SquadState) -> None:
    """Upsert the single ``id=1`` account-level row -- ``budget_ceiling``/``mini_league_ids``,
    written whenever those change (import/clear). The squad/XI/bench/captain/vice columns are
    still written too, for backward compatibility with a season that predates
    ``SavedSquadGameweek`` (see :func:`load_squad_state`'s own docstring), but are never read once
    that table has any row for the season."""
    row = session.get(SavedSquad, 1)
    if row is None:
        row = SavedSquad(id=1)
        session.add(row)

    row.season = season
    row.squad_json = json.dumps([_squad_player_to_dict(player) for player in state.squad])
    row.starting_xi_json = json.dumps(list(state.starting_xi))
    row.bench_order_json = json.dumps(list(state.bench_order))
    row.captain_id = state.captain_id
    row.vice_captain_id = state.vice_captain_id
    row.mini_league_ids = ",".join(str(mid) for mid in state.mini_league_ids)
    row.budget_ceiling = state.budget_ceiling
    row.season_transfers_made = state.season_transfers_made
    session.commit()


def save_squad_account_fields(
    session: Session,
    season: str,
    budget_ceiling: int,
    mini_league_ids: tuple[int, ...],
    season_transfers_made: int,
) -> None:
    """Upsert only the account-level fields on the single ``id=1`` row -- called on every squad
    mutation. Deliberately leaves the legacy squad/XI/bench/captain/vice columns alone (creating
    them empty only if this row doesn't exist yet): once :class:`SavedSquadGameweek` rows exist for
    a season, those columns are read only as history, and overwriting them on every mutation would
    corrupt the one-time migration seed a still-unedited earlier gameweek may be resolving through
    (see :func:`load_squad_state`)."""
    row = session.get(SavedSquad, 1)
    if row is None:
        row = SavedSquad(
            id=1,
            squad_json="[]",
            starting_xi_json="[]",
            bench_order_json="[]",
            captain_id=None,
            vice_captain_id=None,
        )
        session.add(row)

    row.season = season
    row.mini_league_ids = ",".join(str(mid) for mid in mini_league_ids)
    row.budget_ceiling = budget_ceiling
    row.season_transfers_made = season_transfers_made
    session.commit()


def load_squad_state(session: Session, season: str) -> SquadState | None:
    """Returns ``None`` if no legacy row has ever been saved for ``season``. Used two ways: as the
    account-level fields (``budget_ceiling``/``mini_league_ids``/``season_transfers_made``) every
    gameweek shares, and -- only for a season with no ``SavedSquadGameweek`` rows at all yet -- as
    the seed the decision gameweek forks from."""
    row = session.execute(select(SavedSquad).where(SavedSquad.id == 1)).scalar_one_or_none()
    if row is None or row.season != season:
        return None

    return SquadState(
        squad=tuple(_squad_player_from_dict(p) for p in json.loads(row.squad_json)),
        starting_xi=tuple(json.loads(row.starting_xi_json)),
        bench_order=tuple(json.loads(row.bench_order_json)),
        captain_id=row.captain_id,
        vice_captain_id=row.vice_captain_id,
        mini_league_ids=tuple(int(mid) for mid in row.mini_league_ids.split(",") if mid),
        budget_ceiling=row.budget_ceiling,
        season_transfers_made=row.season_transfers_made,
    )


def save_squad_gameweek(session: Session, season: str, gameweek: int, state: SquadState) -> None:
    """Upsert this gameweek's own squad/XI/bench/captain/vice snapshot -- the moment a gameweek is
    first edited, it gets a row here and stops following whatever earlier gameweek it used to
    resolve to (see ``api.state.get_squad_state``)."""
    row = session.execute(
        select(SavedSquadGameweek).where(
            SavedSquadGameweek.season == season, SavedSquadGameweek.gameweek == gameweek
        )
    ).scalar_one_or_none()
    if row is None:
        row = SavedSquadGameweek(season=season, gameweek=gameweek)
        session.add(row)

    row.squad_json = json.dumps([_squad_player_to_dict(player) for player in state.squad])
    row.starting_xi_json = json.dumps(list(state.starting_xi))
    row.bench_order_json = json.dumps(list(state.bench_order))
    row.captain_id = state.captain_id
    row.vice_captain_id = state.vice_captain_id
    session.commit()


def load_squad_gameweek(session: Session, season: str, gameweek: int) -> SquadState | None:
    """Returns ``None`` if this exact gameweek has never had its own edit -- the caller (
    ``api.state.get_squad_state``) is what walks backward to an earlier gameweek in that case, not
    this function, since a plain load has no way to know which account-level fields to merge in."""
    row = session.execute(
        select(SavedSquadGameweek).where(
            SavedSquadGameweek.season == season, SavedSquadGameweek.gameweek == gameweek
        )
    ).scalar_one_or_none()
    if row is None:
        return None

    return SquadState(
        squad=tuple(_squad_player_from_dict(p) for p in json.loads(row.squad_json)),
        starting_xi=tuple(json.loads(row.starting_xi_json)),
        bench_order=tuple(json.loads(row.bench_order_json)),
        captain_id=row.captain_id,
        vice_captain_id=row.vice_captain_id,
    )


def latest_squad_gameweek_at_or_before(session: Session, season: str, gameweek: int) -> int | None:
    """The largest saved gameweek for ``season`` that is ``<= gameweek``, or ``None`` if this
    season has no ``SavedSquadGameweek`` row that early yet."""
    row = session.execute(
        select(SavedSquadGameweek.gameweek)
        .where(SavedSquadGameweek.season == season, SavedSquadGameweek.gameweek <= gameweek)
        .order_by(SavedSquadGameweek.gameweek.desc())
        .limit(1)
    ).scalar_one_or_none()
    return row


def delete_squad_gameweeks_after(session: Session, season: str, gameweek: int) -> None:
    """Drop every saved gameweek snapshot after ``gameweek`` -- called when the decision gameweek's
    squad is reset (clear/import), since any later gameweek's snapshot was a fork of a baseline
    that no longer exists. Those gameweeks resume live-following the decision gameweek until
    edited again."""
    rows = session.execute(
        select(SavedSquadGameweek).where(
            SavedSquadGameweek.season == season, SavedSquadGameweek.gameweek > gameweek
        )
    ).scalars()
    for row in rows:
        session.delete(row)
    session.commit()
