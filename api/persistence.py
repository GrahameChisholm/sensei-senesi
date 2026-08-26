"""JSON (de)serialization for the team-selection page's one live sandbox squad, and the round
trip through ``engine.data.storage.SavedSquad`` — deliberately an API-layer concern: ``features/``
stays pure and I/O-free (this repo's own convention), so serialization lives here, not there.

A single row (``id=1``) — this is a single-user local tool, and the squad is always read and
written whole.
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from api.squad_state import SquadState
from engine.data.storage import SavedSquad
from features.team_state import SquadPlayer

__all__ = ["save_squad_state", "load_squad_state"]


def _squad_player_to_dict(player: SquadPlayer) -> dict:
    return {"player_id": player.player_id, "position": player.position, "price": player.price}


def _squad_player_from_dict(data: dict) -> SquadPlayer:
    return SquadPlayer(player_id=data["player_id"], position=data["position"], price=data["price"])


def save_squad_state(session: Session, season: str, state: SquadState) -> None:
    """Upsert the single ``id=1`` row — written on every successful mutation, so the squad
    survives a restart."""
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
    session.commit()


def load_squad_state(session: Session, season: str) -> SquadState | None:
    """Returns ``None`` if no squad has ever been saved for ``season`` — the caller should start a
    fresh, empty :class:`~api.squad_state.SquadState` in that case."""
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
    )
