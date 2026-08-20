"""JSON (de)serialization for ``features.squad_draft``'s ``CommittedSquad``/``PendingDraft``, and
the round trip through ``engine.data.storage.SavedSquad`` (D17/G6) — deliberately an API-layer
concern: ``features/`` stays pure and I/O-free (this repo's own convention), so serialization
lives here, not there.

A single row (``id=1``) — this is a single-user local tool, and both the committed squad and any
pending draft are always read and written whole.
"""

from __future__ import annotations

import json

from sqlalchemy import select
from sqlalchemy.orm import Session

from engine.data.storage import SavedSquad
from features.chip_calendar import ChipUsage
from features.squad_draft import CommittedSquad, PendingDraft
from features.team_state import MyTeamState, SquadPlayer

__all__ = [
    "team_state_to_dict",
    "team_state_from_dict",
    "chip_usage_to_dict",
    "chip_usage_from_dict",
    "pending_draft_to_dict",
    "pending_draft_from_dict",
    "save_squad_state",
    "load_squad_state",
]


def _squad_player_to_dict(player: SquadPlayer) -> dict:
    return {
        "player_id": player.player_id,
        "position": player.position,
        "purchase_price": player.purchase_price,
        "current_price": player.current_price,
    }


def _squad_player_from_dict(data: dict) -> SquadPlayer:
    return SquadPlayer(
        player_id=data["player_id"],
        position=data["position"],
        purchase_price=data["purchase_price"],
        current_price=data["current_price"],
    )


def team_state_to_dict(state: MyTeamState) -> dict:
    return {
        "squad": [_squad_player_to_dict(player) for player in state.squad],
        "starting_xi": list(state.starting_xi),
        "bench_order": list(state.bench_order),
        "captain_id": state.captain_id,
        "vice_captain_id": state.vice_captain_id,
        "bank": state.bank,
        "free_transfers": state.free_transfers,
        "chips_remaining": sorted(state.chips_remaining),
        "mini_league_ids": list(state.mini_league_ids),
    }


def team_state_from_dict(data: dict) -> MyTeamState:
    return MyTeamState(
        squad=tuple(_squad_player_from_dict(p) for p in data["squad"]),
        starting_xi=tuple(data["starting_xi"]),
        bench_order=tuple(data["bench_order"]),
        captain_id=data["captain_id"],
        vice_captain_id=data["vice_captain_id"],
        bank=data["bank"],
        free_transfers=data["free_transfers"],
        chips_remaining=frozenset(data["chips_remaining"]),
        mini_league_ids=tuple(data.get("mini_league_ids", ())),
    )


def chip_usage_to_dict(usage: ChipUsage) -> dict:
    return {
        "first_half_played": sorted(usage.first_half_played),
        "second_half_played": sorted(usage.second_half_played),
    }


def chip_usage_from_dict(data: dict) -> ChipUsage:
    return ChipUsage(
        first_half_played=frozenset(data.get("first_half_played", ())),
        second_half_played=frozenset(data.get("second_half_played", ())),
    )


def pending_draft_to_dict(draft: PendingDraft) -> dict:
    return {
        "base_gameweek": draft.base_gameweek,
        "working_state": team_state_to_dict(draft.working_state),
        "transfers_made": draft.transfers_made,
        "chip": draft.chip,
    }


def pending_draft_from_dict(data: dict) -> PendingDraft:
    return PendingDraft(
        base_gameweek=data["base_gameweek"],
        working_state=team_state_from_dict(data["working_state"]),
        transfers_made=data["transfers_made"],
        chip=data.get("chip"),
    )


def save_squad_state(
    session: Session,
    season: str,
    committed: CommittedSquad,
    pending: PendingDraft | None,
) -> None:
    """Upsert the single ``id=1`` row — written on every successful draft mutation and on
    confirm/discard, so a pending draft (D17) and the committed squad both survive a restart."""
    row = session.get(SavedSquad, 1)
    if row is None:
        row = SavedSquad(id=1)
        session.add(row)

    row.season = season
    row.committed_gameweek = committed.committed_gameweek
    row.committed_state_json = (
        json.dumps(team_state_to_dict(committed.team_state))
        if committed.team_state is not None
        else None
    )
    row.chip_usage_json = json.dumps(chip_usage_to_dict(committed.chip_usage))
    row.active_chip = committed.active_chip
    row.active_chip_gameweek = committed.active_chip_gameweek
    row.free_hit_snapshot_json = (
        json.dumps(team_state_to_dict(committed.free_hit_snapshot))
        if committed.free_hit_snapshot is not None
        else None
    )
    row.free_hit_snapshot_gameweek = committed.free_hit_snapshot_gameweek
    row.pending_draft_json = (
        json.dumps(pending_draft_to_dict(pending)) if pending is not None else None
    )
    row.gameweek_hit_cost = committed.gameweek_hit_cost
    session.commit()


def load_squad_state(
    session: Session, season: str
) -> tuple[CommittedSquad, PendingDraft | None] | None:
    """Returns ``None`` if no squad has ever been saved for ``season`` — the caller should start a
    fresh build-mode :class:`CommittedSquad` (``team_state=None``) in that case."""
    row = session.execute(select(SavedSquad).where(SavedSquad.id == 1)).scalar_one_or_none()
    if row is None or row.season != season:
        return None

    team_state = (
        team_state_from_dict(json.loads(row.committed_state_json))
        if row.committed_state_json
        else None
    )
    free_hit_snapshot = (
        team_state_from_dict(json.loads(row.free_hit_snapshot_json))
        if row.free_hit_snapshot_json
        else None
    )
    committed = CommittedSquad(
        team_state=team_state,
        chip_usage=chip_usage_from_dict(json.loads(row.chip_usage_json)),
        active_chip=row.active_chip,
        active_chip_gameweek=row.active_chip_gameweek,
        free_hit_snapshot=free_hit_snapshot,
        free_hit_snapshot_gameweek=row.free_hit_snapshot_gameweek,
        committed_gameweek=row.committed_gameweek,
        gameweek_hit_cost=row.gameweek_hit_cost or 0,
    )
    pending = (
        pending_draft_from_dict(json.loads(row.pending_draft_json))
        if row.pending_draft_json
        else None
    )
    return committed, pending
