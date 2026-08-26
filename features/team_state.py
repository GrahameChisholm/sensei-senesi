"""Shared "My Team" state — the one canonical object captaincy.py, squad_rules.py, and
squad_optimizer.py all read from: current squad, starting XI/bench order, and captain/vice.

The squad is a permanent sandbox (no confirm step, no transfer economy): a player's price is
always just their current price, there is no purchase-price/sell-price distinction, and there is
no free-transfer count or chip-usage tracking on this state at all. Bench Boost/Triple Captain are
passed as a plain argument to ``features.squad_points.projected_points`` when previewing points,
never stored here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.scoring import POSITIONS

__all__ = ["SquadPlayer", "MyTeamState"]


@dataclass(frozen=True)
class SquadPlayer:
    """One of the 15 squad players. ``price`` is tenths of a million (FPL's own ``now_cost``
    convention, e.g. ``105`` = £10.5m) and is always the player's current price — there is no
    purchase price to remember separately in a sandbox with no transfer economy."""

    player_id: int
    position: str
    price: int

    def __post_init__(self) -> None:
        if self.position not in POSITIONS:
            raise ValueError(f"unknown position: {self.position!r}")
        if self.price <= 0:
            raise ValueError("price must be positive")


@dataclass(frozen=True)
class MyTeamState:
    """A complete, legal 15/11 squad. ``squad`` is the full 15; ``starting_xi``/``bench_order`` and
    ``captain_id``/``vice_captain_id`` are player_ids drawn from it.

    A squad with fewer than 15 players (mid-build, or mid-edit after a removal) is never
    represented as a ``MyTeamState`` — it's a bare ``tuple[SquadPlayer, ...]`` instead, since this
    type's own invariants require exactly 15/11 at every step.
    """

    squad: tuple[SquadPlayer, ...]
    starting_xi: tuple[int, ...]
    bench_order: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    mini_league_ids: tuple[int, ...] = field(default=())

    def __post_init__(self) -> None:
        if len(self.squad) != 15:
            raise ValueError(f"squad must have exactly 15 players, got {len(self.squad)}")
        squad_ids = {p.player_id for p in self.squad}
        if len(squad_ids) != len(self.squad):
            raise ValueError("squad contains duplicate player_ids")
        if len(self.starting_xi) != 11:
            raise ValueError(
                f"starting_xi must have exactly 11 players, got {len(self.starting_xi)}"
            )
        if set(self.starting_xi) & set(self.bench_order):
            raise ValueError("starting_xi and bench_order must not overlap")
        if set(self.starting_xi) | set(self.bench_order) != squad_ids:
            raise ValueError("starting_xi + bench_order must exactly partition squad")
        if self.captain_id not in self.starting_xi:
            raise ValueError("captain_id must be in starting_xi")
        if self.vice_captain_id not in self.starting_xi:
            raise ValueError("vice_captain_id must be in starting_xi")
        if self.captain_id == self.vice_captain_id:
            raise ValueError("captain_id and vice_captain_id must differ")

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(p.player_id for p in self.squad)

    def player(self, player_id: int) -> SquadPlayer:
        for squad_player in self.squad:
            if squad_player.player_id == player_id:
                return squad_player
        raise KeyError(player_id)
