"""Shared "My Team" state (BUILD_PLAN Phase 4) — the one canonical object captaincy.py,
transfers.py, and chips.py all read from: current squad, starting XI/bench order, captain/vice,
bank, sell price per player, free transfers, and chips remaining.

Most of this the FPL API returns directly from a team ID. Sell price is the one field needing
careful transaction-history tracking (FPL's rule: a risen player only banks half the profit,
rounded down to the nearest GBP0.1m) — so that computation lives here as its own pure helper,
shared by :class:`SquadPlayer` and by transfers.py's sell/buy comparator, rather than being
reimplemented per feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.scoring import POSITIONS

__all__ = [
    "CHIPS",
    "SquadPlayer",
    "MyTeamState",
    "compute_sell_price",
]

# 2026/27: eight chips total, one full set (one of each) per half of the season (BUILD_PLAN 4).
# Which half's set is currently active/exhausted (the GW19-deadline reset) is the caller's job to
# track, not this state object's — `chips_remaining` just names what's playable right now.
CHIPS = ("wildcard", "free_hit", "triple_captain", "bench_boost")


def compute_sell_price(purchase_price: int, current_price: int) -> int:
    """FPL's sell-price rule: a player who has risen in price only banks HALF the profit, rounded
    DOWN to the nearest GBP0.1m; a player who is flat or has fallen sells at the current price,
    with no further penalty beyond the drop already reflected in ``current_price``.

    Both prices are in tenths of a million (FPL's own ``now_cost`` convention, e.g. ``105`` =
    GBP10.5m) so this stays exact integer arithmetic — "half a tenth, rounded down" is plain
    floor division by 2, with no float rounding drift across repeated recomputation.
    """
    profit = current_price - purchase_price
    if profit <= 0:
        return current_price
    return purchase_price + profit // 2


@dataclass(frozen=True)
class SquadPlayer:
    """One of the 15 squad players. Prices are tenths of a million, matching
    :func:`compute_sell_price`'s convention."""

    player_id: int
    position: str
    purchase_price: int
    current_price: int

    def __post_init__(self) -> None:
        if self.position not in POSITIONS:
            raise ValueError(f"unknown position: {self.position!r}")
        if self.purchase_price <= 0 or self.current_price <= 0:
            raise ValueError("prices must be positive")

    @property
    def sell_price(self) -> int:
        return compute_sell_price(self.purchase_price, self.current_price)


@dataclass(frozen=True)
class MyTeamState:
    """The shared state every Phase 4 decision feature (captaincy/transfers/chips) reads from.

    ``squad`` is the full 15; ``starting_xi``/``bench_order`` and ``captain_id``/
    ``vice_captain_id`` are player_ids drawn from it. ``chips_remaining`` names still-available
    chips this half-season (see :data:`CHIPS`).
    """

    squad: tuple[SquadPlayer, ...]
    starting_xi: tuple[int, ...]
    bench_order: tuple[int, ...]
    captain_id: int
    vice_captain_id: int
    bank: int  # tenths of a million
    free_transfers: int
    chips_remaining: frozenset[str]
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
        if self.bank < 0:
            raise ValueError("bank must be non-negative")
        if not 0 <= self.free_transfers <= 5:
            raise ValueError(
                "free_transfers must be between 0 and 5 (BUILD_PLAN 4: banked up to 5)"
            )
        unknown_chips = self.chips_remaining - set(CHIPS)
        if unknown_chips:
            raise ValueError(f"unknown chips in chips_remaining: {sorted(unknown_chips)}")

    @property
    def player_ids(self) -> tuple[int, ...]:
        return tuple(p.player_id for p in self.squad)

    def player(self, player_id: int) -> SquadPlayer:
        for squad_player in self.squad:
            if squad_player.player_id == player_id:
                return squad_player
        raise KeyError(player_id)

    @property
    def total_sell_value(self) -> int:
        """Squad value if every player were sold right now (tenths of a million) — the "how much
        can I actually raise" number transfers.py needs, distinct from the sum of current prices
        since sell price only banks half of any profit."""
        return sum(p.sell_price for p in self.squad)
