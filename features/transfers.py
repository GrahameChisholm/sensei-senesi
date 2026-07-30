"""Transfers: multi-gameweek greedy sell/buy comparator, net of hits, with sell-price handling
(BUILD_PLAN Phase 4).

**v1 scope cut, per BUILD_PLAN 4:** a greedy one-swap-at-a-time comparator, not a full
multi-transfer combinatorial search — each sell/buy pair is evaluated in isolation rather than
searching over which *combination* of several simultaneous transfers maximizes total expected
points across the horizon. Revisit with real multi-transfer search only if real use shows the
greedy version is missing genuinely valuable multi-move plans.

**Forced vs optional, derived from the engine itself.** Rather than requiring a separate
injury/suspension data feed, a sold player is flagged "forced" when the minutes model's own
P(0 minutes) (BUILD_PLAN 2.1 — already fed by FPL's ``status``/``chance_of_playing_next_round``)
stays high across the whole horizon: the engine already believes this player won't feature, which
is exactly what "forced" means for transfer purposes.

**Sell-price handling.** Reuses ``features.team_state.SquadPlayer.sell_price`` (FPL's
half-the-profit-rounded-down rule) rather than reimplementing it — the sell side of every
candidate here is always that number, never the player's current price.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from engine.projections import PlayerHorizonProjection
from features.team_state import MyTeamState

__all__ = [
    "TRANSFER_HIT_COST",
    "UNAVAILABLE_P_ZERO_THRESHOLD",
    "TransferCandidate",
    "TransferPlan",
    "evaluate_transfer",
    "find_transfer_candidates",
]

# Points deducted for a transfer beyond however many free transfers are banked (BUILD_PLAN 4).
# A team-management rule, not a match-scoring constant -- deliberately kept out of
# engine/scoring.py, which is FPL's per-match points rules only.
TRANSFER_HIT_COST = 4

# A squad player whose average P(0 minutes) across the horizon clears this bar is flagged
# "forced" -- the minutes model already believes they're unlikely to feature at all.
UNAVAILABLE_P_ZERO_THRESHOLD = 0.5


def _average_p_zero(horizon: PlayerHorizonProjection) -> float:
    p_zeros = [projection.minutes.p_zero for projection in horizon.gameweeks.values()]
    return sum(p_zeros) / len(p_zeros)


def _is_forced(horizon: PlayerHorizonProjection) -> bool:
    return _average_p_zero(horizon) >= UNAVAILABLE_P_ZERO_THRESHOLD


@dataclass(frozen=True)
class TransferCandidate:
    """One sell/buy pairing, evaluated in isolation (BUILD_PLAN 4's greedy v1 scope)."""

    sell_player_id: int
    buy_player_id: int
    position: str
    sell_price: int  # tenths of a million, proceeds from selling
    buy_price: int  # tenths of a million, cost of buying
    net_spend: int  # buy_price - sell_price; negative means this swap banks money
    horizon_points_sold: float
    horizon_points_bought: float
    points_gain: float  # horizon_points_bought - horizon_points_sold
    hit_cost: int  # 0 if a free transfer is available, else TRANSFER_HIT_COST
    is_forced: bool  # sold player's own minutes model says they're unlikely to feature at all
    reasoning: str

    @property
    def net_points_gain(self) -> float:
        return self.points_gain - self.hit_cost


def evaluate_transfer(
    my_team: MyTeamState,
    sell_player_id: int,
    buy_player_id: int,
    buy_price: int,
    sold_horizon: PlayerHorizonProjection,
    bought_horizon: PlayerHorizonProjection,
) -> TransferCandidate:
    """Evaluate one candidate sell/buy swap in isolation."""
    if sold_horizon.position != bought_horizon.position:
        raise ValueError(
            f"position mismatch: selling {sold_horizon.position!r}, buying "
            f"{bought_horizon.position!r} -- a single swap must replace like-for-like position"
        )
    sell_price = my_team.player(sell_player_id).sell_price
    hit_cost = 0 if my_team.free_transfers > 0 else TRANSFER_HIT_COST
    points_gain = bought_horizon.horizon_total_points - sold_horizon.horizon_total_points
    is_forced = _is_forced(sold_horizon)
    net_spend = buy_price - sell_price

    reasoning_parts = [
        f"{points_gain:+.1f} pts over the horizon ({sold_horizon.horizon_total_points:.1f} -> "
        f"{bought_horizon.horizon_total_points:.1f})"
    ]
    if hit_cost:
        reasoning_parts.append(f"net {points_gain - hit_cost:+.1f} after the -{hit_cost} hit")
    reasoning_parts.append(f"net spend {net_spend / 10:+.1f}m")
    if is_forced:
        reasoning_parts.append(
            "sold player's own minutes model says they're unlikely to feature -- forced, not "
            "optional"
        )
    reasoning = "; ".join(reasoning_parts)

    return TransferCandidate(
        sell_player_id=sell_player_id,
        buy_player_id=buy_player_id,
        position=sold_horizon.position,
        sell_price=sell_price,
        buy_price=buy_price,
        net_spend=net_spend,
        horizon_points_sold=sold_horizon.horizon_total_points,
        horizon_points_bought=bought_horizon.horizon_total_points,
        points_gain=points_gain,
        hit_cost=hit_cost,
        is_forced=is_forced,
        reasoning=reasoning,
    )


@dataclass(frozen=True)
class TransferPlan:
    """Every affordable one-swap candidate, ranked best ``net_points_gain`` first, plus a single
    ``recommended`` pick: the best forced-sell swap if any squad player looks unavailable,
    otherwise the best net-positive optional upgrade, otherwise ``None`` (no transfer earns its
    keep this week)."""

    affordable_candidates: tuple[TransferCandidate, ...]
    recommended: TransferCandidate | None


def find_transfer_candidates(
    my_team: MyTeamState,
    current_projections: Mapping[int, PlayerHorizonProjection],
    candidate_pool: Mapping[int, PlayerHorizonProjection],
    buy_prices: Mapping[int, int],
) -> TransferPlan:
    """Evaluate every affordable, position-matched one-swap candidate: each of the 15 owned
    players against every player in ``candidate_pool`` not already owned.

    ``current_projections``/``candidate_pool`` are ``player_id -> horizon projection`` maps (own
    squad and the wider pool respectively). ``buy_prices`` gives each ``candidate_pool`` player's
    current buy-in price (tenths of a million); sell-side prices come from ``my_team`` itself via
    :attr:`~features.team_state.SquadPlayer.sell_price`.
    """
    owned_ids = set(my_team.player_ids)
    budget_by_sell_id = {
        squad_player.player_id: my_team.bank + squad_player.sell_price
        for squad_player in my_team.squad
    }

    candidates: list[TransferCandidate] = []
    for sell_player_id, sold_horizon in current_projections.items():
        if sell_player_id not in owned_ids:
            continue
        budget = budget_by_sell_id[sell_player_id]
        for buy_player_id, bought_horizon in candidate_pool.items():
            if buy_player_id in owned_ids:
                continue
            if bought_horizon.position != sold_horizon.position:
                continue
            buy_price = buy_prices.get(buy_player_id)
            if buy_price is None or buy_price > budget:
                continue
            candidates.append(
                evaluate_transfer(
                    my_team,
                    sell_player_id,
                    buy_player_id,
                    buy_price,
                    sold_horizon,
                    bought_horizon,
                )
            )

    ranked = tuple(sorted(candidates, key=lambda c: c.net_points_gain, reverse=True))

    forced = [c for c in ranked if c.is_forced]
    if forced:
        recommended = max(forced, key=lambda c: c.net_points_gain)
    else:
        best = ranked[0] if ranked else None
        recommended = best if best is not None and best.net_points_gain > 0 else None

    return TransferPlan(affordable_candidates=ranked, recommended=recommended)
