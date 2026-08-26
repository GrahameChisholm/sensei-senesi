"""Build a real ``features.team_state.MyTeamState`` from FPL manager (entry) API responses (A3).

Feeds directly off :class:`~engine.data.fpl_client.FPLClient`'s four entry-endpoint methods —
``get_entry``, ``get_entry_picks``, ``get_entry_transfers``, ``get_entry_history`` — plus the
bootstrap ``elements`` table already fetched for every other live purpose. No new HTTP surface
beyond those four methods.

**Two fields have no direct live source and are reconstructed, not read** — see
:func:`compute_purchase_prices` and :func:`compute_free_transfers`'s own docstrings for exactly
what each approximates and why: FPL's public API genuinely does not expose either "the price this
squad member was originally bought at, if never transferred since" or "free transfers currently
banked" as a queryable field. Both are UNVERIFIED against a real live pull (no network access in
this environment) — callers with a more reliable source should override them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from engine.scoring import ELEMENT_TYPE_TO_POSITION
from features.chip_calendar import FPL_CHIP_CODE_TO_NAME
from features.team_state import CHIPS, MyTeamState, SquadPlayer

__all__ = [
    "DEFAULT_MAX_FREE_TRANSFERS",
    "DEFAULT_FIRST_HALF_LAST_GAMEWEEK",
    "compute_purchase_prices",
    "compute_free_transfers",
    "compute_chips_remaining",
    "build_my_team_state",
]

DEFAULT_MAX_FREE_TRANSFERS = 5
# 2026/27 splits the season into two halves for chip purposes (BUILD_PLAN 4: "one full set per
# half"); the exact boundary gameweek is a fixture-calendar fact this module has no live source
# for, so it's a parameter with a plausible default rather than a hardcoded assumption.
DEFAULT_FIRST_HALF_LAST_GAMEWEEK = 19


def compute_purchase_prices(
    picks: Sequence[Mapping[str, Any]],
    transfers: Sequence[Mapping[str, Any]],
    now_cost_by_id: Mapping[int, int],
) -> dict[int, int]:
    """Purchase price (tenths of a million) for every player in ``picks``.

    Reconstructed from ``transfers``' most recent ``element_in_cost`` for that player, if a
    transfer-in record exists this season (ties broken by the later ``event``). A player still in
    the squad with **no** transfer-in record has never been transferred in this season — they've
    been there since the initial pre-GW1 squad pick, which
    :meth:`~engine.data.fpl_client.FPLClient.get_entry_transfers` does not cover at all (see that
    method's own docstring). For that player, this falls back to their **current** price
    (``now_cost_by_id``) as the best available live approximation: it understates a player who
    has since risen in price, overstates one who has fallen, and there is no live source for their
    true GW1 price without an archived GW1 snapshot this environment doesn't have for the current
    season. Every ``picks`` entry must have a current price in ``now_cost_by_id`` — a missing one
    is a real data-integrity problem (a squad player bootstrap doesn't know about) worth raising
    on, not silently guessing at.
    """
    most_recent_buy: dict[int, tuple[int, int]] = {}
    for transfer in transfers:
        player_id = int(transfer["element_in"])
        event = int(transfer["event"])
        cost = int(transfer["element_in_cost"])
        current = most_recent_buy.get(player_id)
        if current is None or event >= current[0]:
            most_recent_buy[player_id] = (event, cost)

    prices: dict[int, int] = {}
    for pick in picks:
        player_id = int(pick["element"])
        if player_id in most_recent_buy:
            prices[player_id] = most_recent_buy[player_id][1]
        elif player_id in now_cost_by_id:
            prices[player_id] = int(now_cost_by_id[player_id])
        else:
            raise KeyError(
                f"player_id {player_id} is in the squad but missing from now_cost_by_id — "
                "the bootstrap elements table and this squad's picks are out of sync"
            )
    return prices


def compute_free_transfers(
    history_current: Sequence[Mapping[str, Any]],
    max_free_transfers: int = DEFAULT_MAX_FREE_TRANSFERS,
) -> int:
    """Best-effort reconstruction of free transfers currently banked, from
    :meth:`~engine.data.fpl_client.FPLClient.get_entry_history`'s own ``current`` list (one row per
    gameweek played, each carrying ``event_transfers``).

    Implements FPL's 2024/25+ banking rule as understood — 1 free transfer gained per gameweek
    played, banked up to ``max_free_transfers``, one consumed per transfer made, never going
    negative going into the next gameweek's decision — starting from a baseline of 1 (GW1's
    "first transfer is free" state). **Not verified against a real manager's actual banked
    count** (no live network access in this environment); a caller with a more reliable source
    (or who knows this manager played a chip that doesn't consume transfers a given week) should
    override the result rather than trust this blindly.
    """
    free = 1
    for gameweek in history_current:
        transfers_made = int(gameweek.get("event_transfers", 0))
        free = max(0, free - transfers_made)
        free = min(max_free_transfers, free + 1)
    return free


def compute_chips_remaining(
    chips_played: Sequence[Mapping[str, Any]],
    current_gameweek: int,
    first_half_last_gameweek: int = DEFAULT_FIRST_HALF_LAST_GAMEWEEK,
) -> frozenset[str]:
    """Which chips are still playable this half-season, from
    :meth:`~engine.data.fpl_client.FPLClient.get_entry_history`'s own ``chips`` list (one entry
    per chip already played this season, each carrying ``name``/``event``, ``name`` in FPL's own
    raw codes translated through :data:`~features.chip_calendar.FPL_CHIP_CODE_TO_NAME`) —
    BUILD_PLAN 4's "one full set per half", resetting at ``first_half_last_gameweek``.
    """
    half_start = 1 if current_gameweek <= first_half_last_gameweek else first_half_last_gameweek + 1
    played_this_half = {
        FPL_CHIP_CODE_TO_NAME.get(chip["name"], chip["name"])
        for chip in chips_played
        if int(chip["event"]) >= half_start
    }
    return frozenset(CHIPS) - played_this_half


def build_my_team_state(
    picks: Mapping[str, Any],
    entry: Mapping[str, Any],
    transfers: Sequence[Mapping[str, Any]],
    history: Mapping[str, Any],
    elements: pd.DataFrame,
    current_gameweek: int,
    free_transfers: int | None = None,
    first_half_last_gameweek: int = DEFAULT_FIRST_HALF_LAST_GAMEWEEK,
) -> MyTeamState:
    """Assemble a real :class:`~features.team_state.MyTeamState` from
    :class:`~engine.data.fpl_client.FPLClient`'s entry-endpoint responses and the bootstrap
    ``elements`` table.

    ``picks``/``entry``/``history`` are exactly ``get_entry_picks``/``get_entry``/
    ``get_entry_history``'s own return shapes; ``transfers`` is ``get_entry_transfers``'s list.
    ``free_transfers`` defaults to :func:`compute_free_transfers`'s reconstruction from
    ``history["current"]`` — pass an explicit value to override it (see that function's own
    accuracy caveat).
    """
    now_cost_by_id = dict(zip(elements["id"], elements["now_cost"], strict=True))
    element_type_by_id = dict(zip(elements["id"], elements["element_type"], strict=True))
    purchase_prices = compute_purchase_prices(picks["picks"], transfers, now_cost_by_id)

    squad = tuple(
        SquadPlayer(
            player_id=int(pick["element"]),
            position=ELEMENT_TYPE_TO_POSITION[int(element_type_by_id[int(pick["element"])])],
            purchase_price=purchase_prices[int(pick["element"])],
            current_price=int(now_cost_by_id[int(pick["element"])]),
        )
        for pick in picks["picks"]
    )
    starting_xi = tuple(
        int(pick["element"]) for pick in picks["picks"] if int(pick["position"]) <= 11
    )
    bench_order = tuple(
        int(pick["element"])
        for pick in sorted(
            (p for p in picks["picks"] if int(p["position"]) > 11),
            key=lambda p: int(p["position"]),
        )
    )
    captain_id = next(int(p["element"]) for p in picks["picks"] if p["is_captain"])
    vice_captain_id = next(int(p["element"]) for p in picks["picks"] if p["is_vice_captain"])

    resolved_free_transfers = (
        free_transfers
        if free_transfers is not None
        else compute_free_transfers(history.get("current", []))
    )
    chips_remaining = compute_chips_remaining(
        history.get("chips", []), current_gameweek, first_half_last_gameweek
    )

    return MyTeamState(
        squad=squad,
        starting_xi=starting_xi,
        bench_order=bench_order,
        captain_id=captain_id,
        vice_captain_id=vice_captain_id,
        bank=int(entry["last_deadline_bank"]),
        free_transfers=resolved_free_transfers,
        chips_remaining=chips_remaining,
    )
