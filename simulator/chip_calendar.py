"""2025/26's real chip calendar (``planning/SEASON_SIMULATOR.md``'s own open item: "verify the
real 2025/26 chip allowance/windows before implementing... which may not match" ``features/
chips.py``'s rules).

Confirmed live (2026-07-31, via the real FPL ``bootstrap-static`` endpoint, which at the time of
writing already reflects the *upcoming* 2026/27 season): 2026/27 grants eight chips, one full set
(Wildcard, Free Hit, Bench Boost, Triple Captain) per half of the season — exactly what
``features/chips.py`` already encodes. 2025/26 and every season before it used the long-standing
traditional format instead: **two** Wildcards (one playable in each half, same GW19/GW20
boundary), but only **one** Free Hit, one Bench Boost, and one Triple Captain across the *entire*
season — 2026/27's per-half doubling of Free Hit/Bench Boost/Triple Captain is a new rule, not
something 2025/26 also had.

This module tracks that older format's usage separately from ``features.team_state.MyTeamState
.chips_remaining`` (a flat frozenset with no concept of "1 of 2 wildcards left"), since the
simulator builds a fresh ``MyTeamState`` every gameweek anyway (that dataclass is frozen) —
:func:`available_chips_this_gameweek` derives whatever ``chips_remaining`` set is legal *right
now* from this module's own richer usage-count state, for ``MyTeamState`` construction and for
``features.chips``'s existing evaluators to read.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from features.team_state import CHIPS

__all__ = [
    "FIRST_HALF_LAST_GAMEWEEK",
    "ChipUsage",
    "available_chips_this_gameweek",
    "record_chip_played",
]

FIRST_HALF_LAST_GAMEWEEK = 19


@dataclass(frozen=True)
class ChipUsage:
    """How many of each 2025/26 chip have been played so far. ``wildcards_played`` counts both
    halves together (0, 1, or 2); the other three are one-shot for the whole season."""

    wildcards_played: int = 0
    free_hit_played: bool = False
    bench_boost_played: bool = False
    triple_captain_played: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.wildcards_played <= 2:
            raise ValueError("wildcards_played must be between 0 and 2")


def available_chips_this_gameweek(usage: ChipUsage, gameweek: int) -> frozenset[str]:
    """Which of :data:`features.team_state.CHIPS` are legally playable *this* gameweek under
    2025/26's rules — a fresh derivation every call, matching the immutable ``MyTeamState`` the
    simulator rebuilds each week rather than a stateful toggle. A never-used first-half Wildcard
    simply expires at the GW19/20 boundary (real FPL: no carryover) — this needs no explicit expiry
    step since the formula below only ever checks the *current* half's own usage count.
    """
    available = set()
    wildcard_used_this_half = (
        usage.wildcards_played >= 1
        if gameweek <= FIRST_HALF_LAST_GAMEWEEK
        else usage.wildcards_played >= 2
    )
    if not wildcard_used_this_half:
        available.add("wildcard")
    if not usage.free_hit_played:
        available.add("free_hit")
    if not usage.bench_boost_played:
        available.add("bench_boost")
    if not usage.triple_captain_played:
        available.add("triple_captain")
    return frozenset(available) & set(CHIPS)


def record_chip_played(usage: ChipUsage, chip: str) -> ChipUsage:
    """Advance ``usage`` after ``chip`` is played this gameweek — raises if it wasn't actually
    available (a caller bug: the season loop must check :func:`available_chips_this_gameweek`
    first, not a real-world state this should ever legitimately reach)."""
    if chip == "wildcard":
        return replace(usage, wildcards_played=usage.wildcards_played + 1)
    if chip == "free_hit":
        if usage.free_hit_played:
            raise ValueError("free_hit already played this season")
        return replace(usage, free_hit_played=True)
    if chip == "bench_boost":
        if usage.bench_boost_played:
            raise ValueError("bench_boost already played this season")
        return replace(usage, bench_boost_played=True)
    if chip == "triple_captain":
        if usage.triple_captain_played:
            raise ValueError("triple_captain already played this season")
        return replace(usage, triple_captain_played=True)
    raise ValueError(f"unknown chip: {chip!r}")
