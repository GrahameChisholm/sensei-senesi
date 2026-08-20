"""2026/27's real chip calendar (D14/G7): all four chips (Wildcard, Free Hit, Bench Boost, Triple
Captain), one full set per half of the season.

2026/27 grants **eight chips total, one full set of Wildcard, Free Hit, Bench Boost, and Triple
Captain independently per half of the season**, GW1 to GW19 and GW20 to GW38, real FPL's own
half-season split point. This is a different, and simpler, shape than the older 2025/26 format
(two Wildcards for the whole season, but only one each of Free Hit, Bench Boost, and Triple
Captain across the *entire* season, not per half); mixing the two up would silently under or over
grant a chip.
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

# GW19/20 boundary: real FPL's own half-season split point.
FIRST_HALF_LAST_GAMEWEEK = 19


@dataclass(frozen=True)
class ChipUsage:
    """Which of :data:`~features.team_state.CHIPS` have been played in each half of the season so
    far — independently, matching 2026/27's "one full set per half" rule (unlike the 2025/26
    module's asymmetric wildcard-count tracking, this needs no special-casing per chip: every chip
    is exactly "used or not used" within a half)."""

    first_half_played: frozenset[str] = frozenset()
    second_half_played: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        for label, played in (
            ("first_half_played", self.first_half_played),
            ("second_half_played", self.second_half_played),
        ):
            unknown = played - set(CHIPS)
            if unknown:
                raise ValueError(f"unknown chip(s) in {label}: {sorted(unknown)}")


def available_chips_this_gameweek(usage: ChipUsage, gameweek: int) -> frozenset[str]:
    """Every chip still playable in ``gameweek`` — all four minus whichever ones this
    gameweek's own half has already used. A chip used in the first half becomes available again
    once the second half starts (real FPL: no carryover, and no early expiry either — a chip
    unused in the first half is simply still available in the first half, right up to GW19)."""
    played = (
        usage.first_half_played
        if gameweek <= FIRST_HALF_LAST_GAMEWEEK
        else usage.second_half_played
    )
    return frozenset(CHIPS) - played


def record_chip_played(usage: ChipUsage, chip: str, gameweek: int) -> ChipUsage:
    """Mark ``chip`` used for whichever half ``gameweek`` falls in. Raises if it wasn't actually
    available (a caller bug — ``features.squad_draft.confirm_draft`` must check
    :func:`available_chips_this_gameweek` first, not a real-world state this should ever
    legitimately reach)."""
    if chip not in available_chips_this_gameweek(usage, gameweek):
        raise ValueError(f"chip {chip!r} is not available at gameweek {gameweek}")
    if gameweek <= FIRST_HALF_LAST_GAMEWEEK:
        return replace(usage, first_half_played=usage.first_half_played | {chip})
    return replace(usage, second_half_played=usage.second_half_played | {chip})
