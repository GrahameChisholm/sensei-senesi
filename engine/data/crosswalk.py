"""FPL <-> Understat player ID crosswalk (1.1) — "a real gap, not a nice-to-have".

FPL and Understat share no common ID scheme, and Understat player names don't always match FPL's
transliterations (accents, nicknames, Portuguese/Brazilian name forms especially). Every stats
component depends on correctly joining the two sources, and a silent mismatch doesn't crash
anything — it just quietly attributes one player's xG to another.

Approach (per BUILD_PLAN 1.1): start from the community-maintained
`vaastav/Fantasy-Premier-League <https://github.com/vaastav/Fantasy-Premier-League>`_ repo's
``player_idlist.csv`` (FPL's own name spelling, hand-verified each season by that project), match
it against Understat's own ``player_name`` field by exact string equality — the same technique
that repo's own ``understat.py`` scraper uses to build its (unpublished, locally generated)
``id_dict.csv`` — then fall back to an accent/case/whitespace-normalized comparison, then a small
hand-maintained overlay table for the current season's new signings/transfers. Any Understat
player still unmatched after all three passes **fails loudly**: :func:`build_crosswalk` raises
rather than silently dropping it.
"""

from __future__ import annotations

import csv
import html
import io
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import httpx

VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Hand-maintained overlay for players Understat and the vaastav FPL id-list name spellings
# disagree on (nicknames, name-order swaps, transliteration choices) that the exact/normalized
# matching passes can't resolve on their own. Keyed by Understat player id -> FPL element id.
# Update each transfer window per BUILD_PLAN 1.1. Empty until a real mismatch is observed and
# hand-verified — don't pre-guess entries.
MANUAL_OVERLAY_UNDERSTAT_TO_FPL: dict[int, int] = {}


class CrosswalkError(RuntimeError):
    """Raised when one or more Understat players can't be matched to an FPL id."""

    def __init__(self, unmatched: list[UnderstatPlayer]):
        self.unmatched = unmatched
        names = ", ".join(f"{p.name!r} (understat_id={p.understat_id})" for p in unmatched)
        super().__init__(
            f"{len(unmatched)} Understat player(s) could not be matched to an FPL id: {names}. "
            "Add a manual overlay entry (MANUAL_OVERLAY_UNDERSTAT_TO_FPL) rather than dropping "
            "them silently."
        )


@dataclass(frozen=True)
class UnderstatPlayer:
    understat_id: int
    name: str


@dataclass(frozen=True)
class CrosswalkEntry:
    fpl_id: int
    understat_id: int
    fpl_name: str
    understat_name: str
    matched_by: str  # "exact" | "normalized" | "manual_overlay"


def normalize_name(name: str) -> str:
    """Unescape HTML entities, strip accents/diacritics, lowercase, collapse whitespace — for
    fuzzy-but-safe matching."""
    unescaped = html.unescape(name)
    decomposed = unicodedata.normalize("NFKD", unescaped)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.lower().split())


def season_to_vaastav_label(season_start_year: int) -> str:
    """2024 -> '2024-25', matching the vaastav repo's directory naming."""
    return f"{season_start_year}-{str(season_start_year + 1)[-2:]}"


def fetch_fpl_id_list(season_start_year: int, client: httpx.Client) -> dict[str, int]:
    """FPL name -> FPL element id, from vaastav's hand-verified ``player_idlist.csv``.

    Returns a mapping keyed by ``"{first_name} {second_name}"`` exactly as FPL spells it.
    """
    label = season_to_vaastav_label(season_start_year)
    url = f"{VAASTAV_RAW_BASE}/{label}/player_idlist.csv"
    response = client.get(url)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    return {f"{row['first_name']} {row['second_name']}": int(row["id"]) for row in reader}


def understat_players_from_league_data(league_data: dict[str, Any]) -> list[UnderstatPlayer]:
    """Extract the (id, name) pairs from an UnderstatClient.get_league_data() payload."""
    return [
        UnderstatPlayer(understat_id=int(row["id"]), name=row["player_name"])
        for row in league_data["players"]
    ]


def build_crosswalk(
    understat_players: list[UnderstatPlayer],
    fpl_id_by_name: dict[str, int],
    overlay: dict[int, int] = MANUAL_OVERLAY_UNDERSTAT_TO_FPL,
    strict: bool = True,
) -> list[CrosswalkEntry]:
    """Match every Understat player to an FPL id via exact name, then normalized name, then the
    manual overlay table. Raises :class:`CrosswalkError` for anything still unmatched when
    ``strict`` is True (the default — "fails loudly" per BUILD_PLAN 1.1).
    """
    normalized_fpl_id_by_name = {
        normalize_name(name): fpl_id for name, fpl_id in fpl_id_by_name.items()
    }
    fpl_name_by_id = {fpl_id: name for name, fpl_id in fpl_id_by_name.items()}

    entries: list[CrosswalkEntry] = []
    unmatched: list[UnderstatPlayer] = []

    for player in understat_players:
        fpl_id = fpl_id_by_name.get(player.name)
        matched_by = "exact"
        if fpl_id is None:
            fpl_id = normalized_fpl_id_by_name.get(normalize_name(player.name))
            matched_by = "normalized"
        if fpl_id is None:
            fpl_id = overlay.get(player.understat_id)
            matched_by = "manual_overlay"
        if fpl_id is None:
            unmatched.append(player)
            continue
        entries.append(
            CrosswalkEntry(
                fpl_id=fpl_id,
                understat_id=player.understat_id,
                fpl_name=fpl_name_by_id.get(fpl_id, ""),
                understat_name=player.name,
                matched_by=matched_by,
            )
        )

    if unmatched and strict:
        raise CrosswalkError(unmatched)

    return entries


@dataclass
class CrosswalkBuilder:
    """Convenience: fetch + build in one call, with an injectable ``httpx.Client`` for tests."""

    client: httpx.Client = field(default_factory=httpx.Client)

    def __enter__(self) -> CrosswalkBuilder:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.client.close()

    def build_for_season(
        self,
        season_start_year: int,
        league_data: dict[str, Any],
        overlay: dict[int, int] = MANUAL_OVERLAY_UNDERSTAT_TO_FPL,
        strict: bool = True,
    ) -> list[CrosswalkEntry]:
        fpl_id_by_name = fetch_fpl_id_list(season_start_year, self.client)
        understat_players = understat_players_from_league_data(league_data)
        return build_crosswalk(understat_players, fpl_id_by_name, overlay=overlay, strict=strict)
