"""FPL <-> Understat player ID crosswalk (1.1) — "a real gap, not a nice-to-have".

FPL and Understat share no common ID scheme, and Understat player names don't always match FPL's
transliterations (accents, nicknames, Portuguese/Brazilian name forms especially). Every stats
component depends on correctly joining the two sources, and a silent mismatch doesn't crash
anything — it just quietly attributes one player's xG to another.

Approach (per BUILD_PLAN 1.1): start from the community-maintained
`vaastav/Fantasy-Premier-League <https://github.com/vaastav/Fantasy-Premier-League>`_ repo's
``player_idlist.csv`` (FPL's own full-legal-name spelling, hand-verified each season by that
project), match it against Understat's own ``player_name`` field by exact string equality — the
same technique that repo's own ``understat.py`` scraper uses to build its (unpublished, locally
generated) ``id_dict.csv`` — then fall back to an accent/case/whitespace-normalized comparison.

**ENGINE_IMPROVEMENTS_2.md C.1.** That two-pass approach alone matched only 52% of real 2025/26
outfield players, because Understat's ``player_name`` is almost always the **short display name**
("Bruno Fernandes"), not the full legal name ("Bruno Borges Fernandes") ``player_idlist.csv``
carries — and the misses concentrated in exactly the premium players captaincy decisions turn on.
Two more passes close most of that gap: an exact/normalized match against FPL's own ``web_name``
(pass ``fpl_id_by_web_name`` to :func:`build_crosswalk`, sourced from vaastav's ``players_raw.csv``
via :func:`fetch_fpl_web_names`), then a surname-token and a first-initial+surname match — both
restricted to **unique** candidates only, never an ambiguous best guess, since a wrong match
silently attributes one player's xG to another.

**Crosswalk coverage Phase 1** (a real 2025/26 pull, post-C.1: 60.5% matched, 37 unmatched
significant (>450-minute) players) found the dominant remaining pattern wasn't nicknames at all —
it was Spanish/Portuguese **maternal-surname drops** ("Marcos Senesi Barón" vs. Understat's
"Marcos Senesi"), affecting the large majority of those 37. Two more passes close most of that:
a **token-prefix** match (:func:`_token_prefix_candidates` — one name's tokens are an exact,
in-order extension of the other's) and a **reversed-name-order** match
(:func:`_reversed_name_candidate` — a given-name-first vs. family-name-first convention mismatch,
mostly Japanese/Korean names), each with the same unique-candidate-only discipline as the existing
heuristic passes. Hyphenation is also now normalized as a word separator (see
:func:`normalize_name`), since "Smith-Rowe" vs. "Smith Rowe" was otherwise invisible to every pass.
Together these took the real 2025/26 unmatched-significant count from 37 to 6 — the rest are
genuine nicknames or spelling variants with no safe automated pass ("Rodri" for "Rodrigo ...
Cascante", "Yeremi Pino" for "Yéremy Pino Santos") and went into the manual overlay table instead.

A small hand-maintained overlay table still covers the residual for the current season's new
signings/transfers, and any genuine nickname/spelling variant no automated pass can safely resolve.
Any Understat player still unmatched after every pass **fails loudly**: :func:`build_crosswalk`
raises rather than silently dropping it.

**The overlay is keyed by season** (:data:`MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON`), not a flat
``{understat_id: fpl_id}`` dict, even though ``understat_id`` itself is stable across seasons —
because ``fpl_id`` is **not**: FPL reassigns element ids every season (the same fact
``fetch_vaastav_teams`` already documents for team ids), so an entry hand-verified for one season
would silently point at a completely different real player in any other season. A real multi-season
pull caught this concretely: Understat displays *both* Emerson Palmieri (West Ham) and Emerson
Royal (Tottenham) as the bare string "Emerson" in 2022/23 — a genuine cross-player collision no
automated pass can resolve, needing an overlay entry — but that entry's ``fpl_id`` is only correct
for the 2022/23 season it was verified against.

That per-player failure only catches the "an Understat player has no FPL match" direction, though —
it says nothing about an FPL player who simply never got matched to any Understat id, which is the
direction that actually cost 23% of real season points. :func:`assert_matched_share` is a separate,
opt-in coverage check against a caller-supplied weighting (season minutes, points, ...) for exactly
that direction.
"""

from __future__ import annotations

import csv
import html
import io
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx

VAASTAV_RAW_BASE = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"

# Hand-maintained overlay for players Understat and the vaastav FPL id-list name spellings
# disagree on (nicknames, name-order swaps, transliteration choices, or a genuine cross-player
# collision no automated pass can resolve) that the exact/normalized matching passes can't resolve
# on their own. Keyed by **season start year** -> {Understat player id: FPL element id} — never a
# flat {understat_id: fpl_id} dict, since fpl_id is only meaningful within the one season it was
# verified against (see this module's own docstring for why). Update each transfer window per
# BUILD_PLAN 1.1; :func:`build_season_crosswalk` (backtest/run_season.py) looks up its own season's
# slice automatically.
MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON: dict[int, dict[int, int]] = {
    # Hand-verified against the real 2025/26 pull (crosswalk coverage Phase 1) after the
    # token-prefix/reversed-order passes and hyphen-as-separator normalization closed the rest of
    # that season's gap — every remaining significant (>450-minute) miss was one of these shapes:
    2025: {
        # Nickname is FPL's own designated-taker-style short form, not derivable from the legal name:
        10715: 673,  # Understat "João Palhinha" -> FPL "João Maria Lobo Alves Palhares Costa Palhinha Gonçalves"
        7365: 612,  # Understat "Lucas Paquetá" -> FPL "Lucas Tolentino Coelho de Lima"
        2496: 421,  # Understat "Rodri" -> FPL "Rodrigo 'Rodri' Hernandez Cascante" (nickname quoted in FPL's own name)
        11384: 646,  # Understat "João Gomes" -> FPL "João Victor Gomes da Silva" (drops both the middle
        # given name "Victor" and "da Silva" -- not a strict prefix, since "Victor" breaks the in-order
        # token match token_prefix requires)
        # Spelling/transliteration variants between the two sources for the same real person:
        9024: 712,  # Understat "Yeremi Pino" -> FPL "Yéremy Pino Santos"
        11772: 129,  # Understat "Yehor Yarmolyuk" -> FPL "Yehor Yarmoliuk"
        12168: 713,  # Understat "Alejandro Jiménez" -> FPL "Álex Jiménez Sánchez" (Alejandro/Álex)
    },
    # Hand-verified against a real multi-season backtest pull (ENGINE_IMPROVEMENTS_3.md multi-
    # season Phase 2): a genuine cross-player collision, not a name-spelling issue -- Understat
    # displays both Emerson Palmieri (West Ham) and Emerson Royal (Tottenham) as the bare string
    # "Emerson" in both 2022/23 and 2023/24 (both players stayed at the same clubs both seasons),
    # and FPL's own web_name "Emerson" belongs to Palmieri in both. Royal needs his own id given
    # explicitly each season (the ids themselves differ season to season, per this table's whole
    # reason for existing). Understat only lists one "Emerson" in 2024/25 (Royal isn't tracked
    # under that bare name that season) and 2025/26 (neither is), so no entry is needed there.
    2022: {
        7430: 445,  # Understat "Emerson" (Tottenham, understat_id=7430) -> FPL "Emerson Leite de Souza Junior" (Royal)
    },
    2023: {
        7430: 497,  # Understat "Emerson" (Tottenham, understat_id=7430) -> FPL "Emerson Leite de Souza Junior" (Royal)
    },
}

# Live-ingestion default (engine/data/ingest.py): always the one current/live season, so a flat
# dict is legitimate here (there is only ever one season in play), unlike the backtest driver's
# season-keyed lookup above. Empty until a real mismatch is observed for the live season and
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


class CrosswalkAmbiguityError(RuntimeError):
    """Raised when two different Understat players would both resolve to the same FPL id via a
    precise (exact/web_name/manual-overlay) match — that pass is supposed to be unambiguous by
    construction, so a collision there is a genuine data anomaly (a stale overlay entry, a
    duplicated name in one of the source lists) worth surfacing immediately rather than silently
    picking one player over the other."""

    def __init__(self, fpl_id: int, matched_by: str, first: UnderstatPlayer, second: UnderstatPlayer):
        self.fpl_id = fpl_id
        self.matched_by = matched_by
        self.players = (first, second)
        super().__init__(
            f"Understat players {first.name!r} (id={first.understat_id}) and {second.name!r} "
            f"(id={second.understat_id}) both matched FPL id {fpl_id} via a '{matched_by}' pass — "
            "this pass is supposed to be unambiguous; check for a stale manual-overlay entry or a "
            "duplicated name in the source data rather than picking one arbitrarily."
        )


class CrosswalkCoverageError(RuntimeError):
    """Raised by :func:`assert_matched_share` when the matched FPL id set covers too small a share
    of a caller-supplied weighting (ENGINE_IMPROVEMENTS_2.md C.1) — the direction
    :class:`CrosswalkError`'s per-player check can't see: an FPL player who was simply never
    matched to any Understat id at all, rather than an Understat player with no FPL match."""

    def __init__(self, covered_share: float, min_share: float, label: str):
        self.covered_share = covered_share
        self.min_share = min_share
        self.label = label
        super().__init__(
            f"crosswalk covers only {covered_share:.1%} of total {label} "
            f"(required at least {min_share:.1%})"
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
    # "exact" | "normalized" | "web_name_exact" | "web_name_normalized" | "surname_token" |
    # "initial_surname" | "token_prefix" | "reversed_order" | "manual_overlay"
    matched_by: str


# Letters NFKD decomposition can't reduce to base + combining accent, because they're a distinct
# letterform in Unicode, not a composed one -- so the plain "decompose, then drop non-ASCII" trick
# silently drops them instead of transliterating them (ENGINE_IMPROVEMENTS_2.md C.1: this measurably
# cost real matches for players like "Martin Ødegaard" and "Ferdi Kadıoğlu"). Extend as real misses
# turn up; this is deliberately small and hand-verified, not an attempt at general transliteration.
_NON_DECOMPOSABLE_TRANSLITERATIONS = str.maketrans(
    {
        "Ø": "O",
        "ø": "o",
        "Đ": "D",
        "đ": "d",
        "Ł": "L",
        "ł": "l",
        "ß": "ss",
        "Æ": "Ae",
        "æ": "ae",
        "Œ": "Oe",
        "œ": "oe",
        "ı": "i",  # Turkish dotless i
        "İ": "I",  # Turkish capital dotted I
    }
)


def normalize_name(name: str) -> str:
    """Unescape HTML entities, transliterate letters NFKD can't decompose, strip remaining
    accents/diacritics, treat hyphens as word separators, lowercase, collapse whitespace — for
    fuzzy-but-safe matching.

    Hyphens-as-separators (crosswalk coverage Phase 1) closes a real, systematic miss: Understat
    and FPL disagree on whether a hyphenated surname is one token or two ("Emile Smith-Rowe" vs
    "Emile Smith Rowe", "Dominic Solanke-Mitchell" vs "Dominic Solanke") — applied uniformly to
    both sides before any comparison, so this can only ever *split* a token, never merge two
    distinct names together.
    """
    unescaped = html.unescape(name)
    transliterated = unescaped.translate(_NON_DECOMPOSABLE_TRANSLITERATIONS)
    decomposed = unicodedata.normalize("NFKD", transliterated)
    ascii_only = decomposed.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_only.replace("-", " ").lower().split())


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


def fetch_fpl_web_names(season_start_year: int, client: httpx.Client) -> dict[str, int]:
    """FPL ``web_name`` (short display name, e.g. "Bruno Fernandes") -> FPL element id, from
    vaastav's ``players_raw.csv`` for this season (ENGINE_IMPROVEMENTS_2.md C.1). Understat's own
    ``player_name`` field is almost always this short form, not the full legal name
    :func:`fetch_fpl_id_list`'s ``player_idlist.csv`` carries — the two-pass exact/normalized match
    against the full name alone matched barely half of real 2025/26 outfield players.
    """
    label = season_to_vaastav_label(season_start_year)
    url = f"{VAASTAV_RAW_BASE}/{label}/players_raw.csv"
    response = client.get(url)
    response.raise_for_status()
    reader = csv.DictReader(io.StringIO(response.text))
    return {row["web_name"]: int(row["id"]) for row in reader}


def _surname_token(normalized_name: str) -> str | None:
    tokens = normalized_name.split()
    return tokens[-1] if tokens else None


def _initial_surname_key(normalized_name: str) -> str | None:
    tokens = normalized_name.split()
    return f"{tokens[0][0]} {tokens[-1]}" if tokens else None


def _token_prefix_candidates(
    understat_tokens: list[str], fpl_tokens_by_id: dict[int, list[str]]
) -> list[int]:
    """FPL ids whose normalized name is either an extension of ``understat_tokens`` by one or
    more *trailing* tokens, or vice versa (crosswalk coverage Phase 1). Understat's display name
    is almost always "first name + primary surname"; FPL's ``player_idlist.csv`` full legal name
    often carries one or more *additional* surnames after that (Spanish/Portuguese paternal+
    maternal surname convention: "Marcos Senesi Barón" vs Understat's "Marcos Senesi") — a plain
    surname-token match (:func:`_surname_token`, keyed on the *last* word) systematically misses
    this whole class, since the last word differs while the first N-1 do not. The reverse
    direction (Understat carries an extra trailing token FPL's doesn't, e.g. "Amad Diallo Traore"
    vs FPL's "Amad Diallo") is real too, so both directions are checked.
    """
    candidates = []
    for fpl_id, f_tokens in fpl_tokens_by_id.items():
        m, n = len(understat_tokens), len(f_tokens)
        if m <= n and f_tokens[:m] == understat_tokens:
            candidates.append(fpl_id)
        elif n < m and understat_tokens[:n] == f_tokens:
            candidates.append(fpl_id)
    return candidates


def _reversed_name_candidate(name: str, normalized_fpl_id_by_name: dict[str, int]) -> int | None:
    """FPL id for the exact match of ``name`` with its two whitespace-separated blocks swapped
    (crosswalk coverage Phase 1) — a given-name-first vs. family-name-first convention mismatch
    (e.g. Understat's "Kaoru Mitoma" vs FPL's "Mitoma Kaoru", "Ao Tanaka" vs "Tanaka Ao"), most
    often seen with Japanese and Korean names. Swaps on the **raw**, not hyphen-split, whitespace
    blocks so a hyphenated given name stays intact through the swap ("Hee-Chan Hwang" ->
    "Hwang Hee-Chan", which *then* normalizes — hyphens included — to match FPL's "Hwang
    Hee-chan"). Only applies to exactly two blocks; a three-plus-token name has no single
    unambiguous swap to try.
    """
    blocks = name.split()
    if len(blocks) != 2:
        return None
    reversed_name = " ".join(reversed(blocks))
    return normalized_fpl_id_by_name.get(normalize_name(reversed_name))


def _build_token_index(
    fpl_id_by_name: dict[str, int], key_fn: Callable[[str], str | None]
) -> dict[str, list[int]]:
    """``key_fn`` applied to each FPL name's normalized form -> the (possibly multiple) FPL ids
    sharing that key. Callers only accept a match here when exactly one candidate id shares the
    key — a coincidental surname/initial collision must never resolve ambiguously (this module's
    whole reason for existing is that a silent wrong match attributes one player's xG to
    another)."""
    index: dict[str, list[int]] = {}
    for name, fpl_id in fpl_id_by_name.items():
        key = key_fn(normalize_name(name))
        if key is not None:
            index.setdefault(key, []).append(fpl_id)
    return index


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
    fpl_id_by_web_name: dict[str, int] | None = None,
) -> list[CrosswalkEntry]:
    """Match every Understat player to an FPL id in **two rounds**, run to completion in order
    (never interleaved — this is what fixed a real ordering bug, see below).

    **Round 1 — precise passes**, in priority order: the manual overlay table **first** (a
    hand-verified answer always outranks an automatic pass — see crosswalk coverage Phase 1's
    "two different real players both display as bare 'Emerson' this season" case below for why
    this isn't just a residual fallback), then exact full-name match, normalized full-name match,
    exact ``web_name`` match, normalized ``web_name`` match (both only if ``fpl_id_by_web_name`` is
    supplied — see :func:`fetch_fpl_web_names`, ENGINE_IMPROVEMENTS_2.md C.1). Every pass after the
    overlay is supposed to be unambiguous by construction, so if two different Understat players
    would resolve to the same FPL id via any of them, that's a genuine data anomaly and raises
    :class:`CrosswalkAmbiguityError` immediately rather than guessing — a real 2022/23 case:
    Understat displays both Emerson Palmieri (West Ham) and Emerson Royal (Tottenham) as the bare
    string "Emerson" that season, and FPL's own ``web_name`` "Emerson" belongs to only one of them
    (Palmieri) — so ``web_name_exact`` resolved *both* Understat players to Palmieri's id until the
    overlay was moved ahead of it and given Royal's own correct id explicitly.

    **Round 2 — heuristic passes**, for whatever's left unresolved after round 1: a surname-token
    match, a first-initial+surname match, a token-prefix match, then a reversed-name-order match
    (crosswalk coverage Phase 1 added the latter two — see :func:`_token_prefix_candidates` and
    :func:`_reversed_name_candidate` for what each catches), each accepted only when it identifies
    a candidate that is both (a) **unique** among remaining FPL names sharing that key and (b)
    **not already claimed** by a round-1 precise match or an earlier round-2 winner — never an
    ambiguous best guess. Condition (b) is not redundant with (a): a real 2025/26 pull found two
    on-pitch defenders, "Cristhian Mosquera" and "Yerson Mosquera Valdelamar", where the *first*
    player's Understat display name is a single word ("Mosquera") that happens to equal the
    *second* player's own actual surname — so the naive single-pass version let whichever player
    was iterated first steal the other's rightful id via an unprotected heuristic match. Running
    every precise match to completion before any heuristic match is attempted, and refusing any
    heuristic candidate id that a precise match already owns, closes that hole regardless of
    iteration order.

    Raises :class:`CrosswalkError` for anything still unmatched after both rounds when ``strict``
    is True (the default — "fails loudly" per BUILD_PLAN 1.1). ``fpl_id_by_web_name`` defaults to
    ``None`` (no web_name pass at all), so every existing caller that only ever supplied the
    full-name list is unaffected.
    """
    normalized_fpl_id_by_name = {
        normalize_name(name): fpl_id for name, fpl_id in fpl_id_by_name.items()
    }
    fpl_name_by_id = {fpl_id: name for name, fpl_id in fpl_id_by_name.items()}

    fpl_id_by_web_name = fpl_id_by_web_name or {}
    normalized_fpl_id_by_web_name = {
        normalize_name(name): fpl_id for name, fpl_id in fpl_id_by_web_name.items()
    }
    surname_index = _build_token_index(fpl_id_by_name, _surname_token)
    initial_surname_index = _build_token_index(fpl_id_by_name, _initial_surname_key)
    fpl_tokens_by_id = {
        fpl_id: normalize_name(name).split() for name, fpl_id in fpl_id_by_name.items()
    }

    resolutions: dict[int, tuple[int, str]] = {}  # understat_id -> (fpl_id, matched_by)
    claimed_by: dict[int, UnderstatPlayer] = {}  # fpl_id -> the player that has claimed it so far
    unresolved_after_round1: list[UnderstatPlayer] = []

    # --- Round 1: precise, authoritative passes -----------------------------------------------
    for player in understat_players:
        normalized = normalize_name(player.name)
        precise_candidates: list[tuple[int | None, str]] = [
            # manual_overlay goes FIRST, not last: it's a hand-verified answer for this exact
            # understat_id, so it should always win over an automatic pass rather than only ever
            # filling a gap the automatic passes left. A real 2022/23 case forced this: Understat
            # displays *both* "Emerson Palmieri" (West Ham) and "Emerson Royal" (Tottenham) as the
            # bare string "Emerson" that season, so web_name_exact resolves either of them to
            # whichever one FPL's own web_name "Emerson" belongs to (Palmieri) — with overlay
            # checked last, there was no way to correct Royal's case without also breaking
            # Palmieri's already-correct one.
            (overlay.get(player.understat_id), "manual_overlay"),
            (fpl_id_by_name.get(player.name), "exact"),
            (normalized_fpl_id_by_name.get(normalized), "normalized"),
            (fpl_id_by_web_name.get(player.name), "web_name_exact"),
            (normalized_fpl_id_by_web_name.get(normalized), "web_name_normalized"),
        ]
        matched = False
        for candidate_id, candidate_pass in precise_candidates:
            if candidate_id is None:
                continue
            existing = claimed_by.get(candidate_id)
            if existing is not None and existing.understat_id != player.understat_id:
                raise CrosswalkAmbiguityError(candidate_id, candidate_pass, existing, player)
            claimed_by[candidate_id] = player
            resolutions[player.understat_id] = (candidate_id, candidate_pass)
            matched = True
            break
        if not matched:
            unresolved_after_round1.append(player)

    # --- Round 2: heuristic passes, only against ids no precise (or earlier heuristic) match ---
    # already owns -- a precise match elsewhere always outranks a heuristic guess.
    #
    # Order within round 2 matters, and it's deliberately *not* the order these passes were added
    # in: token_prefix and reversed_order each require the *entire* remaining name to line up (not
    # just one token), so they're tried before surname_token/initial_surname, which only look at a
    # single token and are more exposed to a coincidental cross-player surname collision. A real
    # 2025/26 case: "Diego Gómez" (Understat) has a maternal surname FPL drops ("Diego Gómez
    # Amarilla"), so the *correct* match's own surname-token key is "amarilla", not "gomez" — but
    # "Joe Gomez" (a real, different Liverpool defender) genuinely does key on "gomez", and was
    # the only *other* candidate left at that key, so a surname-token-first ordering picked that
    # wrong id even though it satisfied this pass's own "unique candidate" rule. Trying the
    # full-name passes first uses the stronger signal before the weaker one gets a chance to be
    # uniquely (and wrongly) available.
    still_unmatched: list[UnderstatPlayer] = []
    for player in unresolved_after_round1:
        normalized = normalize_name(player.name)
        heuristic_candidates: list[tuple[int, str]] = []

        prefix_candidates = [
            c
            for c in set(_token_prefix_candidates(normalized.split(), fpl_tokens_by_id))
            if c not in claimed_by
        ]
        if len(prefix_candidates) == 1:
            heuristic_candidates.append((prefix_candidates[0], "token_prefix"))

        reversed_candidate = _reversed_name_candidate(player.name, normalized_fpl_id_by_name)
        if reversed_candidate is not None and reversed_candidate not in claimed_by:
            heuristic_candidates.append((reversed_candidate, "reversed_order"))

        surname_candidates = [
            c for c in surname_index.get(_surname_token(normalized) or "", []) if c not in claimed_by
        ]
        if len(surname_candidates) == 1:
            heuristic_candidates.append((surname_candidates[0], "surname_token"))
        initial_candidates = [
            c
            for c in initial_surname_index.get(_initial_surname_key(normalized) or "", [])
            if c not in claimed_by
        ]
        if len(initial_candidates) == 1:
            heuristic_candidates.append((initial_candidates[0], "initial_surname"))

        matched = False
        for candidate_id, candidate_pass in heuristic_candidates:
            if candidate_id in claimed_by:  # claimed by an earlier winner within this same round
                continue
            claimed_by[candidate_id] = player
            resolutions[player.understat_id] = (candidate_id, candidate_pass)
            matched = True
            break
        if not matched:
            still_unmatched.append(player)

    entries = [
        CrosswalkEntry(
            fpl_id=resolutions[player.understat_id][0],
            understat_id=player.understat_id,
            fpl_name=fpl_name_by_id.get(resolutions[player.understat_id][0], ""),
            understat_name=player.name,
            matched_by=resolutions[player.understat_id][1],
        )
        for player in understat_players
        if player.understat_id in resolutions
    ]

    if still_unmatched and strict:
        raise CrosswalkError(still_unmatched)

    return entries


def assert_matched_share(
    fpl_id_weights: dict[int, float],
    matched_fpl_ids: set[int],
    min_share: float = 0.90,
    label: str = "minutes",
) -> None:
    """Coverage safety net independent of :func:`build_crosswalk`'s per-player ``strict`` check
    (ENGINE_IMPROVEMENTS_2.md C.1) — that check only catches an *Understat* player with no FPL
    match; it says nothing about an FPL player who was simply never matched to any Understat id at
    all, which cost 23% of real season points in the 2025/26 archive before this fix. Weights with
    a total of zero (or an empty mapping) are treated as vacuously covered, since there's nothing
    to fail on."""
    total = sum(fpl_id_weights.values())
    if total <= 0:
        return
    covered = sum(weight for fpl_id, weight in fpl_id_weights.items() if fpl_id in matched_fpl_ids)
    share = covered / total
    if share < min_share:
        raise CrosswalkCoverageError(share, min_share, label)


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
        overlay: dict[int, int] | None = None,
        strict: bool = True,
    ) -> list[CrosswalkEntry]:
        """``overlay`` defaults to this season's own slice of
        :data:`MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON` — never the flat live-ingestion default,
        since an fpl_id hand-verified for one season means a different real player in any other
        (see this module's own docstring). Pass an explicit ``overlay`` to override."""
        if overlay is None:
            overlay = MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON.get(season_start_year, {})
        fpl_id_by_name = fetch_fpl_id_list(season_start_year, self.client)
        understat_players = understat_players_from_league_data(league_data)
        return build_crosswalk(understat_players, fpl_id_by_name, overlay=overlay, strict=strict)
