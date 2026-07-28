"""Tests for engine.data.crosswalk.

Uses the real (trimmed) vaastav idlist + Understat 2023/24 league player fixtures, so the
match/mismatch behaviour exercised here reflects the genuine name-mismatch gap the module exists
to handle, not a synthetic best case.
"""

import csv
import json
from pathlib import Path

import httpx
import pytest

from engine.data.crosswalk import (
    CrosswalkAmbiguityError,
    CrosswalkBuilder,
    CrosswalkCoverageError,
    CrosswalkError,
    UnderstatPlayer,
    assert_matched_share,
    build_crosswalk,
    fetch_fpl_id_list,
    fetch_fpl_web_names,
    normalize_name,
    season_to_vaastav_label,
    understat_players_from_league_data,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
LEAGUE_DATA = json.loads((FIXTURES_DIR / "understat_league_data.json").read_text())
IDLIST_CSV = (FIXTURES_DIR / "vaastav_player_idlist.csv").read_text()


def _fpl_id_by_name() -> dict[str, int]:
    reader = csv.DictReader(IDLIST_CSV.splitlines())
    return {f"{row['first_name']} {row['second_name']}": int(row["id"]) for row in reader}


def test_season_to_vaastav_label():
    assert season_to_vaastav_label(2024) == "2024-25"
    assert season_to_vaastav_label(2009) == "2009-10"


def test_normalize_name_strips_accents_case_and_entities():
    assert normalize_name("Bruno Guimarães") == normalize_name("bruno guimaraes")
    assert normalize_name("Dara O&#039;Shea") == "dara o'shea"
    assert normalize_name("  Extra   Space ") == "extra space"


def test_normalize_name_transliterates_letters_nfkd_cannot_decompose():
    # ENGINE_IMPROVEMENTS_2.md C.1: NFKD decomposition can't reduce these to base + accent because
    # they're distinct Unicode letterforms, not composed ones -- the plain accent-strip previously
    # just dropped them, costing real matches (e.g. "Martin Ødegaard", "Ferdi Kadıoğlu").
    assert normalize_name("Martin Ødegaard") == normalize_name("Martin Odegaard")
    assert normalize_name("Ferdi Kadıoğlu") == normalize_name("Ferdi Kadioglu")


def test_understat_players_from_league_data_extracts_id_and_name():
    players = understat_players_from_league_data(LEAGUE_DATA)
    assert len(players) == 50
    assert all(isinstance(p, UnderstatPlayer) for p in players)
    names = {p.name for p in players}
    assert "Bukayo Saka" in names


def test_build_crosswalk_matches_known_overlap_exactly():
    players = understat_players_from_league_data(LEAGUE_DATA)
    entries = build_crosswalk(players, _fpl_id_by_name(), strict=False)
    matched_names = {e.understat_name for e in entries}
    assert "Bukayo Saka" in matched_names
    saka_entry = next(e for e in entries if e.understat_name == "Bukayo Saka")
    assert saka_entry.matched_by == "exact"
    assert saka_entry.fpl_id > 0


def test_build_crosswalk_strict_raises_on_unmatched_players():
    players = understat_players_from_league_data(LEAGUE_DATA)
    with pytest.raises(CrosswalkError) as exc_info:
        build_crosswalk(players, _fpl_id_by_name(), strict=True)
    # real gap: most of this trimmed 50-player sample has no counterpart in the 100-row idlist
    assert len(exc_info.value.unmatched) > 0


def test_build_crosswalk_non_strict_returns_partial_matches_without_raising():
    players = understat_players_from_league_data(LEAGUE_DATA)
    entries = build_crosswalk(players, _fpl_id_by_name(), strict=False)
    assert 0 < len(entries) < len(players)


def test_manual_overlay_resolves_an_otherwise_unmatched_player():
    players = [UnderstatPlayer(understat_id=99999, name="Some Totally Different Spelling")]
    fpl_id_by_name = {"Real FPL Name": 12345}
    with pytest.raises(CrosswalkError):
        build_crosswalk(players, fpl_id_by_name, strict=True)

    entries = build_crosswalk(players, fpl_id_by_name, overlay={99999: 12345}, strict=True)
    assert len(entries) == 1
    assert entries[0].fpl_id == 12345
    assert entries[0].matched_by == "manual_overlay"


def test_normalized_pass_matches_accent_only_difference():
    players = [UnderstatPlayer(understat_id=1, name="Bruno Guimaraes")]  # no accent
    fpl_id_by_name = {"Bruno Guimarães": 42}  # accented, as FPL spells it
    entries = build_crosswalk(players, fpl_id_by_name, strict=True)
    assert entries[0].matched_by == "normalized"
    assert entries[0].fpl_id == 42


def test_web_name_exact_pass_matches_short_display_name():
    # Understat's player_name is the short display name; the full-name-only passes can't match
    # this at all (ENGINE_IMPROVEMENTS_2.md C.1).
    players = [UnderstatPlayer(understat_id=1, name="Bruno Fernandes")]
    fpl_id_by_name = {"Bruno Borges Fernandes": 42}
    fpl_id_by_web_name = {"Bruno Fernandes": 42}

    entries = build_crosswalk(
        players, fpl_id_by_name, strict=True, fpl_id_by_web_name=fpl_id_by_web_name
    )

    assert len(entries) == 1
    assert entries[0].fpl_id == 42
    assert entries[0].matched_by == "web_name_exact"


def test_web_name_normalized_pass_matches_accented_short_name():
    players = [UnderstatPlayer(understat_id=1, name="Bruno Guimaraes")]  # no accent
    fpl_id_by_name = {"Bruno Guimaraes Rodriguez Moura": 7}
    fpl_id_by_web_name = {"Bruno Guimarães": 7}  # accented web_name

    entries = build_crosswalk(
        players, fpl_id_by_name, strict=True, fpl_id_by_web_name=fpl_id_by_web_name
    )

    assert entries[0].matched_by == "web_name_normalized"
    assert entries[0].fpl_id == 7


def test_surname_token_pass_matches_unique_surname():
    players = [UnderstatPlayer(understat_id=1, name="J. Smith")]
    fpl_id_by_name = {"John David Smith": 9}

    entries = build_crosswalk(players, fpl_id_by_name, strict=True)

    assert entries[0].matched_by == "surname_token"
    assert entries[0].fpl_id == 9


def test_surname_token_pass_does_not_resolve_ambiguous_surname():
    # Two different FPL players share the surname "Smith" -- must not guess, must fall through
    # (here, to strict failure) rather than silently pick one (ENGINE_IMPROVEMENTS_2.md C.1).
    players = [UnderstatPlayer(understat_id=1, name="J. Smith")]
    fpl_id_by_name = {"John David Smith": 9, "Jack Smith": 10}

    with pytest.raises(CrosswalkError):
        build_crosswalk(players, fpl_id_by_name, strict=True)


def test_initial_surname_pass_matches_unique_first_initial_and_surname():
    players = [UnderstatPlayer(understat_id=1, name="J. Smith")]
    # Ambiguous by surname alone, but unique once the first initial is also considered.
    fpl_id_by_name = {"John David Smith": 9, "Karl Smith": 10}

    entries = build_crosswalk(players, fpl_id_by_name, strict=True)

    assert entries[0].matched_by == "initial_surname"
    assert entries[0].fpl_id == 9


def test_build_crosswalk_raises_ambiguity_when_two_understat_players_map_to_same_fpl_id_via_precise_pass():
    # A genuine data anomaly (not a heuristic false-positive): two different Understat players
    # both resolving to the same FPL id via *exact* web_name matches must fail loudly rather than
    # silently collapse two players into one.
    players = [
        UnderstatPlayer(understat_id=1, name="Bruno Fernandes"),
        UnderstatPlayer(understat_id=2, name="Bruno Fernandes Junior"),
    ]
    fpl_id_by_web_name = {"Bruno Fernandes": 42, "Bruno Fernandes Junior": 42}

    with pytest.raises(CrosswalkAmbiguityError):
        build_crosswalk(
            players, {}, strict=True, fpl_id_by_web_name=fpl_id_by_web_name
        )


def test_build_crosswalk_rejects_heuristic_collision_and_falls_through_to_unmatched():
    # Two real, distinct players share a surname ("Bueno") but only one appears in this season's
    # FPL id list -- both would resolve via surname-token to the same id if taken independently.
    # The second player's token match must be rejected (not accepted, not a hard error) and fall
    # through to strict failure as a genuine unmatched player.
    players = [
        UnderstatPlayer(understat_id=1, name="Hugo Bueno"),
        UnderstatPlayer(understat_id=2, name="Santiago Bueno"),
    ]
    fpl_id_by_name = {"Hugo Bueno": 99}  # only one "Bueno" in the FPL list this season

    with pytest.raises(CrosswalkError) as exc_info:
        build_crosswalk(players, fpl_id_by_name, strict=True)
    assert len(exc_info.value.unmatched) == 1
    assert exc_info.value.unmatched[0].understat_id == 2

    entries = build_crosswalk(players, fpl_id_by_name, strict=False)
    assert len(entries) == 1
    assert entries[0].understat_id == 1
    assert entries[0].fpl_id == 99


def test_build_crosswalk_precise_match_wins_over_an_earlier_heuristic_guess_for_the_same_id():
    # Regression for a real ambiguity found in the 2025/26 data: "Yerson Mosquera Valdelamar"'s
    # own surname token is "valdelamar", not "mosquera" -- but Understat lists him simply as
    # "Yerson Mosquera", whose last token happens to equal a *different* real player's actual
    # surname ("Cristhian Mosquera"). A naive single-pass match processed Yerson first, let his
    # heuristic surname-token guess claim Cristhian's own id, and then raised a false "ambiguity"
    # when Cristhian's real exact match came along -- the round-based match must instead let the
    # precise match win outright, independent of iteration order.
    players = [
        UnderstatPlayer(understat_id=1, name="Yerson Mosquera"),  # heuristic-only candidate
        UnderstatPlayer(understat_id=2, name="Cristhian Mosquera"),  # exact match available
    ]
    fpl_id_by_name = {
        "Cristhian Mosquera": 662,
        "Yerson Mosquera Valdelamar": 634,
    }

    entries = build_crosswalk(players, fpl_id_by_name, strict=False)
    by_understat_id = {e.understat_id: e for e in entries}

    assert by_understat_id[2].fpl_id == 662
    assert by_understat_id[2].matched_by == "exact"
    # Yerson's own true id (634) is reachable via the token-prefix pass (crosswalk coverage
    # Phase 1) -- "Yerson Mosquera" is an exact-order prefix of "Yerson Mosquera Valdelamar" --
    # and since Cristhian's exact match already claimed a *different* id (662), there's no
    # conflict: Yerson correctly resolves to his own id rather than either stealing Cristhian's or
    # staying unmatched.
    assert by_understat_id[1].fpl_id == 634
    assert by_understat_id[1].matched_by == "token_prefix"


def test_normalize_name_treats_hyphens_as_separators():
    # Crosswalk coverage Phase 1: "Smith-Rowe" vs "Smith Rowe" was otherwise invisible to every
    # matching pass.
    assert normalize_name("Emile Smith-Rowe") == normalize_name("Emile Smith Rowe")
    assert normalize_name("Rayan Aït-Nouri") == normalize_name("Rayan Ait Nouri")


def test_token_prefix_pass_matches_a_dropped_maternal_surname():
    # Real 2025/26 pattern: FPL's full legal name carries a Spanish/Portuguese maternal surname
    # Understat's short display name drops.
    players = [UnderstatPlayer(understat_id=1, name="Marcos Senesi")]
    fpl_id_by_name = {"Marcos Senesi Barón": 72}

    entries = build_crosswalk(players, fpl_id_by_name, strict=True)

    assert entries[0].matched_by == "token_prefix"
    assert entries[0].fpl_id == 72


def test_token_prefix_pass_matches_when_understat_carries_the_extra_token():
    # The reverse direction: Understat's own name is the longer one (e.g. "Amad Diallo Traore" vs
    # FPL's "Amad Diallo").
    players = [UnderstatPlayer(understat_id=1, name="Amad Diallo Traore")]
    fpl_id_by_name = {"Amad Diallo": 452}

    entries = build_crosswalk(players, fpl_id_by_name, strict=True)

    assert entries[0].matched_by == "token_prefix"
    assert entries[0].fpl_id == 452


def test_token_prefix_pass_does_not_resolve_ambiguous_prefix():
    # Two different FPL players both extend the same shorter Understat name -- must not guess.
    players = [UnderstatPlayer(understat_id=1, name="John Smith")]
    fpl_id_by_name = {"John Smith Anderson": 9, "John Smith Roberts": 10}

    with pytest.raises(CrosswalkError):
        build_crosswalk(players, fpl_id_by_name, strict=True)


def test_reversed_order_pass_matches_given_name_family_name_swap():
    # Real 2025/26 pattern: Japanese/Korean given-name-first vs family-name-first convention.
    players = [UnderstatPlayer(understat_id=1, name="Kaoru Mitoma")]
    fpl_id_by_name = {"Mitoma Kaoru": 157}

    entries = build_crosswalk(players, fpl_id_by_name, strict=True)

    assert entries[0].matched_by == "reversed_order"
    assert entries[0].fpl_id == 157


def test_reversed_order_pass_handles_a_hyphenated_given_name_block():
    # "Hee-Chan Hwang" vs "Hwang Hee-chan" -- the hyphenated block must swap as one unit, not get
    # split into three tokens and reversed token-by-token.
    players = [UnderstatPlayer(understat_id=1, name="Hee-Chan Hwang")]
    fpl_id_by_name = {"Hwang Hee-chan": 642}

    entries = build_crosswalk(players, fpl_id_by_name, strict=True)

    assert entries[0].matched_by == "reversed_order"
    assert entries[0].fpl_id == 642


def test_reversed_order_pass_does_not_apply_to_three_token_names():
    # No single unambiguous swap exists for a three-token name -- must fall through rather than
    # guess a reversal.
    players = [UnderstatPlayer(understat_id=1, name="John David Smith")]
    fpl_id_by_name = {"Smith David John": 9}

    with pytest.raises(CrosswalkError):
        build_crosswalk(players, fpl_id_by_name, strict=True)


def test_manual_overlay_resolves_a_real_2025_26_nickname_and_spelling_variant():
    # Sanity check on real entries in MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON[2025] (crosswalk
    # coverage Phase 1) -- confirms the overlay table itself is wired correctly, not just the
    # mechanism.
    from engine.data.crosswalk import MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON

    players = [
        UnderstatPlayer(understat_id=2496, name="Rodri"),
        UnderstatPlayer(understat_id=9024, name="Yeremi Pino"),
    ]
    fpl_id_by_name = {
        "Rodrigo 'Rodri' Hernandez Cascante": 421,
        "Yéremy Pino Santos": 712,
    }

    entries = build_crosswalk(
        players,
        fpl_id_by_name,
        overlay=MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON[2025],
        strict=True,
    )
    by_id = {e.understat_id: e for e in entries}

    assert by_id[2496].fpl_id == 421
    assert by_id[2496].matched_by == "manual_overlay"
    assert by_id[9024].fpl_id == 712
    assert by_id[9024].matched_by == "manual_overlay"


def test_manual_overlay_is_season_keyed_not_a_flat_id_map():
    # ENGINE_IMPROVEMENTS_3.md multi-season Phase 2: fpl_id is only meaningful within the one
    # season it was hand-verified against -- FPL reassigns element ids every season, so the same
    # understat_id must resolve differently depending on which season's overlay slice is used.
    from engine.data.crosswalk import MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON

    assert MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON[2025][2496] == 421  # Rodri, 2025/26
    assert 2496 not in MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON.get(2022, {})


def test_manual_overlay_resolves_a_real_2022_23_cross_player_collision():
    # Real 2022/23 case: Understat displays both Emerson Palmieri (West Ham) and Emerson Royal
    # (Tottenham) as the bare string "Emerson" -- FPL's own web_name "Emerson" belongs to only one
    # of them, so web_name_exact alone would (and, before this fix, did) resolve both to the same
    # id. The overlay must be checked *before* web_name_exact, not after, to correct this.
    from engine.data.crosswalk import MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON

    players = [
        UnderstatPlayer(understat_id=7430, name="Emerson"),  # Royal (Tottenham) -- needs the overlay
        UnderstatPlayer(understat_id=1245, name="Emerson"),  # Palmieri (West Ham) -- resolves via web_name
    ]
    fpl_id_by_name = {
        "Emerson Leite de Souza Junior": 445,  # Royal
        "Emerson Palmieri dos Santos": 545,  # Palmieri
    }
    fpl_id_by_web_name = {"Emerson Royal": 445, "Emerson": 545}

    entries = build_crosswalk(
        players,
        fpl_id_by_name,
        overlay=MANUAL_OVERLAY_UNDERSTAT_TO_FPL_BY_SEASON[2022],
        strict=True,
        fpl_id_by_web_name=fpl_id_by_web_name,
    )
    by_id = {e.understat_id: e for e in entries}

    assert by_id[7430].fpl_id == 445
    assert by_id[7430].matched_by == "manual_overlay"
    assert by_id[1245].fpl_id == 545
    assert by_id[1245].matched_by == "web_name_exact"


def test_assert_matched_share_passes_when_coverage_meets_threshold():
    assert_matched_share({1: 900.0, 2: 100.0}, matched_fpl_ids={1}, min_share=0.85)  # no raise


def test_assert_matched_share_raises_when_coverage_below_threshold():
    with pytest.raises(CrosswalkCoverageError):
        assert_matched_share({1: 500.0, 2: 500.0}, matched_fpl_ids={1}, min_share=0.90)


def test_assert_matched_share_vacuously_passes_on_zero_total_weight():
    assert_matched_share({}, matched_fpl_ids=set(), min_share=0.90)  # no raise


def test_fetch_fpl_web_names_parses_csv_via_injected_client():
    csv_body = "id,web_name,first_name,second_name\n42,Bruno Fernandes,Bruno,Borges Fernandes\n"

    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            request.url.path
            == "/vaastav/Fantasy-Premier-League/master/data/2025-26/players_raw.csv"
        )
        return httpx.Response(200, text=csv_body)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch_fpl_web_names(2025, client)
    assert result["Bruno Fernandes"] == 42


def test_fetch_fpl_id_list_parses_csv_via_injected_client():
    def handler(request: httpx.Request) -> httpx.Response:
        assert (
            request.url.path
            == "/vaastav/Fantasy-Premier-League/master/data/2023-24/player_idlist.csv"
        )
        return httpx.Response(200, text=IDLIST_CSV)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch_fpl_id_list(2023, client)
    assert result["Kai Havertz"] == 4


def test_crosswalk_builder_end_to_end_with_injected_client():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=IDLIST_CSV)

    with CrosswalkBuilder(client=httpx.Client(transport=httpx.MockTransport(handler))) as builder:
        entries = builder.build_for_season(2023, LEAGUE_DATA, strict=False)
    assert len(entries) > 0
