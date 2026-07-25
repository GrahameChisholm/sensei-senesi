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
    CrosswalkBuilder,
    CrosswalkError,
    UnderstatPlayer,
    build_crosswalk,
    fetch_fpl_id_list,
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
