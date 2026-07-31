"""Tests for features.chip_calendar -- 2026/27's real, symmetric one-full-set-per-half chip
ruleset (D14/G7), a deliberate sibling to (not a reuse of) simulator/chip_calendar.py's 2025/26
ruleset.
"""

from __future__ import annotations

import pytest

from features.chip_calendar import (
    FIRST_HALF_LAST_GAMEWEEK,
    ChipUsage,
    available_chips_this_gameweek,
    record_chip_played,
)
from features.team_state import CHIPS


class TestChipUsage:
    def test_unknown_chip_in_first_half_raises(self):
        with pytest.raises(ValueError, match="unknown chip"):
            ChipUsage(first_half_played=frozenset({"not_a_real_chip"}))

    def test_unknown_chip_in_second_half_raises(self):
        with pytest.raises(ValueError, match="unknown chip"):
            ChipUsage(second_half_played=frozenset({"not_a_real_chip"}))

    def test_default_usage_is_empty(self):
        usage = ChipUsage()
        assert usage.first_half_played == frozenset()
        assert usage.second_half_played == frozenset()


class TestAvailableChipsThisGameweek:
    def test_all_four_available_at_gw1(self):
        assert available_chips_this_gameweek(ChipUsage(), 1) == frozenset(CHIPS)

    def test_playing_one_chip_leaves_the_other_three_available(self):
        usage = record_chip_played(ChipUsage(), "wildcard", gameweek=3)
        available = available_chips_this_gameweek(usage, 3)
        assert available == frozenset(CHIPS) - {"wildcard"}
        assert len(available) == 3

    def test_first_half_usage_does_not_affect_second_half(self):
        usage = record_chip_played(ChipUsage(), "bench_boost", gameweek=10)
        assert "bench_boost" not in available_chips_this_gameweek(usage, 10)
        assert "bench_boost" in available_chips_this_gameweek(usage, 20)

    def test_second_half_usage_does_not_affect_first_half(self):
        usage = record_chip_played(ChipUsage(), "triple_captain", gameweek=25)
        assert "triple_captain" in available_chips_this_gameweek(usage, 15)
        assert "triple_captain" not in available_chips_this_gameweek(usage, 30)

    def test_gw19_is_first_half_gw20_is_second_half(self):
        assert FIRST_HALF_LAST_GAMEWEEK == 19
        usage = record_chip_played(ChipUsage(), "free_hit", gameweek=19)
        assert "free_hit" not in available_chips_this_gameweek(usage, 19)
        assert "free_hit" in available_chips_this_gameweek(usage, 20)

    def test_all_four_chips_playable_independently_in_the_same_half(self):
        usage = ChipUsage()
        for chip in CHIPS:
            usage = record_chip_played(usage, chip, gameweek=5)
        assert available_chips_this_gameweek(usage, 5) == frozenset()
        # But every one of them is available again once the second half starts.
        assert available_chips_this_gameweek(usage, 20) == frozenset(CHIPS)


class TestRecordChipPlayed:
    def test_playing_an_unavailable_chip_raises(self):
        usage = record_chip_played(ChipUsage(), "wildcard", gameweek=3)
        with pytest.raises(ValueError, match="not available"):
            record_chip_played(usage, "wildcard", gameweek=5)  # still first half, already used

    def test_playing_an_unknown_chip_raises(self):
        with pytest.raises(ValueError):
            record_chip_played(ChipUsage(), "not_a_real_chip", gameweek=1)

    def test_does_not_mutate_the_input_usage(self):
        usage = ChipUsage()
        record_chip_played(usage, "wildcard", gameweek=1)
        assert usage.first_half_played == frozenset()

    def test_second_half_wildcard_is_independent_of_first_half(self):
        usage = record_chip_played(ChipUsage(), "wildcard", gameweek=5)
        usage = record_chip_played(usage, "wildcard", gameweek=25)  # a second, independent wildcard
        assert "wildcard" not in available_chips_this_gameweek(usage, 10)
        assert "wildcard" not in available_chips_this_gameweek(usage, 30)
