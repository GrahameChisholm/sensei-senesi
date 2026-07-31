"""Tests for simulator/chip_calendar.py -- 2025/26's chip rules (two Wildcards, one-shot Free
Hit/Bench Boost/Triple Captain), distinct from features/chips.py's 2026/27 rules."""

from __future__ import annotations

import pytest

from simulator.chip_calendar import ChipUsage, available_chips_this_gameweek, record_chip_played


def test_all_four_chips_available_with_no_usage():
    assert available_chips_this_gameweek(ChipUsage(), gameweek=5) == {
        "wildcard",
        "free_hit",
        "triple_captain",
        "bench_boost",
    }


def test_wildcard_unavailable_in_first_half_after_use():
    usage = record_chip_played(ChipUsage(), "wildcard")
    available = available_chips_this_gameweek(usage, gameweek=10)
    assert "wildcard" not in available


def test_wildcard_available_again_in_second_half_after_first_half_use():
    usage = record_chip_played(ChipUsage(), "wildcard")
    available = available_chips_this_gameweek(usage, gameweek=20)
    assert "wildcard" in available


def test_wildcard_unavailable_in_second_half_after_both_used():
    usage = record_chip_played(ChipUsage(), "wildcard")
    usage = record_chip_played(usage, "wildcard")
    assert "wildcard" not in available_chips_this_gameweek(usage, gameweek=25)


@pytest.mark.parametrize("chip", ["free_hit", "bench_boost", "triple_captain"])
def test_one_shot_chip_unavailable_after_use_all_season(chip):
    usage = record_chip_played(ChipUsage(), chip)
    assert chip not in available_chips_this_gameweek(usage, gameweek=1)
    assert chip not in available_chips_this_gameweek(usage, gameweek=38)


@pytest.mark.parametrize("chip", ["free_hit", "bench_boost", "triple_captain"])
def test_one_shot_chip_cannot_be_played_twice(chip):
    usage = record_chip_played(ChipUsage(), chip)
    with pytest.raises(ValueError, match="already played"):
        record_chip_played(usage, chip)


def test_record_unknown_chip_raises():
    with pytest.raises(ValueError, match="unknown chip"):
        record_chip_played(ChipUsage(), "not_a_real_chip")


def test_wildcards_played_out_of_range_raises():
    with pytest.raises(ValueError, match="between 0 and 2"):
        ChipUsage(wildcards_played=3)
