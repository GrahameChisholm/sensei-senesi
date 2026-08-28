"""Fixtures: custom per-team difficulty rating built from the engine's own inputs (BUILD_PLAN
Phase 4) — opponent expected goals conceded, opponent attacking strength, and home/away — rather
than FPL's arbitrary 1-5 colour scale. Feeds captaincy.py, transfers.py, and chips.py, and
surfaces as its own view later.

**Two ratings, not one.** A fixture that's easy for goals/assists isn't necessarily easy for a
clean sheet (a leaky-but-potent attacking side is a great fixture for your forward and a poor one
for your defender), so this module keeps ``attack_rating`` (goal/assist prospects) and
``defense_rating`` (clean-sheet prospects) separate rather than collapsing them into one number —
the same distinction goals (2.2) and clean sheets (2.4) already draw from the same opponent
xG/xGA inputs.

**Only the opponent's strength, normalized by the league average — never this team's own rate.**
Difficulty describes how hard *any* team would find this fixture, so a team's own attack/defence
quality has no place in it (that's exactly what the projections themselves already capture).

**Home/away split.** Home defence is measurably stronger than away defence (BUILD_PLAN 2.4), so
the opponent's rate used here must be the split that matches the venue *they'll* be playing at —
if this team is at home, the opponent is away, so their away-split rate applies (see
:func:`fixture_difficulty`).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

__all__ = [
    "TeamFixture",
    "TeamRates",
    "FixtureDifficulty",
    "HorizonDifficulty",
    "fixture_difficulty",
    "project_fixture_difficulties",
    "team_horizon_difficulty",
    "fixture_counts_by_gameweek",
]


@dataclass(frozen=True)
class TeamFixture:
    """One team's own-perspective view of a single fixture — the engine's storage schema
    (``engine.data.storage.Fixture``) keeps a match-perspective ``team_h``/``team_a`` row;
    callers project that into one of these per team before calling :func:`fixture_difficulty`."""

    team_id: int
    opponent_id: int
    gameweek: int
    is_home: bool


@dataclass(frozen=True)
class TeamRates:
    """A team's home/away-split attacking and defensive per-90 xG rates.

    Built from a full-sample, league-shrunk per-team rate (``engine.data.team_rates``) with a
    separate, data-efficient home/away multiplier applied on top, not a true per-venue split of
    the underlying rate itself. A direct venue split (``engine.models.clean_sheets.split_by_venue``)
    was tried and reverted after it measurably made clean-sheet calibration worse than a constant
    baseline: halving the sample behind every team rate cost more than the venue effect was worth.
    See ``engine.data.team_rates``'s module docstring for the full rationale and the shrinkage
    math this module's callers rely on to populate these fields."""

    home_xg_per_90: float
    away_xg_per_90: float
    home_xga_per_90: float
    away_xga_per_90: float

    def __post_init__(self) -> None:
        for value in (
            self.home_xg_per_90,
            self.away_xg_per_90,
            self.home_xga_per_90,
            self.away_xga_per_90,
        ):
            if value < 0:
                raise ValueError("rates must be non-negative")


# Fixed cut points for the human-facing 1 (easiest) - 5 (hardest) rating, applied to a
# "higher = harder" factor normalized around 1.0 (opponent rate / league average). Kept simple and
# interpretable per the "explainable over clever" principle — these are not fitted; revisit with
# real rate-distribution evidence from backtesting if the buckets prove miscalibrated.
_RATING_BREAKPOINTS = (0.7, 0.85, 1.15, 1.3)


def _bucket_rating(harder_factor: float) -> int:
    """Map a "higher = harder fixture" factor to an FPL-familiar 1 (easiest) - 5 (hardest)
    rating."""
    for rating, breakpoint in enumerate(_RATING_BREAKPOINTS, start=1):
        if harder_factor < breakpoint:
            return rating
    return len(_RATING_BREAKPOINTS) + 1


@dataclass(frozen=True)
class FixtureDifficulty:
    """One team's custom difficulty rating for one fixture. ``attack_factor``/``defense_factor``
    are the raw opponent-strength-vs-league-average ratios (>1 = opponent is weaker than average
    in that respect, i.e. an easier fixture for that return type); ``attack_rating``/
    ``defense_rating`` are the bucketed 1 (easiest) - 5 (hardest) figures for display."""

    team_id: int
    opponent_id: int
    gameweek: int
    is_home: bool
    attack_factor: float
    defense_factor: float
    attack_rating: int
    defense_rating: int

    @property
    def overall_rating(self) -> float:
        """Simple average of the two ratings — a single sort key for callers (e.g. a first-pass
        wildcard/transfer scan) that don't need the attack/defense split. Prefer the two ratings
        directly whenever the return type (goals/assists vs clean sheet) is actually known."""
        return (self.attack_rating + self.defense_rating) / 2.0


def fixture_difficulty(
    fixture: TeamFixture,
    opponent_rates: TeamRates,
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
) -> FixtureDifficulty:
    """Build one team's difficulty rating for one fixture from the opponent's own rates.

    ``opponent_rates`` must be the *opponent's* :class:`TeamRates` (not this team's own) — this
    function reads whichever of their home/away splits actually applies given ``fixture.is_home``
    (if this team is home, the opponent is away, so their away rates are what's actually faced).
    """
    if league_avg_xg_per_90 <= 0 or league_avg_xga_per_90 <= 0:
        raise ValueError("league averages must be positive")

    opponent_xg = (
        opponent_rates.away_xg_per_90 if fixture.is_home else opponent_rates.home_xg_per_90
    )
    opponent_xga = (
        opponent_rates.away_xga_per_90 if fixture.is_home else opponent_rates.home_xga_per_90
    )

    # Higher opponent xGA = leakier opponent defence = easier for this team's goals/assists.
    attack_factor = opponent_xga / league_avg_xga_per_90
    # Higher opponent xG = sharper opponent attack = harder for this team's clean sheet.
    defense_factor = opponent_xg / league_avg_xg_per_90

    return FixtureDifficulty(
        team_id=fixture.team_id,
        opponent_id=fixture.opponent_id,
        gameweek=fixture.gameweek,
        is_home=fixture.is_home,
        attack_factor=attack_factor,
        defense_factor=defense_factor,
        # attack_factor is "higher = easier", so invert it before bucketing (bucket expects
        # "higher = harder"); defense_factor is already "higher = harder" as-is.
        attack_rating=_bucket_rating(1.0 / attack_factor if attack_factor > 0 else float("inf")),
        defense_rating=_bucket_rating(defense_factor),
    )


def project_fixture_difficulties(
    fixtures: Sequence[TeamFixture],
    team_rates: Mapping[int, TeamRates],
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
) -> list[FixtureDifficulty]:
    """Batch entry point, mirroring ``engine.pipeline.project_gameweek_pool``'s per-row
    orchestration style: one :class:`FixtureDifficulty` per fixture, keyed off each fixture's own
    opponent's rates in ``team_rates``."""
    return [
        fixture_difficulty(
            fixture,
            team_rates[fixture.opponent_id],
            league_avg_xg_per_90,
            league_avg_xga_per_90,
        )
        for fixture in fixtures
    ]


@dataclass(frozen=True)
class HorizonDifficulty:
    """One team's difficulty rating averaged over several gameweeks — the "are this team's
    fixtures good over the next N gameweeks" view transfers.py and chips.py's Wildcard/Free Hit
    evaluators need, as opposed to :class:`FixtureDifficulty`'s single-gameweek view.

    ``fixture_count`` may exceed ``len(gameweeks)`` if the horizon contains a double gameweek —
    it is not itself used to detect blanks/doubles (see :func:`fixture_counts_by_gameweek` for
    that), just carried through so callers can see at a glance whether every gameweek in the
    average actually contributed a fixture.
    """

    team_id: int
    gameweeks: tuple[int, ...]
    fixture_count: int
    mean_attack_factor: float
    mean_defense_factor: float
    attack_rating: int
    defense_rating: int

    @property
    def overall_rating(self) -> float:
        return (self.attack_rating + self.defense_rating) / 2.0


def team_horizon_difficulty(fixture_difficulties: Sequence[FixtureDifficulty]) -> HorizonDifficulty:
    """Aggregate one team's per-fixture difficulties (:func:`fixture_difficulty`/
    :func:`project_fixture_difficulties`, already filtered to one team) into a single
    horizon-level rating. A double-gameweek fixture contributes two entries here and is simply
    averaged in like any other — total EV impact of the extra fixture is the projections' job to
    capture, not this rating's."""
    if not fixture_difficulties:
        raise ValueError("fixture_difficulties must not be empty")
    team_ids = {fd.team_id for fd in fixture_difficulties}
    if len(team_ids) != 1:
        raise ValueError("all fixture_difficulties must belong to the same team")

    count = len(fixture_difficulties)
    mean_attack_factor = sum(fd.attack_factor for fd in fixture_difficulties) / count
    mean_defense_factor = sum(fd.defense_factor for fd in fixture_difficulties) / count

    return HorizonDifficulty(
        team_id=team_ids.pop(),
        gameweeks=tuple(fd.gameweek for fd in fixture_difficulties),
        fixture_count=count,
        mean_attack_factor=mean_attack_factor,
        mean_defense_factor=mean_defense_factor,
        attack_rating=_bucket_rating(
            1.0 / mean_attack_factor if mean_attack_factor > 0 else float("inf")
        ),
        defense_rating=_bucket_rating(mean_defense_factor),
    )


def fixture_counts_by_gameweek(
    fixtures: Iterable[TeamFixture], team_id: int, gameweeks: Iterable[int]
) -> dict[int, int]:
    """Number of fixtures ``team_id`` has in each of ``gameweeks`` — 0 marks a blank, 2+ a double
    (BUILD_PLAN 4: chips.py's Free Hit/Wildcard evaluators read this same horizon). ``gameweeks``
    must be passed explicitly (rather than inferred from ``fixtures``) so a blank gameweek — one
    with no matching fixture row at all — still shows up as 0 instead of silently missing."""
    counts = dict.fromkeys(gameweeks, 0)
    for fixture in fixtures:
        if fixture.team_id == team_id and fixture.gameweek in counts:
            counts[fixture.gameweek] += 1
    return counts
