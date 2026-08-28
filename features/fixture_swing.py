"""Fixture swing detection: is a team's run of fixtures getting easier or harder over the
gameweeks just ahead, compared to the block coming after that.

Pure domain logic over already-assembled inputs (``features/fixtures.py``'s difficulty model,
already-fetched ``TeamFixture``/``TeamRates``), not I/O, following ``features/differentials.py``'s
role in this codebase. Distinct from any change in an individual player's own outlook: this module
only ever looks at the opponent-strength inputs ``features/fixtures.py`` already isolates for that
reason, so "fixtures are turning" and "a player's own form changed" stay separate signals.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from features.fixtures import (
    HorizonDifficulty,
    TeamFixture,
    TeamRates,
    project_fixture_difficulties,
    team_horizon_difficulty,
)

__all__ = [
    "TeamSwing",
    "compute_team_swing",
    "rank_team_swings",
]


@dataclass(frozen=True)
class TeamSwing:
    """One team's fixture-difficulty swing between a near window and a far window.

    ``attack_swing``/``defense_swing`` are each ``near.rating - far.rating`` for that return type,
    on the same 1 (easiest) - 5 (hardest) scale ``HorizonDifficulty`` already uses. A **positive**
    value means the near-term fixtures are harder than the far ones, i.e. this team's run is about
    to get *easier* — the buy-before-it-turns signal. A **negative** value means the near-term
    fixtures are easier than what's coming — the sell-before-it-sours signal. Easy to get backwards
    since it's a delta of a "higher = harder" scale, not a plain difficulty rating, hence spelling
    it out here rather than leaving the sign to be inferred from the field names alone.

    ``near``/``far`` are ``None`` when that window is a genuine blank for this team (no fixture at
    all across every gameweek in the window) — matching ``api.fixtures_view``'s existing convention
    of surfacing a blank explicitly rather than raising, since a double/blank gameweek is real,
    expected fixture-list state, not an error. The swing fields are ``None`` whenever either window
    is, since a swing needs both ends to mean anything.
    """

    team_id: int
    near: HorizonDifficulty | None
    far: HorizonDifficulty | None
    attack_swing: float | None
    defense_swing: float | None


def _horizon_difficulty_or_none(
    team_id: int,
    fixtures: Sequence[TeamFixture],
    team_rates: Mapping[int, TeamRates],
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
    window_gameweeks: Sequence[int],
) -> HorizonDifficulty | None:
    window = set(window_gameweeks)
    team_fixtures = [f for f in fixtures if f.team_id == team_id and f.gameweek in window]
    if not team_fixtures:
        return None
    difficulties = project_fixture_difficulties(
        team_fixtures, team_rates, league_avg_xg_per_90, league_avg_xga_per_90
    )
    return team_horizon_difficulty(difficulties)


def compute_team_swing(
    team_id: int,
    fixtures: Sequence[TeamFixture],
    team_rates: Mapping[int, TeamRates],
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
    near_gameweeks: Sequence[int],
    far_gameweeks: Sequence[int],
) -> TeamSwing:
    """One team's swing between ``near_gameweeks`` and ``far_gameweeks`` — ``fixtures`` is the
    full fixture list across every team, filtered down to ``team_id`` internally, matching
    ``features.fixtures.fixture_counts_by_gameweek``'s own calling convention."""
    near = _horizon_difficulty_or_none(
        team_id, fixtures, team_rates, league_avg_xg_per_90, league_avg_xga_per_90, near_gameweeks
    )
    far = _horizon_difficulty_or_none(
        team_id, fixtures, team_rates, league_avg_xg_per_90, league_avg_xga_per_90, far_gameweeks
    )
    if near is None or far is None:
        return TeamSwing(team_id=team_id, near=near, far=far, attack_swing=None, defense_swing=None)
    return TeamSwing(
        team_id=team_id,
        near=near,
        far=far,
        attack_swing=float(near.attack_rating - far.attack_rating),
        defense_swing=float(near.defense_rating - far.defense_rating),
    )


def rank_team_swings(
    team_ids: Iterable[int],
    fixtures: Sequence[TeamFixture],
    team_rates: Mapping[int, TeamRates],
    league_avg_xg_per_90: float,
    league_avg_xga_per_90: float,
    near_gameweeks: Sequence[int],
    far_gameweeks: Sequence[int],
) -> list[TeamSwing]:
    """One :class:`TeamSwing` per ``team_ids`` entry, unsorted — matching
    ``api.fixtures_view.build_fixture_ticker_rows``'s own "frontend sorts" convention."""
    return [
        compute_team_swing(
            team_id,
            fixtures,
            team_rates,
            league_avg_xg_per_90,
            league_avg_xga_per_90,
            near_gameweeks,
            far_gameweeks,
        )
        for team_id in team_ids
    ]
