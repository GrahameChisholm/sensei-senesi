"""Fixture-swing panel row assembly -- GET /teams/fixture-swing serves this.

Mirrors ``api/fixtures_view.py``'s shape: pure row assembly from already-loaded ``AppState``, no
FPL rule logic of its own. ``features.fixture_swing.rank_team_swings`` does the real per-team swing
math; this module's own job is building that function's inputs from ``AppState`` and attaching
``has_owned_player`` -- the owned-squad cross-reference done server-side, matching
``/players/differentials``'s own ``hide_owned`` convention in ``api/main.py``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from api.state import AppState
from features.fixture_swing import TeamSwing, rank_team_swings
from features.fixtures import TeamFixture, TeamRates

__all__ = [
    "TeamSwingRow",
    "build_fixture_swing_rows",
]


@dataclass(frozen=True)
class TeamSwingRow:
    swing: TeamSwing
    has_owned_player: bool


def _league_average(team_rates: dict[int, TeamRates], home_attr: str, away_attr: str) -> float:
    """Plain mean, across every team with a rate snapshot, of that team's own ``(home + away) /
    2`` rate for one stat -- this recovers each team's underlying shrunk rate exactly regardless of
    its venue multiplier (``home = shrunk * mult``, ``away = shrunk * (2 - mult)``, so their
    average is ``shrunk`` every time), giving a single, venue-neutral per-90 league average to
    normalize every fixture's difficulty against, the live-snapshot equivalent of
    ``backtest.run_season.build_fixture_rate_frame``'s per-gameweek league average.
    """
    values = [
        (getattr(rates, home_attr) + getattr(rates, away_attr)) / 2.0
        for rates in team_rates.values()
    ]
    return sum(values) / len(values)


def build_fixture_swing_rows(
    app_state: AppState,
    near_gameweeks: Sequence[int],
    far_gameweeks: Sequence[int],
    owned_team_ids: Iterable[int],
) -> list[TeamSwingRow]:
    """One row per team in ``app_state.teams``, unsorted -- matching every other panel in this
    codebase's "frontend sorts" convention.

    A fixture whose opponent has no rate snapshot yet (no live source, or a true GW1 pull with no
    completed matches anywhere) is excluded from the difficulty calculation entirely, rather than
    raising -- ``features.fixture_swing``'s own "no fixture in this window" handling already
    degrades a team down to a ``None`` window in exactly that situation, so filtering here reuses
    that path rather than adding a second one. Returns an empty list outright when no team has a
    rate snapshot at all (the true pre-season case), since there is no league average to normalize
    against yet.
    """
    team_rates = app_state.team_rates
    if not team_rates:
        return []

    fixtures = [
        TeamFixture(
            team_id=row["team_id"],
            opponent_id=row["opponent_id"],
            gameweek=row["gameweek"],
            is_home=row["is_home"],
        )
        for row in app_state.fixtures
        if row["opponent_id"] in team_rates
    ]
    league_avg_xg = _league_average(team_rates, "home_xg_per_90", "away_xg_per_90")
    league_avg_xga = _league_average(team_rates, "home_xga_per_90", "away_xga_per_90")
    owned = set(owned_team_ids)

    swings: list[TeamSwing] = rank_team_swings(
        app_state.teams.keys(),
        fixtures,
        team_rates,
        league_avg_xg,
        league_avg_xga,
        near_gameweeks,
        far_gameweeks,
    )
    return [TeamSwingRow(swing=swing, has_owned_player=swing.team_id in owned) for swing in swings]
