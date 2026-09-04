"""Fixture ticker row assembly -- the teams by next N gameweeks grid GET /fixtures serves.

Uses FPL's own ``difficulty`` field straight from the cache's ``fixtures`` list (already carried
through by ``scripts/build_projections.py::build_fixture_list``), not a custom stats derived
rating -- ``features/fixtures.py`` has a fuller custom difficulty model, deliberately not used
here (see that module's docstring for why "difficulty" and "expected goals" are different
questions). This module additionally attaches each fixture's genuine matchup expected goals
(``features.fixtures.fixture_expected_goals``), live via ``AppState.team_rates``, alongside the
FPL difficulty rating rather than replacing it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from features.fixtures import TeamFixture, TeamRates, fixture_expected_goals

__all__ = [
    "DEFAULT_FIXTURE_TICKER_HORIZON",
    "FixtureCellEntry",
    "GameweekDifficultyCell",
    "TeamDifficultyRow",
    "build_fixture_ticker_rows",
]

DEFAULT_FIXTURE_TICKER_HORIZON = 5


@dataclass(frozen=True)
class FixtureCellEntry:
    opponent_id: int
    is_home: bool
    difficulty: int
    # None when either side has no live rate snapshot yet (e.g. early season) -- the frontend
    # simply omits the expected-goals line for that fixture rather than showing a placeholder.
    expected_goals_for: float | None = None
    expected_goals_against: float | None = None


@dataclass(frozen=True)
class GameweekDifficultyCell:
    """One team's one gameweek cell. An empty ``fixtures`` tuple is a blank gameweek; two entries
    is a double gameweek."""

    gameweek: int
    fixtures: tuple[FixtureCellEntry, ...]


@dataclass(frozen=True)
class TeamDifficultyRow:
    team_id: int
    gameweeks: tuple[GameweekDifficultyCell, ...]
    # None only when this team has no fixture at all anywhere in the horizon.
    average_difficulty: float | None


def _expected_goals_for_row(
    row: dict, team_rates: Mapping[int, TeamRates], league_avg_xga_per_90: float
) -> tuple[float | None, float | None]:
    team_id, opponent_id = row["team_id"], row["opponent_id"]
    if team_id not in team_rates or opponent_id not in team_rates:
        return None, None
    fixture = TeamFixture(
        team_id=team_id, opponent_id=opponent_id, gameweek=row["gameweek"], is_home=row["is_home"]
    )
    result = fixture_expected_goals(
        fixture, team_rates[team_id], team_rates[opponent_id], league_avg_xga_per_90
    )
    return result.expected_goals_for, result.expected_goals_against


def build_fixture_ticker_rows(
    fixtures: Sequence[dict],
    team_ids: Iterable[int],
    gameweeks: Sequence[int],
    team_rates: Mapping[int, TeamRates] | None = None,
    league_avg_xga_per_90: float | None = None,
) -> list[TeamDifficultyRow]:
    """One row per ``team_ids`` entry, unsorted (the frontend sorts, matching
    ``api.panel.build_panel_rows``'s own convention) -- every one of ``gameweeks`` gets a cell,
    even a genuinely blank one, so a blank gameweek is visible rather than just missing.

    ``team_rates``/``league_avg_xga_per_90`` are optional and must be given together -- when
    provided, each fixture also carries its genuine matchup expected goals
    (``features.fixtures.fixture_expected_goals``), computed from whichever team has a live rate
    snapshot (a fixture whose team or opponent has none gets ``None`` for both fields rather than
    raising, the same "no live source yet" degrade-gracefully convention
    ``api.fixture_swing_panel`` already uses). Omitting them leaves every expected-goals field
    ``None``, matching this function's pre-expected-goals behaviour exactly.
    """
    gameweek_set = set(gameweeks)
    rows: list[TeamDifficultyRow] = []
    for team_id in team_ids:
        team_fixtures = [
            row for row in fixtures if row["team_id"] == team_id and row["gameweek"] in gameweek_set
        ]
        cells: list[GameweekDifficultyCell] = []
        all_difficulties: list[int] = []
        for gameweek in gameweeks:
            entries = []
            for row in team_fixtures:
                if row["gameweek"] != gameweek:
                    continue
                if team_rates is not None and league_avg_xga_per_90 is not None:
                    xg_for, xg_against = _expected_goals_for_row(
                        row, team_rates, league_avg_xga_per_90
                    )
                else:
                    xg_for = xg_against = None
                entries.append(
                    FixtureCellEntry(
                        opponent_id=row["opponent_id"],
                        is_home=row["is_home"],
                        difficulty=row["difficulty"],
                        expected_goals_for=xg_for,
                        expected_goals_against=xg_against,
                    )
                )
            entries = tuple(entries)
            cells.append(GameweekDifficultyCell(gameweek=gameweek, fixtures=entries))
            all_difficulties.extend(entry.difficulty for entry in entries)

        average_difficulty = (
            sum(all_difficulties) / len(all_difficulties) if all_difficulties else None
        )
        rows.append(
            TeamDifficultyRow(
                team_id=team_id,
                gameweeks=tuple(cells),
                average_difficulty=average_difficulty,
            )
        )
    return rows
