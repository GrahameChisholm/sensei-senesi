"""Fixture ticker row assembly -- the teams by next N gameweeks grid GET /fixtures serves.

Uses FPL's own ``difficulty`` field straight from the cache's ``fixtures`` list (already carried
through by ``scripts/build_projections.py::build_fixture_list``), not a custom stats derived
rating. ``features/fixtures.py`` has a fuller custom model but needs team rate data with no live
source yet; this module deliberately does not depend on it.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

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


def build_fixture_ticker_rows(
    fixtures: Sequence[dict],
    team_ids: Iterable[int],
    gameweeks: Sequence[int],
) -> list[TeamDifficultyRow]:
    """One row per ``team_ids`` entry, unsorted (the frontend sorts, matching
    ``api.panel.build_panel_rows``'s own convention) -- every one of ``gameweeks`` gets a cell,
    even a genuinely blank one, so a blank gameweek is visible rather than just missing."""
    gameweek_set = set(gameweeks)
    rows: list[TeamDifficultyRow] = []
    for team_id in team_ids:
        team_fixtures = [
            row for row in fixtures if row["team_id"] == team_id and row["gameweek"] in gameweek_set
        ]
        cells: list[GameweekDifficultyCell] = []
        all_difficulties: list[int] = []
        for gameweek in gameweeks:
            entries = tuple(
                FixtureCellEntry(
                    opponent_id=row["opponent_id"],
                    is_home=row["is_home"],
                    difficulty=row["difficulty"],
                )
                for row in team_fixtures
                if row["gameweek"] == gameweek
            )
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
