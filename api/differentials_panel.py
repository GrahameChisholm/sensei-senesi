"""Differentials page row assembly (DIFFERENTIALS_PLAN Phase 3) -- one row per qualifying
differential, with identity/team plus a display-only fixture cell for each of the app's
3-gameweek horizon (D4: fixtures and expected points never feed the ranking, they are shown
alongside it). Mirrors ``api.player_stats_panel.build_player_stats_rows``'s pattern closely,
reusing ``api.panel``'s ``FixtureCell``/fixture-lookup shape directly rather than duplicating it --
only the differential metrics themselves (``features.differentials``) are new here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from api.panel import FixtureCell
from engine.projections import PlayerHorizonProjection
from features.differentials import PlayerDifferential

__all__ = ["DifferentialRow", "build_differential_rows"]


@dataclass(frozen=True)
class DifferentialRow:
    differential: PlayerDifferential
    name: str
    team_id: int | None
    fixtures: tuple[FixtureCell, ...]


def build_differential_rows(
    differentials: Sequence[PlayerDifferential],
    player_names: Mapping[int, str],
    team_id_by_player: Mapping[int, int],
    projections: Mapping[int, PlayerHorizonProjection],
    fixture_map: Mapping[tuple[int, int], list[tuple[int, bool]]],
    horizon_gameweeks: Sequence[int],
) -> list[DifferentialRow]:
    """One row per differential, unsorted (D10 -- there is no single ranking to apply; the caller
    sorts on whichever column it wants, the same click-to-sort contract as every other panel)."""
    rows: list[DifferentialRow] = []
    for differential in differentials:
        team_id = team_id_by_player.get(differential.player_id)
        horizon = projections.get(differential.player_id)
        fixtures: list[FixtureCell] = []
        for gameweek in horizon_gameweeks:
            expected_points = (
                horizon.gameweeks[gameweek].expected_points
                if horizon is not None and gameweek in horizon.gameweeks
                else None
            )
            entries = fixture_map.get((team_id, gameweek), []) if team_id is not None else []
            opponent_id, is_home = entries[0] if entries else (None, None)
            fixtures.append(
                FixtureCell(
                    gameweek=gameweek,
                    opponent_id=opponent_id,
                    is_home=is_home,
                    expected_points=expected_points,
                )
            )
        rows.append(
            DifferentialRow(
                differential=differential,
                name=player_names.get(differential.player_id, ""),
                team_id=team_id,
                fixtures=tuple(fixtures),
            )
        )
    return rows
