"""Player Stats page row assembly (PLAYER_STATS_PLAN Phase 3/D17) -- one row per player with
their actual performance over the selected gameweek range plus a fixture cell (opponent, venue,
expected points) for each of the app's 3-gameweek horizon. Mirrors ``api.panel.build_panel_rows``'s
existing pattern closely, reusing its ``FixtureCell``/``build_team_fixture_map`` directly rather
than duplicating them -- only the actual-stats side (``features.player_stats``) is new.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from api.panel import FixtureCell
from engine.projections import PlayerHorizonProjection
from features.player_stats import PlayerActualStats

__all__ = ["PlayerAvailability", "PlayerStatsRow", "build_player_stats_rows"]


@dataclass(frozen=True)
class PlayerAvailability:
    """FPL's own live availability signal for one player -- distinct from ``low_confidence``,
    which is the engine's cold-start flag and carries no information about injury or rotation
    risk. ``status``/``chance_of_playing_next_round`` are always present in the projection cache;
    ``news`` is the free-text display field, kept strictly for display per BUILD_PLAN 2.1 (never
    parsed into a feature -- ``has_news``, in ``engine.data.availability_log``, is the structured
    signal for "was anything reported at all")."""

    status: str
    chance_of_playing_next_round: float
    news: str | None


@dataclass(frozen=True)
class PlayerStatsRow:
    player_id: int
    name: str
    team_id: int | None
    position: str
    price: int | None
    low_confidence: bool
    availability: PlayerAvailability | None
    actuals: PlayerActualStats
    fixtures: tuple[FixtureCell, ...]


def build_player_stats_rows(
    actual_stats_by_player: Mapping[int, PlayerActualStats],
    projections: Mapping[int, PlayerHorizonProjection],
    player_names: Mapping[int, str],
    buy_prices: Mapping[int, int],
    team_id_by_player: Mapping[int, int],
    position_by_player: Mapping[int, str],
    low_confidence_ids: set[int],
    availability_by_player: Mapping[int, PlayerAvailability],
    fixture_map: Mapping[tuple[int, int], list[tuple[int, bool]]],
    horizon_gameweeks: Sequence[int],
) -> list[PlayerStatsRow]:
    """One row per player who has actual stats in the requested range (D14 -- every other filter
    is applied client-side, so this returns the whole matching pool at once, unsorted; D6's
    click-to-sort happens in the browser)."""
    rows: list[PlayerStatsRow] = []
    for player_id, actuals in actual_stats_by_player.items():
        team_id = team_id_by_player.get(player_id)
        horizon = projections.get(player_id)
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
            PlayerStatsRow(
                player_id=player_id,
                name=player_names.get(player_id, ""),
                team_id=team_id,
                position=position_by_player.get(player_id, ""),
                price=buy_prices.get(player_id),
                low_confidence=player_id in low_confidence_ids,
                availability=availability_by_player.get(player_id),
                actuals=actuals,
                fixtures=tuple(fixtures),
            )
        )
    return rows
