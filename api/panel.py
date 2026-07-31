"""Player-panel row assembly (Phase 6) — one row per player with three separate per-gameweek
fixture cells (opponent, venue, expected points), regardless of the pitch's own Next GW/Next 3 GWs
toggle (D11's rule: the panel always shows three). Reuses ``features.players.search_players`` for
filtering/ranking; only the per-fixture enrichment across multiple gameweeks is new here.

**Known limitation, not a bug.** ``search_players`` resolves an explicitly-passed ``gameweek`` to
exactly that gameweek or excludes the player — so a player blank in the *first* target gameweek
specifically (but with real fixtures in the second/third) won't appear in the panel. Filtering
consistently by one gameweek while still enriching with the other two is the simplest coherent
behaviour; revisit only if this proves a real gap in practice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from engine.projections import PlayerHorizonProjection
from features.players import search_players

__all__ = ["FixtureCell", "PlayerPanelRow", "build_team_fixture_map", "build_panel_rows"]


@dataclass(frozen=True)
class FixtureCell:
    gameweek: int
    opponent_id: int | None
    is_home: bool | None
    expected_points: float | None  # None = blank gameweek for this player's team


@dataclass(frozen=True)
class PlayerPanelRow:
    player_id: int
    name: str
    team_id: int | None
    position: str
    price: int | None
    low_confidence: bool
    fixtures: tuple[FixtureCell, ...]


def build_team_fixture_map(
    fixtures: Sequence[dict],
) -> dict[tuple[int, int], list[tuple[int, bool]]]:
    """``(team_id, gameweek) -> [(opponent_id, is_home), ...]`` — a list (not a single value) so a
    double gameweek is representable, even though the panel's v1 display only ever shows the
    first entry."""
    mapping: dict[tuple[int, int], list[tuple[int, bool]]] = {}
    for row in fixtures:
        key = (row["team_id"], row["gameweek"])
        mapping.setdefault(key, []).append((row["opponent_id"], row["is_home"]))
    return mapping


def build_panel_rows(
    projections: Mapping[int, PlayerHorizonProjection],
    player_names: Mapping[int, str],
    buy_prices: Mapping[int, int],
    team_id_by_player: Mapping[int, int],
    low_confidence_ids: set[int],
    fixture_map: Mapping[tuple[int, int], list[tuple[int, bool]]],
    gameweeks: Sequence[int],
    search: str | None = None,
    position: str | None = None,
    max_price: int | None = None,
) -> list[PlayerPanelRow]:
    """Filter/sort by ``gameweeks[0]`` (via ``search_players``), then enrich every result with a
    fixture cell for each of ``gameweeks`` — always three, never resized by any UI toggle."""
    primary_gameweek = gameweeks[0]
    summaries = search_players(
        projections, player_names, buy_prices, search, position, max_price, primary_gameweek
    )

    rows: list[PlayerPanelRow] = []
    for summary in summaries:
        team_id = team_id_by_player.get(summary.player_id)
        horizon = projections.get(summary.player_id)
        fixtures: list[FixtureCell] = []
        for gameweek in gameweeks:
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
            PlayerPanelRow(
                player_id=summary.player_id,
                name=summary.name,
                team_id=team_id,
                position=summary.position,
                price=summary.price,
                low_confidence=summary.player_id in low_confidence_ids,
                fixtures=tuple(fixtures),
            )
        )
    return rows
