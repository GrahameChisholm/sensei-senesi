"""Player search / comparison (BUILD_PLAN 5.2) — pure functions over
:class:`~api.state.AppState`'s ``projections``/``player_names``/``buy_prices``, no API/HTTP
concerns (this module never imports from ``api/``, matching every other ``features/`` module's
layering).

**One gameweek at a time.** Every projection here is keyed by ``gameweek`` — a player's full
horizon isn't collapsed into one "expected points" number, since that would hide which week a
result is even about. Callers pick a gameweek explicitly (defaulting to the earliest one a given
player has a projection for), matching ``features.captaincy``'s own single-gameweek convention.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from engine.aggregate import ComponentBreakdown
from engine.projections import PlayerHorizonProjection

__all__ = [
    "PlayerSummary",
    "PlayerDetail",
    "search_players",
    "get_player_detail",
]


@dataclass(frozen=True)
class PlayerSummary:
    """One player's search-result row — enough to list and filter by, not the full breakdown
    (see :class:`PlayerDetail` for that)."""

    player_id: int
    name: str
    position: str
    price: int | None
    gameweek: int
    expected_points: float


@dataclass(frozen=True)
class PlayerDetail:
    """One player's full projection for one gameweek — the "click to see what drove it" view
    (BUILD_PLAN 2.7), same breakdown/simulation data every other feature already surfaces."""

    player_id: int
    name: str
    position: str
    price: int | None
    gameweek: int
    expected_points: float
    breakdown: ComponentBreakdown
    floor: float | None
    ceiling: float | None
    prob_big_haul: float | None


def _earliest_gameweek(horizon: PlayerHorizonProjection) -> int:
    return min(horizon.gameweeks)


def _resolve_gameweek(horizon: PlayerHorizonProjection, gameweek: int | None) -> int | None:
    if gameweek is not None:
        return gameweek if gameweek in horizon.gameweeks else None
    return _earliest_gameweek(horizon) if horizon.gameweeks else None


def search_players(
    projections: Mapping[int, PlayerHorizonProjection],
    player_names: Mapping[int, str],
    buy_prices: Mapping[int, int],
    search: str | None = None,
    position: str | None = None,
    max_price: int | None = None,
    gameweek: int | None = None,
) -> list[PlayerSummary]:
    """Every player in ``projections`` matching the given filters, ranked by expected points
    (descending). ``search`` matches case-insensitively against ``player_names`` (a player with no
    entry there — a projection from a source that hasn't populated names yet — is excluded from
    name search but still returned when ``search`` is ``None``). ``gameweek`` defaults to each
    player's own earliest projected gameweek; a player with no projection for an explicitly
    requested ``gameweek`` is excluded rather than raising, since "not projected this week" is a
    real, common state (blank gameweek, unmatched crosswalk), not a caller error.
    """
    normalized_search = search.strip().lower() if search else None
    results: list[PlayerSummary] = []
    for player_id, horizon in projections.items():
        name = player_names.get(player_id, "")
        if normalized_search is not None and normalized_search not in name.lower():
            continue
        if position is not None and horizon.position != position:
            continue
        price = buy_prices.get(player_id)
        if max_price is not None and (price is None or price > max_price):
            continue
        resolved_gameweek = _resolve_gameweek(horizon, gameweek)
        if resolved_gameweek is None:
            continue
        projection = horizon.gameweeks[resolved_gameweek]
        results.append(
            PlayerSummary(
                player_id=player_id,
                name=name,
                position=horizon.position,
                price=price,
                gameweek=resolved_gameweek,
                expected_points=projection.expected_points,
            )
        )
    results.sort(key=lambda summary: summary.expected_points, reverse=True)
    return results


def get_player_detail(
    player_id: int,
    projections: Mapping[int, PlayerHorizonProjection],
    player_names: Mapping[int, str],
    buy_prices: Mapping[int, int],
    gameweek: int | None = None,
) -> PlayerDetail:
    """The full projection breakdown for one player, one gameweek — ``gameweek`` defaults to
    their earliest projected gameweek. Raises :class:`KeyError` for an unknown ``player_id`` or a
    ``gameweek`` this player has no projection for — unlike :func:`search_players`, a caller asking
    about one specific player has made a specific claim about them that should fail loudly if
    wrong, not be silently dropped from a list.
    """
    if player_id not in projections:
        raise KeyError(f"no projection for player_id {player_id}")
    horizon = projections[player_id]
    resolved_gameweek = _resolve_gameweek(horizon, gameweek)
    if resolved_gameweek is None:
        raise KeyError(f"player_id {player_id} has no projection for gameweek {gameweek}")
    projection = horizon.gameweeks[resolved_gameweek]
    simulation = projection.simulation
    return PlayerDetail(
        player_id=player_id,
        name=player_names.get(player_id, ""),
        position=horizon.position,
        price=buy_prices.get(player_id),
        gameweek=resolved_gameweek,
        expected_points=projection.expected_points,
        breakdown=projection.breakdown,
        floor=simulation.floor if simulation is not None else None,
        ceiling=simulation.ceiling if simulation is not None else None,
        prob_big_haul=simulation.prob_big_haul if simulation is not None else None,
    )
