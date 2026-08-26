"""Per-player, per-gameweek actual performance (Player Stats page, PLAYER_STATS_PLAN D1/D13/G4) --
live-only for v1 (D4), sourced from
:meth:`~engine.data.fpl_client.FPLClient.get_element_summary`'s ``history`` list rather than any
new external client. FPL's own official ``expected_goals``/``expected_assists``/
``expected_goal_involvements``/``expected_goals_conceded`` fields (Opta-sourced) satisfy D1's
"underlying process" need with zero new ingestion.

Converting a gameweek's raw counts into points-per-component (D13) is *not* a pure lookup for
every component -- see :func:`actual_points_for_gameweek`'s docstring for the two components
(goals-conceded penalty, and why defensive contribution/bonus are passed through rather than
recomputed) where per-gameweek order of operations matters (PLAYER_STATS_PLAN's G4 note). Every
component here mirrors :func:`engine.aggregate.aggregate_gameweek`'s own lines, just driven by
what actually happened instead of what was projected.

``selected``, ``starts``, ``value``, ``transfers_in``, ``transfers_out``, and ``bps``
(DIFFERENTIALS_PLAN Phase 1) are read the same way but were previously discarded on ingestion.
Each defaults to ``None`` rather than a zero-like placeholder, both here and on
:class:`PlayerGameweekActual` itself -- a cache written before these fields existed deserializes
straight into ``None`` for all six (:class:`~api.state.AppState`'s loader passes a plain dict
through ``PlayerGameweekActual(**data)``, so an absent key needs a real default). ``selected``
defaulting to ``0`` in particular would be indistinguishable from "owned by nobody", and ownership
trend is computed by differencing ``selected`` across gameweeks -- a stale cache would then produce
a confident, wrong, flat trend for every player instead of an honest unknown.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from engine.data.fpl_client import FPLClient
from engine.scoring import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    GK,
    GOAL_POINTS,
    GOALS_CONCEDED_PENALTY,
    GOALS_CONCEDED_PER_PENALTY,
    GOALS_CONCEDED_POSITIONS,
    OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS,
    PENALTY_SAVE_POINTS,
    RED_CARD_POINTS,
    SAVES_PER_POINT,
    YELLOW_CARD_POINTS,
)

__all__ = [
    "PlayerGameweekActual",
    "ActualComponentPoints",
    "load_live_player_history",
    "actual_points_for_gameweek",
]


@dataclass(frozen=True)
class PlayerGameweekActual:
    """One player's real, already-decided outcome for one already-played gameweek -- the raw
    counts an FPL manager actually thinks in, not points. ``bonus`` and ``defensive_contribution``
    are FPL's own recorded point values for that gameweek (not raw action/BPS counts), matching
    how :func:`actual_points_for_gameweek` treats them: already decided, never recomputed."""

    gameweek: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_saved: int
    penalties_missed: int
    saves: int
    yellow_cards: int
    red_cards: int
    bonus: int
    defensive_contribution: int
    total_points: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    # DIFFERENTIALS_PLAN Phase 1 -- see module docstring for why these default to None, not 0.
    selected: int | None = None
    starts: int | None = None
    value: int | None = None
    transfers_in: int | None = None
    transfers_out: int | None = None
    bps: int | None = None


@dataclass(frozen=True)
class ActualComponentPoints:
    """This gameweek's raw counts converted to points, one field per
    :class:`~engine.aggregate.ComponentBreakdown` component, computed for this single gameweek
    before any summing across a range (PLAYER_STATS_PLAN's G4 note)."""

    appearance: float
    goals: float
    assists: float
    clean_sheet: float
    goals_conceded: float
    defensive_contribution: float
    saves: float
    bonus: float
    cards: float
    penalty_misses: float
    own_goals: float


def _optional_int(row: dict, key: str) -> int | None:
    """``row[key]`` as ``int``, or ``None`` when the key is absent -- distinct from ``0``, which
    is a real, meaningful value for several of these fields (see module docstring)."""
    value = row.get(key)
    return None if value is None else int(value)


def _actual_from_history_row(row: dict) -> PlayerGameweekActual:
    return PlayerGameweekActual(
        gameweek=int(row["round"]),
        minutes=int(row["minutes"]),
        goals_scored=int(row["goals_scored"]),
        assists=int(row["assists"]),
        clean_sheets=int(row["clean_sheets"]),
        goals_conceded=int(row["goals_conceded"]),
        own_goals=int(row["own_goals"]),
        penalties_saved=int(row["penalties_saved"]),
        penalties_missed=int(row["penalties_missed"]),
        saves=int(row["saves"]),
        yellow_cards=int(row["yellow_cards"]),
        red_cards=int(row["red_cards"]),
        bonus=int(row["bonus"]),
        defensive_contribution=int(row.get("defensive_contribution", 0)),
        total_points=int(row["total_points"]),
        expected_goals=float(row.get("expected_goals", 0.0)),
        expected_assists=float(row.get("expected_assists", 0.0)),
        expected_goal_involvements=float(row.get("expected_goal_involvements", 0.0)),
        expected_goals_conceded=float(row.get("expected_goals_conceded", 0.0)),
        selected=_optional_int(row, "selected"),
        starts=_optional_int(row, "starts"),
        value=_optional_int(row, "value"),
        transfers_in=_optional_int(row, "transfers_in"),
        transfers_out=_optional_int(row, "transfers_out"),
        bps=_optional_int(row, "bps"),
    )


def load_live_player_history(
    client: FPLClient, player_ids: Sequence[int]
) -> dict[int, list[PlayerGameweekActual]]:
    """Every requested player's this-season gameweek history, sorted by gameweek. A player with no
    played gameweeks yet gets an empty list, not a missing key or an error -- "hasn't played" is a
    real, common state (new signing, blank early gameweeks), not a caller mistake."""
    summaries = client.iter_element_summaries(list(player_ids))
    return {
        player_id: sorted(
            (_actual_from_history_row(row) for row in summary.get("history", [])),
            key=lambda actual: actual.gameweek,
        )
        for player_id, summary in summaries.items()
    }


def actual_points_for_gameweek(
    actual: PlayerGameweekActual, position: str
) -> ActualComponentPoints:
    """Convert one gameweek's raw counts into points-per-component, importing every value from
    ``engine.scoring`` rather than hardcoding any of them (matching that module's own rule).

    Two components are *not* simple per-field lookups:

    - **Goals-conceded penalty** is per match (``-1`` per 2 conceded, GK/DEF only). Computing it
      here, on one gameweek's raw ``goals_conceded``, then summing the resulting penalty across a
      range is correct; summing raw ``goals_conceded`` across a range first and applying the
      penalty once would not be (1 conceded in two separate gameweeks is 0 penalty in each, not
      "2 conceded" -> ``-1``).
    - **Bonus and defensive contribution** are copied straight from FPL's own recorded values
      (``actual.bonus``, ``actual.defensive_contribution``) rather than recomputed from BPS or raw
      defensive-action counts. Recomputing either from match events is what the *prediction*
      engine does for an unplayed gameweek (``engine.models.bonus``,
      ``engine.models.defensive_contribution``); for an already-played gameweek, FPL has already
      decided both.
    """
    appearance = 2.0 if actual.minutes >= 60 else (1.0 if actual.minutes >= 1 else 0.0)
    goals_conceded_penalty = (
        (actual.goals_conceded // GOALS_CONCEDED_PER_PENALTY) * GOALS_CONCEDED_PENALTY
        if position in GOALS_CONCEDED_POSITIONS
        else 0.0
    )
    saves_points = (
        (actual.saves // SAVES_PER_POINT) + actual.penalties_saved * PENALTY_SAVE_POINTS
        if position == GK
        else 0.0
    )
    return ActualComponentPoints(
        appearance=appearance,
        goals=actual.goals_scored * GOAL_POINTS[position],
        assists=actual.assists * ASSIST_POINTS,
        clean_sheet=actual.clean_sheets * CLEAN_SHEET_POINTS[position],
        goals_conceded=float(goals_conceded_penalty),
        defensive_contribution=float(actual.defensive_contribution),
        saves=float(saves_points),
        bonus=float(actual.bonus),
        cards=actual.yellow_cards * YELLOW_CARD_POINTS + actual.red_cards * RED_CARD_POINTS,
        penalty_misses=actual.penalties_missed * PENALTY_MISS_POINTS,
        own_goals=actual.own_goals * OWN_GOAL_POINTS,
    )
