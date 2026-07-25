"""Full-player-pool, per-gameweek orchestration — wires every Phase 2 component together (2.1-2.7).

``tests/test_phase2_integration.py`` proves the chain minutes -> goals/assists/clean-sheets/
defensive-contribution/cards/bonus -> aggregate -> top-level projection is wired correctly for
*one* player. Phase 3's walk-forward harness and Phase 5's weekly job both need to run that same
chain for an entire player pool, one row per player, every gameweek — this module is that batch
orchestrator, so callers stop hand-wiring the per-player chain themselves.

This module does not fetch or prepare data — it takes one row per player already carrying every
rate/feature input each component needs (already-computed EWMA rates, opponent adjustments, a
fitted minutes model, a fitted bonus model) and returns one prediction row per player. Sourcing
those inputs from real point-in-time snapshots is Phase 1's job (``engine/data/``, ``engine/
rates.py``).
"""

from __future__ import annotations

import pandas as pd

from engine.aggregate import ComponentBreakdown, aggregate_gameweek
from engine.models.assists import project_assists
from engine.models.bonus import BonusModel, build_features
from engine.models.cards import project_cards
from engine.models.clean_sheets import project_clean_sheet
from engine.models.defensive_contribution import project_defensive_contribution
from engine.models.goals import project_goals
from engine.models.minutes import FEATURE_COLUMNS as MINUTES_FEATURE_COLUMNS
from engine.models.minutes import MinutesModel
from engine.models.saves import project_saves
from engine.projections import PlayerGameweekProjection, project_player_gameweek
from engine.scoring import GK

__all__ = ["GAMEWEEK_POOL_COLUMNS", "project_gameweek_pool"]

# Every column ``project_gameweek_pool`` reads from its ``players`` input, beyond the minutes
# model's own FEATURE_COLUMNS. GK-only and outfield-only columns are documented inline below.
GAMEWEEK_POOL_COLUMNS = [
    "player_id",
    "position",
    *MINUTES_FEATURE_COLUMNS,
    "npxg_per_90",  # goals (2.2)
    "xa_per_90",  # assists (2.3)
    "team_xg_per_90",  # clean sheets (2.4)
    "team_xga_per_90",  # clean sheets (2.4) / saves (2.6, GK only)
    "opponent_xg_per_90",  # clean sheets (2.4)
    "opponent_xga_per_90",  # goals (2.2) / assists (2.3)
    "league_avg_xga_per_90",  # goals / assists / clean sheets / saves shared normalizer
    "yellow_card_rate_per_90",  # cards (2.6)
    "red_card_rate_per_90",  # cards (2.6)
    # Outfield only (BUILD_PLAN 2.5) — required unless position == GK:
    #   "dc_per_90", "opponent_possession_share"
    # GK only (BUILD_PLAN 2.6) — required when position == GK:
    #   "opponent_shots_on_target_per_90", "is_home"
]


def _project_one_player(
    player_id: int,
    position: str,
    gameweek: int,
    row: pd.Series,
    minutes_distribution,
    bonus_model: BonusModel,
) -> tuple[PlayerGameweekProjection, ComponentBreakdown, float]:
    """Returns (projection, breakdown, clean_sheet_probability) — the probability is surfaced
    separately since it isn't otherwise recoverable from the breakdown alone (BUILD_PLAN 3.2's
    calibration check needs the raw probability, not the points it converted to)."""
    expected_minutes = minutes_distribution.expected_minutes

    goals = project_goals(
        player_npxg_per_90=row["npxg_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
    )
    assists = project_assists(
        player_xa_per_90=row["xa_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
    )
    clean_sheet = project_clean_sheet(
        team_xg_per_90=row["team_xg_per_90"],
        team_xga_per_90=row["team_xga_per_90"],
        opponent_xg_per_90=row["opponent_xg_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
    )
    cards = project_cards(
        yellow_card_rate_per_90=row["yellow_card_rate_per_90"],
        red_card_rate_per_90=row["red_card_rate_per_90"],
        expected_minutes=expected_minutes,
    )

    if position == GK:
        defensive_contribution = None
        saves = project_saves(
            opponent_shots_on_target_per_90=row["opponent_shots_on_target_per_90"],
            team_xga_per_90=row["team_xga_per_90"],
            league_avg_xga_per_90=row["league_avg_xga_per_90"],
            is_home=bool(row["is_home"]),
            expected_minutes=expected_minutes,
        )
        defensive_action_rate_for_bonus = 0.0
    else:
        defensive_contribution = project_defensive_contribution(
            position=position,
            player_actions_per_90=row["dc_per_90"],
            opponent_possession_share=row["opponent_possession_share"],
            expected_minutes=expected_minutes,
        )
        saves = None
        defensive_action_rate_for_bonus = row["dc_per_90"]

    bonus_features = pd.DataFrame(
        [
            build_features(
                expected_goals=goals.expected_goals,
                expected_assists=assists.expected_assists,
                clean_sheet_probability=clean_sheet.clean_sheet_probability,
                defensive_action_rate=defensive_action_rate_for_bonus,
                position=position,
            )
        ]
    )
    bonus = bonus_model.predict(bonus_features)[0]

    breakdown = aggregate_gameweek(
        position,
        minutes_distribution,
        goals,
        assists,
        clean_sheet,
        bonus,
        cards,
        defensive_contribution=defensive_contribution,
        saves=saves,
    )
    projection = project_player_gameweek(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        minutes=minutes_distribution,
        breakdown=breakdown,
    )
    return projection, breakdown, clean_sheet.clean_sheet_probability


def project_gameweek_pool(
    players: pd.DataFrame,
    gameweek: int,
    minutes_model: MinutesModel,
    bonus_model: BonusModel,
) -> pd.DataFrame:
    """Run the full Phase 2 chain for every row of ``players`` (one row per player, columns per
    :data:`GAMEWEEK_POOL_COLUMNS`) and return one prediction row per player.

    The returned frame carries ``expected_points`` (the headline number) plus every line of the
    BUILD_PLAN 2.7 component breakdown and ``clean_sheet_probability`` — the same fields the web
    app's "detail on click" view and Phase 3's calibration checks both need, so callers don't have
    to re-derive them from a separate call.
    """
    if players.empty:
        raise ValueError("players must not be empty")

    minutes_distributions = minutes_model.predict(players)
    rows = []
    for (_, row), minutes_distribution in zip(
        players.iterrows(), minutes_distributions, strict=True
    ):
        player_id = int(row["player_id"])
        position = row["position"]
        _projection, breakdown, clean_sheet_probability = _project_one_player(
            player_id, position, gameweek, row, minutes_distribution, bonus_model
        )
        rows.append(
            {
                "player_id": player_id,
                "position": position,
                "gameweek": gameweek,
                "expected_points": breakdown.total,
                "expected_minutes": minutes_distribution.expected_minutes,
                "clean_sheet_probability": clean_sheet_probability,
                "appearance": breakdown.appearance,
                "goals": breakdown.goals,
                "assists": breakdown.assists,
                "clean_sheet": breakdown.clean_sheet,
                "goals_conceded": breakdown.goals_conceded,
                "defensive_contribution": breakdown.defensive_contribution,
                "saves": breakdown.saves,
                "bonus": breakdown.bonus,
                "cards": breakdown.cards,
                "penalty_misses": breakdown.penalty_misses,
            }
        )
    return pd.DataFrame(rows)
