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

from collections.abc import Mapping
from dataclasses import dataclass, field

import pandas as pd

from engine.aggregate import ComponentBreakdown, aggregate_gameweek
from engine.models.assists import project_assists
from engine.models.bonus import BonusModel, build_features
from engine.models.cards import project_cards, project_own_goals
from engine.models.clean_sheets import DEFAULT_DIXON_COLES_RHO, project_clean_sheet
from engine.models.defensive_contribution import (
    DEFAULT_OVERDISPERSION,
    project_defensive_contribution,
)
from engine.models.goals import DEFAULT_PENALTY_CONVERSION_RATE, project_goals
from engine.models.minutes import FEATURE_COLUMNS as MINUTES_FEATURE_COLUMNS
from engine.models.minutes import MinutesModel
from engine.models.saves import (
    DEFAULT_AWAY_SHOT_MULTIPLIER,
    DEFAULT_SAVE_CONVERSION_RATE,
    project_saves,
)
from engine.projections import PlayerGameweekProjection, project_player_gameweek
from engine.scoring import DEF, FWD, GK, MID

__all__ = ["GAMEWEEK_POOL_COLUMNS", "FittedConstants", "project_gameweek_pool"]


@dataclass(frozen=True)
class FittedConstants:
    """Per-gameweek-refit values for the five component constants that BUILD_PLAN 2.8's
    regression layer was meant to fit but never did (ENGINE_IMPROVEMENTS.md Tier 1.2) — Dixon-Coles
    ``rho``, the saves model's conversion rate and away-shot multiplier, defensive contribution's
    Negative Binomial overdispersion (per position), and per-player penalty conversion rates. Also
    carries ``shrinkage_k`` (ENGINE_IMPROVEMENTS_2.md B.2), the goals/assists thin-sample-rate
    shrinkage strength — unlike the five Tier 1.2 constants, this has no closed-form per-gameweek
    estimator, so it's an evidence-based fixed hyperparameter selected once via an end-to-end
    backtest sweep (``backtest/run_season.py``'s ``SHRINKAGE_K``) rather than refit every gameweek.

    Every field defaults to the corresponding component's own existing ``DEFAULT_*`` constant, so
    ``project_gameweek_pool(players, gameweek, minutes_model, bonus_model)`` called with no
    ``fitted_constants`` argument is byte-for-byte today's (pre-Tier-1.2) behavior. Construct one
    of these once per gameweek from ``training_history`` via each component's own ``fit_*``
    function (``fit_dixon_coles_rho``, ``fit_save_conversion_rate``, ``fit_away_shot_multiplier``,
    ``fit_overdispersion``, ``fit_penalty_conversion_rates``) — see ``backtest/run_season.py``.
    """

    dixon_coles_rho: float = DEFAULT_DIXON_COLES_RHO
    save_conversion_rate: float = DEFAULT_SAVE_CONVERSION_RATE
    away_shot_multiplier: float = DEFAULT_AWAY_SHOT_MULTIPLIER
    dc_overdispersion_alpha: Mapping[str, float] = field(
        default_factory=lambda: {
            DEF: DEFAULT_OVERDISPERSION,
            MID: DEFAULT_OVERDISPERSION,
            FWD: DEFAULT_OVERDISPERSION,
        }
    )
    penalty_conversion_rate_by_player: Mapping[int, float] = field(default_factory=dict)
    league_avg_penalty_conversion_rate: float = DEFAULT_PENALTY_CONVERSION_RATE
    # 0.0 = shrinkage disabled, matching project_goals'/project_assists' own opt-in default.
    shrinkage_k: float = 0.0


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


_SHARED_REQUIRED_COLUMNS = [
    "npxg_per_90",
    "xa_per_90",
    "team_xg_per_90",
    "team_xga_per_90",
    "opponent_xg_per_90",
    "opponent_xga_per_90",
    "league_avg_xga_per_90",
    "yellow_card_rate_per_90",
    "red_card_rate_per_90",
    *MINUTES_FEATURE_COLUMNS,
]
_OUTFIELD_ONLY_REQUIRED_COLUMNS = ["dc_per_90", "opponent_possession_share"]
_GK_ONLY_REQUIRED_COLUMNS = ["opponent_shots_on_target_per_90", "is_home"]


def _validate_no_nan_inputs(players: pd.DataFrame) -> None:
    """Fail loudly, listing exactly which ``(player_id, column)`` pairs are missing, rather than
    letting a NaN silently propagate to ``expected_points`` (ENGINE_IMPROVEMENTS_2.md C.2). A
    crosswalk miss or any other upstream gap otherwise produces a NaN total that sorts out of
    every ranking with no error anywhere — the backtest's own ``dropna`` masks this entirely, but
    the live path (this function) has no other guard against it."""
    offenders: list[tuple[int, str]] = []
    for _, row in players.iterrows():
        required = list(_SHARED_REQUIRED_COLUMNS)
        required += _GK_ONLY_REQUIRED_COLUMNS if row["position"] == GK else _OUTFIELD_ONLY_REQUIRED_COLUMNS
        for col in required:
            if col in row.index and pd.isna(row[col]):
                offenders.append((int(row["player_id"]), col))
    if offenders:
        raise ValueError(
            f"NaN in required projection input(s), player_id/column pairs: {offenders} — check "
            "upstream feature engineering (e.g. an unmatched ID-crosswalk player) for these "
            "players rather than letting the prediction silently disappear"
        )


def _project_one_player(
    player_id: int,
    position: str,
    gameweek: int,
    row: pd.Series,
    minutes_distribution,
    bonus_model: BonusModel,
    fitted_constants: FittedConstants,
) -> tuple[PlayerGameweekProjection, ComponentBreakdown, float, dict[str, float]]:
    """Returns (projection, breakdown, clean_sheet_probability, raw_components) — the probability
    and the ``raw_components`` dict (``p_clears_threshold``, ``expected_goals``,
    ``expected_assists``, ``expected_bonus``) are surfaced separately since they aren't otherwise
    recoverable from the breakdown alone (BUILD_PLAN 3.2's calibration check needs the raw
    probability/quantity, not the points it converted to — ENGINE_IMPROVEMENTS_2.md A.4 extends
    this from clean-sheet-only to every component)."""
    expected_minutes = minutes_distribution.expected_minutes

    # ENGINE_IMPROVEMENTS_2.md B.2: thin-sample rate shrinkage, opt-in via fitted_constants.shrinkage_k
    # (0.0 by default = disabled, matching each component's own project_*'s pre-B.2 behavior).
    # `understat_effective_minutes` defaults to 0.0 (full shrinkage toward the prior) when the row
    # doesn't carry it — the correct behavior for a player with no prior Understat history at all.
    understat_effective_minutes = float(row.get("understat_effective_minutes", 0.0))

    goals = project_goals(
        player_npxg_per_90=row["npxg_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
        team_expected_penalties=float(row.get("team_expected_penalties", 0.0)),
        taker_share=float(row.get("taker_share", 0.0)),
        penalty_conversion_rate=fitted_constants.penalty_conversion_rate_by_player.get(
            player_id, fitted_constants.league_avg_penalty_conversion_rate
        ),
        individual_weight=understat_effective_minutes,
        team_xg_per_90=row["team_xg_per_90"],
        shrinkage_k=fitted_constants.shrinkage_k,
    )
    assists = project_assists(
        player_xa_per_90=row["xa_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
        individual_weight=understat_effective_minutes,
        team_xg_per_90=row["team_xg_per_90"],
        shrinkage_k=fitted_constants.shrinkage_k,
    )
    clean_sheet = project_clean_sheet(
        team_xg_per_90=row["team_xg_per_90"],
        team_xga_per_90=row["team_xga_per_90"],
        opponent_xg_per_90=row["opponent_xg_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
        rho=fitted_constants.dixon_coles_rho,
        # Bucket-weighted goals-conceded expectation (ENGINE_IMPROVEMENTS_2.md B.1) rather than the
        # point-estimate `expected_minutes` above, which understates it via Jensen's inequality.
        p_1_to_59=minutes_distribution.p_1_to_59,
        minutes_given_1_to_59=minutes_distribution.expected_minutes_given_1_to_59,
        p_60_plus=minutes_distribution.p_60_plus,
        minutes_given_60_plus=minutes_distribution.expected_minutes_given_60_plus,
    )
    cards = project_cards(
        yellow_card_rate_per_90=row["yellow_card_rate_per_90"],
        red_card_rate_per_90=row["red_card_rate_per_90"],
        expected_minutes=expected_minutes,
    )
    # ENGINE_IMPROVEMENTS_2.md D.6: optional, like team_expected_penalties/taker_share below —
    # silently omitted (own goals not modelled) when the row doesn't carry this column, rather
    # than requiring every existing caller to supply it.
    own_goal_rate_per_90 = row.get("own_goal_rate_per_90")
    own_goals = (
        project_own_goals(float(own_goal_rate_per_90), expected_minutes)
        if own_goal_rate_per_90 is not None
        else None
    )

    p_clears_threshold = float("nan")  # DC isn't modelled for GK
    if position == GK:
        defensive_contribution = None
        saves = project_saves(
            opponent_shots_on_target_per_90=row["opponent_shots_on_target_per_90"],
            team_xga_per_90=row["team_xga_per_90"],
            league_avg_xga_per_90=row["league_avg_xga_per_90"],
            is_home=bool(row["is_home"]),
            expected_minutes=expected_minutes,
            save_conversion_rate=fitted_constants.save_conversion_rate,
            away_shot_multiplier=fitted_constants.away_shot_multiplier,
        )
        defensive_action_rate_for_bonus = 0.0
    else:
        defensive_contribution = project_defensive_contribution(
            position=position,
            player_actions_per_90=row["dc_per_90"],
            opponent_possession_share=row["opponent_possession_share"],
            expected_minutes=expected_minutes,
            alpha=fitted_constants.dc_overdispersion_alpha.get(position, DEFAULT_OVERDISPERSION),
            # Bucket-weighted threshold probability (ENGINE_IMPROVEMENTS_2.md B.1) — measured to
            # understate the true probability by ~34% overall at the point estimate, worst for
            # rotation-risk players.
            p_1_to_59=minutes_distribution.p_1_to_59,
            minutes_given_1_to_59=minutes_distribution.expected_minutes_given_1_to_59,
            p_60_plus=minutes_distribution.p_60_plus,
            minutes_given_60_plus=minutes_distribution.expected_minutes_given_60_plus,
        )
        p_clears_threshold = defensive_contribution.p_clears_threshold
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
        own_goals=own_goals,
    )
    projection = project_player_gameweek(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        minutes=minutes_distribution,
        breakdown=breakdown,
    )
    raw_components = {
        "p_clears_threshold": p_clears_threshold,
        "expected_goals": goals.expected_goals,
        "expected_assists": assists.expected_assists,
        "expected_bonus": bonus.expected_bonus,
    }
    return projection, breakdown, clean_sheet.clean_sheet_probability, raw_components


def project_gameweek_pool(
    players: pd.DataFrame,
    gameweek: int,
    minutes_model: MinutesModel,
    bonus_model: BonusModel,
    fitted_constants: FittedConstants | None = None,
) -> pd.DataFrame:
    """Run the full Phase 2 chain for every row of ``players`` (one row per player, columns per
    :data:`GAMEWEEK_POOL_COLUMNS`) and return one prediction row per player.

    The returned frame carries ``expected_points`` (the headline number) plus every line of the
    BUILD_PLAN 2.7 component breakdown, the minutes model's own bucket probabilities (``p_zero``,
    ``p_1_to_59``, ``p_60_plus``), ``clean_sheet_probability`` (team-level, NOT gated by whether
    this player individually plays 60+ minutes) and ``player_clean_sheet_probability`` (gated:
    ``clean_sheet_probability * p_60_plus``, the like-for-like quantity to compare against FPL's
    own player-level ``clean_sheets`` outcome column — see ENGINE_IMPROVEMENTS.md Correction 1,
    which found comparing the *ungated* team probability against the *gated* actual outcome
    produced a spurious ~2.4x miscalibration signal that vanished once compared correctly). These
    are the same fields the web app's "detail on click" view and Phase 3's calibration checks both
    need, so callers don't have to re-derive them from a separate call.

    ``fitted_constants``, if given, supplies per-gameweek-refit values for the five component
    constants BUILD_PLAN 2.8's regression layer was meant to fit (ENGINE_IMPROVEMENTS.md Tier 1.2)
    — omitting it reproduces every component's own untuned ``DEFAULT_*`` behavior unchanged. Two
    further optional per-row columns activate the goals model's penalty sub-model when present
    (silently defaulting to 0.0, i.e. disabled, when absent): ``team_expected_penalties`` and
    ``taker_share`` (see ``engine/models/goals.py``'s ``realized_penalty_goals``/
    ``fit_penalty_conversion_rates`` for how to derive these from real data).
    """
    if players.empty:
        raise ValueError("players must not be empty")
    _validate_no_nan_inputs(players)
    fitted_constants = fitted_constants or FittedConstants()

    minutes_distributions = minutes_model.predict(players)
    rows = []
    for (_, row), minutes_distribution in zip(
        players.iterrows(), minutes_distributions, strict=True
    ):
        player_id = int(row["player_id"])
        position = row["position"]
        _projection, breakdown, clean_sheet_probability, raw_components = _project_one_player(
            player_id, position, gameweek, row, minutes_distribution, bonus_model, fitted_constants
        )
        rows.append(
            {
                "player_id": player_id,
                "position": position,
                "gameweek": gameweek,
                "expected_points": breakdown.total,
                "expected_minutes": minutes_distribution.expected_minutes,
                "expected_minutes_given_1_to_59": minutes_distribution.expected_minutes_given_1_to_59,
                "expected_minutes_given_60_plus": minutes_distribution.expected_minutes_given_60_plus,
                "p_zero": minutes_distribution.p_zero,
                "p_1_to_59": minutes_distribution.p_1_to_59,
                "p_60_plus": minutes_distribution.p_60_plus,
                "clean_sheet_probability": clean_sheet_probability,
                "player_clean_sheet_probability": (
                    clean_sheet_probability * minutes_distribution.p_60_plus
                ),
                "p_clears_threshold": raw_components["p_clears_threshold"],
                "expected_goals": raw_components["expected_goals"],
                "expected_assists": raw_components["expected_assists"],
                "expected_bonus": raw_components["expected_bonus"],
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
                "own_goals": breakdown.own_goals,
            }
        )
    return pd.DataFrame(rows)
