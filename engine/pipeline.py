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
from engine.models.assists import DEFAULT_ASSIST_SHARE_OF_TEAM_XG, project_assists
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
    project_saves_from_own_rate,
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
    carries ``goals_shrinkage_k``/``assists_shrinkage_k`` (ENGINE_IMPROVEMENTS_2.md B.2, split per
    ENGINE_IMPROVEMENTS_4.md), the goals/assists thin-sample-rate shrinkage strengths — unlike the
    five Tier 1.2 constants, these have no closed-form per-gameweek estimator, so each is an
    evidence-based fixed hyperparameter selected once via an end-to-end backtest sweep
    (``backtest/run_season.py``'s ``GOALS_SHRINKAGE_K``/``ASSISTS_SHRINKAGE_K``) rather than refit
    every gameweek. Kept as two separate fields, not one shared value, because a real sweep found
    the two components disagree sharply on what it should be (assists' own-rate calibration keeps
    improving all the way to near-total shrinkage, goals' does not).

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
    # ENGINE_IMPROVEMENTS_4.md: previously one shared `shrinkage_k` fed both goals and assists,
    # which forced a single compromise value even though a real walk-forward sweep found the two
    # components disagree sharply on what that value should be -- assists' played-only mean
    # calibration goes from -22.6% (k=0, no shrinkage) to +1.9% (k=1000, fully shrunk to the
    # team-xG-derived prior) while goals' optimum sits at k~20-30. Split so each component can be
    # swept and set independently. 0.0 = shrinkage disabled, matching project_goals'/
    # project_assists' own opt-in default.
    goals_shrinkage_k: float = 0.0
    assists_shrinkage_k: float = 0.0
    # ENGINE_IMPROVEMENTS_4.md: per-position empirical replacement for assists.py's flat
    # DEFAULT_ASSIST_SHARE_OF_TEAM_XG (0.12 for every position) — the actual defect a real sweep
    # found behind a severe FWD over-prediction bias once assists_shrinkage_k was raised enough to
    # fix assists' aggregate calibration: shrinking every position toward the SAME flat prior
    # inflates whichever position's true assist-per-team-xG share is lower than that constant.
    # Defaults to the component's own flat constant per position, via fields(default_factory=...)
    # since a dict comprehension needs `engine.models.assists.DEFAULT_ASSIST_SHARE_OF_TEAM_XG` and
    # `engine.scoring`'s position names, both already imported below.
    assist_share_of_team_xg_by_position: Mapping[str, float] = field(
        default_factory=lambda: {
            DEF: DEFAULT_ASSIST_SHARE_OF_TEAM_XG,
            MID: DEFAULT_ASSIST_SHARE_OF_TEAM_XG,
            FWD: DEFAULT_ASSIST_SHARE_OF_TEAM_XG,
        }
    )
    # ENGINE_IMPROVEMENTS_3.md A.3: per-position league-average card/own-goal rates and their
    # shrinkage strengths — same opt-in-disabled-by-default shape as `shrinkage_k` above. Red and
    # own-goal shrinkage strengths are typically much larger than yellow's, since both are rare
    # enough that a single observation is not a reliable individual rate.
    league_avg_yellow_card_rate_by_position: Mapping[str, float] = field(default_factory=dict)
    league_avg_red_card_rate_by_position: Mapping[str, float] = field(default_factory=dict)
    league_avg_own_goal_rate_by_position: Mapping[str, float] = field(default_factory=dict)
    yellow_card_shrinkage_k: float = 0.0
    red_card_shrinkage_k: float = 0.0
    own_goal_shrinkage_k: float = 0.0
    # ENGINE_IMPROVEMENTS_3.md D.1: shrinkage prior/strength for the goalkeeper own-rate saves
    # fallback (`project_saves_from_own_rate`) — `save_conversion_rate`/`away_shot_multiplier`
    # above remain for `project_saves`'s opponent-adjusted formula, still blocked on real
    # opponent shots-on-target data and not currently called by this module.
    league_avg_save_rate_per_90: float = 0.0
    save_rate_shrinkage_k: float = 0.0


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
    #   "own_save_rate_per_90" (ENGINE_IMPROVEMENTS_3.md D.1 fallback — see
    #   project_saves_from_own_rate; the "real" opponent-adjusted project_saves needs
    #   "opponent_shots_on_target_per_90"/"is_home", still blocked on real shot data and not
    #   required here)
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
_GK_ONLY_REQUIRED_COLUMNS = ["own_save_rate_per_90"]


def _validate_no_nan_inputs(players: pd.DataFrame) -> None:
    """Fail loudly, listing exactly which ``(player_id, column)`` pairs are missing, rather than
    letting a NaN silently propagate to ``expected_points`` (ENGINE_IMPROVEMENTS_2.md C.2). A
    crosswalk miss or any other upstream gap otherwise produces a NaN total that sorts out of
    every ranking with no error anywhere — the backtest's own ``dropna`` masks this entirely, but
    the live path (this function) has no other guard against it."""
    offenders: list[tuple[int, str]] = []
    for _, row in players.iterrows():
        required = list(_SHARED_REQUIRED_COLUMNS)
        required += (
            _GK_ONLY_REQUIRED_COLUMNS if row["position"] == GK else _OUTFIELD_ONLY_REQUIRED_COLUMNS
        )
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

    # ENGINE_IMPROVEMENTS_2.md B.2: thin-sample rate shrinkage, opt-in via
    # fitted_constants.goals_shrinkage_k / assists_shrinkage_k (ENGINE_IMPROVEMENTS_4.md split)
    # (0.0 by default = disabled, matching each component's own project_*'s pre-B.2 behavior).
    # `understat_effective_minutes` defaults to 0.0 (full shrinkage toward the prior) when the row
    # doesn't carry it — the correct behavior for a player with no prior Understat history at all.
    #
    # ENGINE_IMPROVEMENTS_3.md D.1: goalkeepers never have real Understat history (the crosswalk
    # doesn't try to match them), so `understat_effective_minutes` is always 0.0 for GK — which
    # would otherwise mean shrinkage falls back to the team-xG-derived *prior* rate every single
    # gameweek (an average outfield player's ~12% share of team xG) rather than a keeper's real
    # near-zero goal/assist involvement. Shrinkage is disabled entirely for GK (`individual_weight
    # =None`) so `npxg_per_90`/`xa_per_90`'s explicit 0.0 (see `engineer_features`) is used as-is.
    understat_effective_minutes = (
        float(row.get("understat_effective_minutes", 0.0)) if position != GK else None
    )

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
        shrinkage_k=fitted_constants.goals_shrinkage_k,
    )
    assists = project_assists(
        player_xa_per_90=row["xa_per_90"],
        opponent_xga_per_90=row["opponent_xga_per_90"],
        league_avg_xga_per_90=row["league_avg_xga_per_90"],
        expected_minutes=expected_minutes,
        individual_weight=understat_effective_minutes,
        team_xg_per_90=row["team_xg_per_90"],
        shrinkage_k=fitted_constants.assists_shrinkage_k,
        assist_share_of_team_xg=fitted_constants.assist_share_of_team_xg_by_position.get(
            position, DEFAULT_ASSIST_SHARE_OF_TEAM_XG
        ),
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
    # ENGINE_IMPROVEMENTS_3.md A.3: point-in-time evidence weight behind this row's own card/own-
    # goal rates, the shrinkage target for thin-sample outliers — defaults to 0.0 (full shrinkage
    # toward the prior) when the row doesn't carry it, same convention as
    # `understat_effective_minutes` above.
    card_effective_minutes = float(row.get("card_effective_minutes", 0.0))
    cards = project_cards(
        yellow_card_rate_per_90=row["yellow_card_rate_per_90"],
        red_card_rate_per_90=row["red_card_rate_per_90"],
        expected_minutes=expected_minutes,
        individual_weight=card_effective_minutes,
        league_avg_yellow_card_rate_per_90=fitted_constants.league_avg_yellow_card_rate_by_position.get(
            position
        ),
        league_avg_red_card_rate_per_90=fitted_constants.league_avg_red_card_rate_by_position.get(
            position
        ),
        yellow_shrinkage_k=fitted_constants.yellow_card_shrinkage_k,
        red_shrinkage_k=fitted_constants.red_card_shrinkage_k,
    )
    # ENGINE_IMPROVEMENTS_2.md D.6: optional, like team_expected_penalties/taker_share below —
    # silently omitted (own goals not modelled) when the row doesn't carry this column, rather
    # than requiring every existing caller to supply it.
    own_goal_rate_per_90 = row.get("own_goal_rate_per_90")
    own_goals = (
        project_own_goals(
            float(own_goal_rate_per_90),
            expected_minutes,
            individual_weight=card_effective_minutes,
            league_avg_own_goal_rate_per_90=fitted_constants.league_avg_own_goal_rate_by_position.get(
                position
            ),
            shrinkage_k=fitted_constants.own_goal_shrinkage_k,
        )
        if own_goal_rate_per_90 is not None
        else None
    )

    p_clears_threshold = float("nan")  # DC isn't modelled for GK
    if position == GK:
        defensive_contribution = None
        # ENGINE_IMPROVEMENTS_3.md D.1: own-rate fallback, not the opponent-adjusted
        # `project_saves` (still blocked on real opponent shots-on-target data) — see that
        # function's own docstring.
        saves = project_saves_from_own_rate(
            own_save_rate_per_90=row["own_save_rate_per_90"],
            expected_minutes=expected_minutes,
            individual_weight=card_effective_minutes,
            league_avg_save_rate_per_90=fitted_constants.league_avg_save_rate_per_90,
            shrinkage_k=fitted_constants.save_rate_shrinkage_k,
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
                expected_minutes=expected_minutes,
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
        # NaN for outfield players (saves isn't modelled for them), mirroring p_clears_threshold's
        # own GK/outfield split above — Phase 3's saves calibration check needs the raw expected
        # count, not `breakdown.saves`'s already-floor-divided points.
        "expected_saves": saves.expected_saves if saves is not None else float("nan"),
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
                "expected_minutes_given_1_to_59": (
                    minutes_distribution.expected_minutes_given_1_to_59
                ),
                "expected_minutes_given_60_plus": (
                    minutes_distribution.expected_minutes_given_60_plus
                ),
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
                "expected_saves": raw_components["expected_saves"],
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
