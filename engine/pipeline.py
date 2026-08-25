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

import numpy as np
import pandas as pd

from engine.aggregate import ComponentBreakdown, aggregate_gameweek
from engine.models.assists import DEFAULT_ASSIST_SHARE_OF_TEAM_XG, project_assists
from engine.models.bonus import (
    MAX_BONUS,
    BonusModel,
    BonusProjection,
    build_features,
    expected_bonus_from_fixture_strengths,
)
from engine.models.cards import project_cards, project_own_goals
from engine.models.clean_sheets import DEFAULT_DIXON_COLES_RHO, project_clean_sheet
from engine.models.defensive_contribution import (
    DEFAULT_OVERDISPERSION,
    project_defensive_contribution,
)
from engine.models.goals import DEFAULT_PENALTY_CONVERSION_RATE, project_goals
from engine.models.minutes import FEATURE_COLUMNS as MINUTES_FEATURE_COLUMNS
from engine.models.minutes import MinutesDistribution, MinutesModel
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
    # ENGINE_IMPROVEMENTS_5.md Tier 2.3: per-position multipliers converting the xG/xA-derived rate
    # into the quantity FPL actually awards. Empty (the default) means 1.0 everywhere, i.e. the
    # pre-Tier-2.3 behaviour. Fitted in `backtest.run_season.fit_fn` from training history only.
    #
    # These correct two *opposite* real errors that the gate's played-rows calibration could not
    # see, because restricting to players who played selects the branch on which an unconditional
    # expectation always looks low, masking an over-prediction and exaggerating an
    # under-prediction. Evaluated at each row's realised minutes instead, the 2025/26 walk-forward
    # showed goals running 18% hot and assists 12% cold; the all-rows mean calibration agreed
    # (goals 24% over, assists 8% under), and only the played-rows figure dissented.
    goal_conversion_factor_by_position: Mapping[str, float] = field(default_factory=dict)
    assist_conversion_factor_by_position: Mapping[str, float] = field(default_factory=dict)


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

# Optional, not required: "team" and "opponent_team_name" (ENGINE_IMPROVEMENTS_3.md D.2 / T-G).
# When both are present, project_gameweek_pool redistributes bonus across each real fixture's
# players with expected_bonus_from_fixture_strengths instead of using BonusModel's raw per-player
# prediction directly as points. See _distribute_bonus_by_fixture. Same silent-default convention
# as the other optional columns above (own_goal_rate_per_90, understat_effective_minutes, ...):
# omitting them reproduces the pre-T-G per-player behavior unchanged, since there is no fixture to
# group by. backtest/run_season.py's engineered pool already carries both columns (see
# simulate_gameweek_pool's identical (team, opponent_team_name) grouping).


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


@dataclass
class _PlayerComponents:
    """Every per-player component computed before bonus is finalized (T-G). Bonus is the one
    component that cannot be settled player-by-player, since a real fixture's 3/2/1 depends on
    every other player in that match. Split out of what used to be one ``_project_one_player``
    function so ``project_gameweek_pool`` can gather every player's raw bonus "strength" across a
    whole fixture (:func:`_distribute_bonus_by_fixture`) before committing to a final per-player
    bonus value and only then building the aggregate breakdown."""

    goals: object
    assists: object
    clean_sheet: object
    cards: object
    own_goals: object | None
    defensive_contribution: object | None
    saves: object | None
    p_clears_threshold: float
    bonus_features: dict[str, float]


def _compute_player_components(
    player_id: int,
    position: str,
    row: pd.Series,
    minutes_distribution,
    fitted_constants: FittedConstants,
) -> _PlayerComponents:
    """Runs every component (2.1-2.6) except bonus itself, and assembles the bonus regression's
    own feature row (``build_features``) without yet calling ``bonus_model.predict`` on it, since
    the caller batches that prediction and the fixture-level redistribution across the whole pool.
    """
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
        conversion_factor=fitted_constants.goal_conversion_factor_by_position.get(position, 1.0),
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
        conversion_factor=fitted_constants.assist_conversion_factor_by_position.get(position, 1.0),
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

    bonus_features = build_features(
        expected_goals=goals.expected_goals,
        expected_assists=assists.expected_assists,
        clean_sheet_probability=clean_sheet.clean_sheet_probability,
        defensive_action_rate=defensive_action_rate_for_bonus,
        position=position,
        expected_minutes=expected_minutes,
    )

    return _PlayerComponents(
        goals=goals,
        assists=assists,
        clean_sheet=clean_sheet,
        cards=cards,
        own_goals=own_goals,
        defensive_contribution=defensive_contribution,
        saves=saves,
        p_clears_threshold=p_clears_threshold,
        bonus_features=bonus_features,
    )


# ENGINE_IMPROVEMENTS_3.md D.2 / T-G: floor added to a fixture's raw BonusModel "strengths" before
# they are handed to expected_bonus_from_fixture_strengths, matching backtest/diagnostics.py's own
# rank_based_bonus_diagnostics precedent. Keeps a strength of exactly 0.0 (a player the minutes
# model is confident won't play) from breaking the Plackett-Luce normalisation, while still leaving
# that player a vanishingly small, not literally zero, chance of a top-3 finish.
#
# ENGINE_IMPROVEMENTS_5.md Tier 2.2: lowered from 0.01 to 1e-6 after measuring what it actually
# cost. A fixture's contest spans both full squads, roughly 48 players, of whom only about 22 take
# the pitch. At 0.01 the ~26 non-players each sat at a floor comparable to a real squad player's
# strength, so *collectively* they absorbed 17.4% of every fixture's 6.0 points, against 0.0% of
# real bonus. The floor only needs to be large enough to keep the normalisation well defined, not
# large enough to compete. Dropping it alone cuts the leak to 15.8%; combined with the availability
# weighting below it reaches 9.0%.
_BONUS_STRENGTH_FLOOR = 1e-6

# ENGINE_IMPROVEMENTS_5.md Tier 2.1: availability floor used when scaling a player's fixture bonus
# allocation up to its "if they start" counterpart. Guards the division for a player the minutes
# model is certain will not feature, whose conditional figure is a counterfactual about an event of
# probability ~0 and so is not meaningfully estimable either way.
_MIN_AVAILABILITY_FOR_CONDITIONAL = 0.02


def _plays_60_counterfactual(distribution: MinutesDistribution) -> MinutesDistribution:
    """The same player under "they definitely start and see out 60+ minutes"
    (ENGINE_IMPROVEMENTS_5.md Tier 2.1). Feeding this back through the component chain gives
    ``E[points | plays]``, the quantity a manager actually reasons about, as opposed to
    ``expected_points``, which is that number already multiplied by the chance they feature.

    A conditional expectation cannot be recovered by dividing the blended figure by ``p_60_plus``:
    the components are gated in three different ways (appearance and clean sheet are step functions
    of the 60-minute threshold, the rate components scale linearly with expected minutes, and
    ``p_1_to_59`` contributes to some but not others), so a single divisor over-inflates. Measured
    on the real 2025/26 walk-forward, de-gating appearance that way yields 3.35 points against a
    hard maximum of 2.0. Re-running the chain is the only correct route.

    ``expected_minutes_given_60_plus`` is carried through unchanged, so a player the model expects
    to be substituted on 70 minutes when they do start keeps that, rather than being credited with
    a flat 90.
    """
    return MinutesDistribution(
        p_zero=0.0,
        p_1_to_59=0.0,
        p_60_plus=1.0,
        expected_minutes_given_1_to_59=distribution.expected_minutes_given_1_to_59,
        expected_minutes_given_60_plus=distribution.expected_minutes_given_60_plus,
    )


def _distribute_bonus_by_fixture(
    players: pd.DataFrame,
    raw_bonus_by_player_id: Mapping[int, float],
    availability_by_player_id: Mapping[int, float] | None = None,
) -> dict[int, float]:
    """Turn BonusModel's independent per-player prediction into a true per-fixture 3/2/1
    allocation (T-G). Real FPL bonus is winner-take-all across the ~22 players in one match, not
    an absolute per-player threshold, so a linear regression clipped to [0, 3] and used directly as
    points cannot reproduce that concentration (the whole gameweek's maximum was 0.61 against a
    real elite match performance of roughly 1.0 to 1.5). BonusModel's raw prediction is kept as
    each player's Plackett-Luce "strength", a proxy for who is most likely to top that fixture's
    BPS, and :func:`~engine.models.bonus.expected_bonus_from_fixture_strengths` converts the whole
    fixture's strengths into an expected 3/2/1 that sums to 6.0 per fixture by construction.

    Grouping mirrors ``backtest/run_season.py``'s ``simulate_gameweek_pool``: fixtures are found
    via ``(team, opponent_team_name)``, each team processed exactly once. When either column is
    absent from ``players`` (for example a synthetic pool in a unit test with no fixture context,
    or a horizon/backtest caller that hasn't supplied them) every player's raw prediction is
    returned unchanged, the same silent-default convention as this module's other optional
    columns, since there is no fixture to redistribute across. A team whose opponent doesn't also
    appear in this pool (a data gap) is likewise left at its raw prediction rather than
    redistributed across an incomplete fixture, which would misallocate the real match's full 6.0
    across fewer than the ~22 players who actually contested it.

    ``availability_by_player_id`` (ENGINE_IMPROVEMENTS_5.md Tier 2.2), if given, is each player's
    ``P(60+ minutes)`` and multiplies their strength before the contest. The contest necessarily
    spans both full *squads* (~48 players) because a lineup isn't known in advance, but only ~22
    play, and an unweighted contest hands the other ~26 a real share: measured on the 2025/26
    walk-forward, 17.4% of all predicted bonus landed on players who never appeared, against 0.0%
    of actual bonus. Weighting by availability, together with the much smaller
    :data:`_BONUS_STRENGTH_FLOOR`, cuts that to 9.0%, raises the played-60+ calibration ratio from
    0.623 to 0.749, and increases the gap between a player who earned 3 bonus and one who earned
    none from +0.117 to +0.173 points, with rank correlation against realised bonus holding at
    0.240.

    Deliberately *not* done by excluding the unlikely-to-play tail from the contest. Restricting to
    the top 14 per side scores marginally better on leak and separation but assigns a hard 0.0 to
    6,822 rows, requires an arbitrary cutoff, and measured *worse* on rank correlation (0.236). A
    deep-squad player who unexpectedly starts and tops BPS should come out small, not impossible.
    Omitted (the default), strengths are unweighted, exactly this function's pre-Tier-2.2 behaviour.
    """
    if "team" not in players.columns or "opponent_team_name" not in players.columns:
        return dict(raw_bonus_by_player_id)

    final_bonus = dict(raw_bonus_by_player_id)
    processed_teams: set = set()
    for team, group in players.groupby("team"):
        if team in processed_teams:
            continue
        opponent = group["opponent_team_name"].iloc[0]
        opponent_group = players[players["team"] == opponent]
        processed_teams.add(team)
        if opponent_group.empty:
            continue
        processed_teams.add(opponent)

        fixture_player_ids = (
            pd.concat([group["player_id"], opponent_group["player_id"]]).astype(int).to_numpy()
        )
        strengths = np.array(
            [raw_bonus_by_player_id[pid] for pid in fixture_player_ids], dtype=float
        ).clip(min=0.0)
        if availability_by_player_id is not None:
            strengths = strengths * np.array(
                [availability_by_player_id[pid] for pid in fixture_player_ids], dtype=float
            ).clip(min=0.0)
        strengths = strengths + _BONUS_STRENGTH_FLOOR
        expected = expected_bonus_from_fixture_strengths(strengths)
        for player_id, value in zip(fixture_player_ids, expected, strict=True):
            final_bonus[int(player_id)] = float(value)
    return final_bonus


def _finalize_player(
    player_id: int,
    position: str,
    gameweek: int,
    minutes_distribution,
    components: _PlayerComponents,
    bonus: BonusProjection,
) -> tuple[PlayerGameweekProjection, ComponentBreakdown, float, dict[str, float]]:
    """Combines one player's already-computed components with its final (post fixture
    redistribution) bonus value into the aggregate breakdown and top-level projection. Returns
    (projection, breakdown, clean_sheet_probability, raw_components). The probability and the
    ``raw_components`` dict (``p_clears_threshold``, ``expected_goals``, ``expected_assists``,
    ``expected_bonus``) are surfaced separately since they aren't otherwise recoverable from the
    breakdown alone (BUILD_PLAN 3.2's calibration check needs the raw probability/quantity, not
    the points it converted to. ENGINE_IMPROVEMENTS_2.md A.4 extends this from clean-sheet-only
    to every component)."""
    breakdown = aggregate_gameweek(
        position,
        minutes_distribution,
        components.goals,
        components.assists,
        components.clean_sheet,
        bonus,
        components.cards,
        defensive_contribution=components.defensive_contribution,
        saves=components.saves,
        own_goals=components.own_goals,
    )
    projection = project_player_gameweek(
        player_id=player_id,
        position=position,
        gameweek=gameweek,
        minutes=minutes_distribution,
        breakdown=breakdown,
    )
    raw_components = {
        "p_clears_threshold": components.p_clears_threshold,
        "expected_goals": components.goals.expected_goals,
        "expected_assists": components.assists.expected_assists,
        "expected_bonus": bonus.expected_bonus,
        # NaN for outfield players (saves isn't modelled for them), mirroring p_clears_threshold's
        # own GK/outfield split above — Phase 3's saves calibration check needs the raw expected
        # count, not `breakdown.saves`'s already-floor-divided points.
        "expected_saves": (
            components.saves.expected_saves if components.saves is not None else float("nan")
        ),
    }
    return projection, breakdown, components.clean_sheet.clean_sheet_probability, raw_components


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

    ``bonus`` (and the raw ``expected_bonus`` column) is a true per-fixture 3/2/1 allocation, not
    ``bonus_model``'s independent per-player prediction used directly (T-G, ENGINE_IMPROVEMENTS_3.md
    D.2). See ``_distribute_bonus_by_fixture`` for how, and for the optional ``team``/
    ``opponent_team_name`` columns that activate it.
    """
    if players.empty:
        raise ValueError("players must not be empty")
    _validate_no_nan_inputs(players)
    fitted_constants = fitted_constants or FittedConstants()

    minutes_distributions = minutes_model.predict(players)
    player_ids: list[int] = []
    positions: list[str] = []
    components_by_player: dict[int, _PlayerComponents] = {}
    minutes_by_player: dict[int, object] = {}
    bonus_feature_rows: list[dict[str, float]] = []
    # Tier 2.1: the same chain re-run under "they definitely play", giving E[points | plays]
    # alongside the availability-weighted expected_points. See _plays_60_counterfactual.
    conditional_components_by_player: dict[int, _PlayerComponents] = {}
    conditional_minutes_by_player: dict[int, MinutesDistribution] = {}
    for (_, row), minutes_distribution in zip(
        players.iterrows(), minutes_distributions, strict=True
    ):
        player_id = int(row["player_id"])
        position = row["position"]
        components = _compute_player_components(
            player_id, position, row, minutes_distribution, fitted_constants
        )
        player_ids.append(player_id)
        positions.append(position)
        components_by_player[player_id] = components
        minutes_by_player[player_id] = minutes_distribution
        bonus_feature_rows.append(components.bonus_features)

        conditional_minutes = _plays_60_counterfactual(minutes_distribution)
        conditional_components = _compute_player_components(
            player_id, position, row, conditional_minutes, fitted_constants
        )
        conditional_components_by_player[player_id] = conditional_components
        conditional_minutes_by_player[player_id] = conditional_minutes

    # Bonus is batched across the whole pool, not per player like every other component: it is
    # the one component whose final value depends on other players in the same fixture (T-G).
    raw_bonus = bonus_model.predict(pd.DataFrame(bonus_feature_rows))
    raw_bonus_by_player = dict(
        zip(player_ids, (projection.expected_bonus for projection in raw_bonus), strict=True)
    )
    # Tier 2.2: the contest spans both full squads, so it is weighted by each player's own
    # P(60+ minutes) to stop the ~26 non-players per fixture absorbing a real share of its 6.0.
    availability_by_player = {
        player_id: float(distribution.p_60_plus)
        for player_id, distribution in minutes_by_player.items()
    }
    final_bonus_by_player = _distribute_bonus_by_fixture(
        players, raw_bonus_by_player, availability_by_player
    )
    # Tier 2.1: the conditional bonus is *not* a second fixture contest. A contest allocates one
    # fixture's fixed 6.0 among players who are competing with each other, whereas the conditional
    # figures are one mutually-exclusive counterfactual per player ("if this player starts"), so
    # they carry no obligation to sum to 6.0 and forcing them to would be wrong.
    #
    # Within a contest, expected bonus is very nearly proportional to strength while any one
    # player's strength is small next to the fixture total, so raising this player's availability
    # weight from `p_60_plus` to 1.0 while leaving the other ~47 untouched scales their allocation
    # by `1 / p_60_plus` to first order. That is what is applied here, clipped to the model's own
    # MAX_BONUS. Running a genuine per-player contest instead would mean ~48 Plackett-Luce
    # evaluations per fixture for a second-order correction to a component worth 6.6% of predicted
    # points.
    conditional_bonus_by_player = {
        player_id: min(
            MAX_BONUS,
            final_bonus_by_player[player_id]
            / max(float(minutes_by_player[player_id].p_60_plus), _MIN_AVAILABILITY_FOR_CONDITIONAL),
        )
        for player_id in player_ids
    }

    rows = []
    for player_id, position in zip(player_ids, positions, strict=True):
        minutes_distribution = minutes_by_player[player_id]
        bonus = BonusProjection(expected_bonus=final_bonus_by_player[player_id])
        _projection, breakdown, clean_sheet_probability, raw_components = _finalize_player(
            player_id,
            position,
            gameweek,
            minutes_distribution,
            components_by_player[player_id],
            bonus,
        )
        _, conditional_breakdown, _, _ = _finalize_player(
            player_id,
            position,
            gameweek,
            conditional_minutes_by_player[player_id],
            conditional_components_by_player[player_id],
            BonusProjection(expected_bonus=conditional_bonus_by_player[player_id]),
        )
        rows.append(
            {
                "player_id": player_id,
                "position": position,
                "gameweek": gameweek,
                "expected_points": breakdown.total,
                # Tier 2.1: E[points | plays 60+]. `expected_points` is this multiplied by the
                # chance the player features, which is the right number to sum over a squad but the
                # wrong one to rank a shortlist by, since it conflates "good" with "nailed on".
                "conditional_expected_points": conditional_breakdown.total,
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
