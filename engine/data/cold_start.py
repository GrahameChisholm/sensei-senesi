"""Position x price baseline projections for players with no reconstructable engine history
(the team-selection page's D5: every player at a newly-promoted club, or a new signing from a
league Understat doesn't cover, at a season's true opening gameweek).

``backtest.run_season.engineer_features``' dropna is *correct* to exclude these players from a
real projection — ENGINE_IMPROVEMENTS_2.md C.1/C.2 is explicit that a missing rate must never be
defaulted to zero, since that would silently understate a real gap as a real (if low) number.
But for a team-selection page, silently vanishing every player from three entire clubs is not
acceptable either — so those players get an honest, clearly-labelled positional prior instead,
fitted from real prior-season data using this engine's own scoring constants
(:mod:`engine.scoring`, the "single source of truth for FPL scoring rules" — nothing here
hardcodes a point value independently), never hand-tuned.

Deliberately fits from the **raw** prior-season ``merged_gw`` frame (the same shape
``backtest.run_season.fetch_vaastav_merged_gw``/the cached parquet already provide), not the
engineered/dropna'd training frame — the whole point is to cover exactly the players that frame
excludes, so fitting only on its survivors would be circular.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from engine.aggregate import ComponentBreakdown
from engine.models.minutes import MinutesDistribution
from engine.projections import PlayerGameweekProjection, project_player_gameweek
from engine.scoring import (
    APPEARANCE_POINTS,
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS,
    DEFENSIVE_CONTRIBUTION_POINTS,
    DEFENSIVE_CONTRIBUTION_THRESHOLD,
    GK,
    GOAL_POINTS,
    GOALS_CONCEDED_PENALTY,
    GOALS_CONCEDED_PER_PENALTY,
    GOALS_CONCEDED_POSITIONS,
    OWN_GOAL_POINTS,
    PENALTY_MISS_POINTS,
    RED_CARD_POINTS,
    SAVES_PER_POINT,
    YELLOW_CARD_POINTS,
)

__all__ = [
    "PRICE_BUCKET_WIDTH",
    "ColdStartPriors",
    "fit_cold_start_priors",
    "baseline_projection",
]

# £0.5m bands (prices are tenths of a million, FPL's own now_cost convention) -- fine enough to
# separate a budget enabler from a premium pick within one position, coarse enough that a real
# prior season still has more than a handful of player-gameweeks per bucket.
PRICE_BUCKET_WIDTH = 5

_REQUIRED_COLUMNS = (
    "position",
    "value",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "defensive_contribution",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "penalties_missed",
    "own_goals",
)


def _price_bucket(price: int) -> int:
    return (int(price) // PRICE_BUCKET_WIDTH) * PRICE_BUCKET_WIDTH


def _rank_tier(rank: float) -> str:
    """Coarse tier for a player's price rank among their own club's players at the same position
    (1 = the most expensive, presumptively most nailed, player at that club and position). Coarse
    enough that even a squad carrying only two or three players in a position still has every tier
    represented somewhere in a real prior season, so the fitted groups stay adequately sized."""
    if rank <= 1:
        return "1"
    if rank <= 2:
        return "2"
    return "3+"


@dataclass(frozen=True)
class _PositionPriceStats:
    n_rows: int
    p_zero: float
    p_1_to_59: float
    p_60_plus: float
    expected_minutes_given_1_to_59: float
    expected_minutes_given_60_plus: float
    breakdown: ComponentBreakdown


def _stats_for_group(position: str, rows: pd.DataFrame) -> _PositionPriceStats:
    minutes = rows["minutes"].astype(float)
    zero = minutes == 0
    one_to_59 = (minutes > 0) & (minutes < 60)
    sixty_plus = minutes >= 60

    p_zero = float(zero.mean())
    p_1_to_59 = float(one_to_59.mean())
    p_60_plus = float(sixty_plus.mean())
    # A fallback default for a bucket with no rows in the relevant bucket at all (only possible if
    # every row is p_zero) -- 30/90 are the same "typical sub" / "typical starter" minutes the
    # engine's own minutes model defaults toward elsewhere.
    expected_minutes_given_1_to_59 = float(minutes[one_to_59].mean()) if one_to_59.any() else 30.0
    expected_minutes_given_60_plus = float(minutes[sixty_plus].mean()) if sixty_plus.any() else 90.0

    appearance = p_1_to_59 * APPEARANCE_POINTS["1-59"] + p_60_plus * APPEARANCE_POINTS["60+"]
    goals = float(rows["goals_scored"].astype(float).mean()) * GOAL_POINTS[position]
    assists = float(rows["assists"].astype(float).mean()) * ASSIST_POINTS
    clean_sheet = float(rows["clean_sheets"].astype(float).mean()) * CLEAN_SHEET_POINTS[position]
    goals_conceded = (
        float(rows["goals_conceded"].astype(float).mean())
        / GOALS_CONCEDED_PER_PENALTY
        * GOALS_CONCEDED_PENALTY
        if position in GOALS_CONCEDED_POSITIONS
        else 0.0
    )
    # Not modelled for GK (engine.aggregate.aggregate_gameweek's own position gate). Checked
    # directly against the real cached 2025-26 merged_gw data: vaastav's own `defensive_contribution`
    # column is the raw per-match ACTION COUNT (tackles/interceptions/blocks/clearances, observed
    # 0-29+ in that data), not FPL's recorded points for the component, matching what
    # engine.models.defensive_contribution already assumes for the main engine path. The flat
    # DEFENSIVE_CONTRIBUTION_POINTS bonus is all-or-nothing per match (BUILD_PLAN scoring.py), so
    # the empirical prior is the observed rate of clearing the position's action threshold, not the
    # mean action count itself.
    defensive_contribution = (
        float(
            (
                rows["defensive_contribution"].astype(float)
                >= DEFENSIVE_CONTRIBUTION_THRESHOLD[position]
            ).mean()
        )
        * DEFENSIVE_CONTRIBUTION_POINTS
        if position != GK
        else 0.0
    )
    # Saves (GK only): vaastav's `saves` column is a raw save COUNT, so convert to the same
    # continuous per-gameweek expected-points unit the engine's own saves model already uses
    # (see engine/projections.py's own real sample: a fractional saves-points value, not an
    # integer per-row floor).
    saves = float(rows["saves"].astype(float).mean()) / SAVES_PER_POINT if position == GK else 0.0
    bonus = float(rows["bonus"].astype(float).mean())
    cards = (
        float(rows["yellow_cards"].astype(float).mean()) * YELLOW_CARD_POINTS
        + float(rows["red_cards"].astype(float).mean()) * RED_CARD_POINTS
    )
    penalty_misses = float(rows["penalties_missed"].astype(float).mean()) * PENALTY_MISS_POINTS
    own_goals = float(rows["own_goals"].astype(float).mean()) * OWN_GOAL_POINTS

    breakdown = ComponentBreakdown(
        appearance=appearance,
        goals=goals,
        assists=assists,
        clean_sheet=clean_sheet,
        goals_conceded=goals_conceded,
        defensive_contribution=defensive_contribution,
        saves=saves,
        bonus=bonus,
        cards=cards,
        penalty_misses=penalty_misses,
        own_goals=own_goals,
    )
    return _PositionPriceStats(
        n_rows=len(rows),
        p_zero=p_zero,
        p_1_to_59=p_1_to_59,
        p_60_plus=p_60_plus,
        expected_minutes_given_1_to_59=expected_minutes_given_1_to_59,
        expected_minutes_given_60_plus=expected_minutes_given_60_plus,
        breakdown=breakdown,
    )


@dataclass(frozen=True)
class ColdStartPriors:
    """Fitted position x price-bucket baselines — never hand-tuned constants (this repo's own
    "explainable, fitted, not hand-tuned" discipline). :func:`baseline_projection` falls back to
    a position-only aggregate when a specific (position, price bucket) combination has no real
    prior-season rows (a live price band that never existed last season), and prefers the more
    specific ``by_position_bucket_and_rank`` cut over ``by_position_and_bucket`` whenever a caller
    supplies a within-club price rank and that exact combination was observed."""

    by_position_and_bucket: dict[tuple[str, int], _PositionPriceStats]
    by_position: dict[str, _PositionPriceStats]
    # Within-club price-rank cut of the same prior season, keyed (position, price bucket, tier
    # from _rank_tier). Empty when the fitting frame carries no club identity (``team``/``element``
    # columns) -- callers that never pass a rank to :func:`baseline_projection`, and the handful of
    # existing tests that fit from a club-less fixture, are unaffected either way, since lookups
    # against this dict only happen when a rank was actually supplied.
    by_position_bucket_and_rank: dict[tuple[str, int, str], _PositionPriceStats] = field(
        default_factory=dict
    )


def fit_cold_start_priors(prior_merged_gw: pd.DataFrame) -> ColdStartPriors:
    """Fit position x price-bucket baselines from one real prior season's raw ``merged_gw`` rows.

    Raises if a required raw column is absent — a wiring bug upstream (a different vaastav export
    shape), not a legitimate "no data" state this should silently degrade for.
    """
    missing = [c for c in _REQUIRED_COLUMNS if c not in prior_merged_gw.columns]
    if missing:
        raise ValueError(f"prior_merged_gw is missing expected column(s): {missing}")

    df = prior_merged_gw.copy()
    df["_price_bucket"] = df["value"].astype(int).map(_price_bucket)

    by_position_and_bucket: dict[tuple[str, int], _PositionPriceStats] = {}
    for (position, bucket), rows in df.groupby(["position", "_price_bucket"]):
        by_position_and_bucket[(position, int(bucket))] = _stats_for_group(position, rows)

    by_position: dict[str, _PositionPriceStats] = {}
    for position, rows in df.groupby("position"):
        by_position[position] = _stats_for_group(position, rows)

    # Within-club price-rank cut (see module docstring and _rank_tier). Only possible when the
    # fitting frame carries club identity -- ``team``/``element``, vaastav's raw column names for a
    # player's club and their FPL id -- which the club-less fixtures some existing tests fit from
    # deliberately don't have. Absent that, this stays empty and baseline_projection's fallback
    # chain behaves exactly as it did before this differentiator existed.
    by_position_bucket_and_rank: dict[tuple[str, int, str], _PositionPriceStats] = {}
    if "team" in df.columns and "element" in df.columns:
        player_price = df.groupby(["element", "team", "position"])["value"].mean().reset_index()
        player_price["_rank"] = player_price.groupby(["team", "position"])["value"].rank(
            ascending=False, method="first"
        )
        player_price["_rank_tier"] = player_price["_rank"].map(_rank_tier)
        ranked = df.merge(
            player_price[["element", "team", "position", "_rank_tier"]],
            on=["element", "team", "position"],
            how="left",
        )
        for (position, bucket, tier), rows in ranked.dropna(subset=["_rank_tier"]).groupby(
            ["position", "_price_bucket", "_rank_tier"]
        ):
            by_position_bucket_and_rank[(position, int(bucket), str(tier))] = _stats_for_group(
                position, rows
            )

    return ColdStartPriors(
        by_position_and_bucket=by_position_and_bucket,
        by_position=by_position,
        by_position_bucket_and_rank=by_position_bucket_and_rank,
    )


def baseline_projection(
    player_id: int,
    position: str,
    price: int,
    gameweek: int,
    priors: ColdStartPriors,
    within_club_position_rank: int | None = None,
) -> PlayerGameweekProjection:
    """The flagged, low-confidence projection for a player with no reconstructable history — the
    same real :class:`~engine.projections.PlayerGameweekProjection` shape every other consumer
    (pitch, panel, ``features.formation.select_starting_xi``, chips) already expects, with
    ``simulation=None`` (no Monte Carlo run backs this number) marking it as a prior rather than
    an engine output — callers key off that, plus their own ``source: "cold_start"`` cache field,
    to render the low-confidence UI marker (D5).

    ``within_club_position_rank`` is an optional price rank (1 = the most expensive, and so
    presumptively most nailed, player at that same club and position; see the module docstring)
    that the caller must compute across the club's own squad, since a single player's price and
    position alone can't reveal it. When supplied, and the fitted prior season has real rows for
    that exact (position, price bucket, rank tier) combination, it takes priority over the flatter
    position/price-bucket prior, so two cold-start players who share a price bucket, such as a
    promoted club's first- and third-choice striker, no longer collapse onto the same projection.
    Omitting it, or landing on a combination the prior season never saw, falls back to the
    position/price-bucket prior and then the position-only prior exactly as before.
    """
    bucket = _price_bucket(price)
    stats = None
    if within_club_position_rank is not None:
        tier = _rank_tier(within_club_position_rank)
        stats = priors.by_position_bucket_and_rank.get((position, bucket, tier))
    if stats is None:
        stats = priors.by_position_and_bucket.get((position, bucket))
    if stats is None:
        stats = priors.by_position.get(position)
    if stats is None:
        raise ValueError(f"no cold-start prior available for position {position!r}")

    minutes = MinutesDistribution(
        p_zero=stats.p_zero,
        p_1_to_59=stats.p_1_to_59,
        p_60_plus=stats.p_60_plus,
        expected_minutes_given_1_to_59=stats.expected_minutes_given_1_to_59,
        expected_minutes_given_60_plus=stats.expected_minutes_given_60_plus,
    )
    return project_player_gameweek(player_id, position, gameweek, minutes, stats.breakdown)
