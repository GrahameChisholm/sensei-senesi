"""Saves model (GK) — opponent shots on target -> expected saves -> points (2.6).

One of the smaller components (BUILD_PLAN 2.6): expected opponent shots on target, adjusted for
this team's own defensive strength and venue, converted to expected saves and then to points
(1 per 3, plus a penalty-save bonus).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from engine.models._discrete import expected_floor_division
from engine.rates import shrink_toward_prior
from engine.scoring import PENALTY_SAVE_POINTS, SAVES_PER_POINT

# Away goalkeepers face somewhat more shots/pressure on average (BUILD_PLAN 2.6 "Inputs" table).
# Not asserted precise -- Phase 3 calibrates against real data.
DEFAULT_AWAY_SHOT_MULTIPLIER = 1.1

# League-average fraction of shots on target that are saved (rather than scored) -- a rough
# placeholder pending Phase 3 calibration, not a per-keeper skill estimate.
DEFAULT_SAVE_CONVERSION_RATE = 0.67

# League-average penalty-save rate for a goalkeeper with no individual history to lean on.
DEFAULT_PENALTY_SAVE_RATE = 0.2


def expected_shots_faced(
    opponent_shots_on_target_per_90: float,
    team_xga_per_90: float,
    league_avg_xga_per_90: float,
    is_home: bool,
    away_shot_multiplier: float = DEFAULT_AWAY_SHOT_MULTIPLIER,
    expected_minutes: float = 90.0,
) -> float:
    """Expected shots on target faced this gameweek: the opponent's own shot volume, scaled up
    for a weaker defence (higher own team xGA drives keeper workload — BUILD_PLAN 2.6) and for
    playing away."""
    if league_avg_xga_per_90 <= 0:
        raise ValueError("league_avg_xga_per_90 must be positive")
    if opponent_shots_on_target_per_90 < 0 or team_xga_per_90 < 0 or expected_minutes < 0:
        raise ValueError("rates and expected_minutes must be non-negative")
    if away_shot_multiplier <= 0:
        raise ValueError("away_shot_multiplier must be positive")
    defensive_factor = team_xga_per_90 / league_avg_xga_per_90
    venue_factor = 1.0 if is_home else away_shot_multiplier
    return (
        opponent_shots_on_target_per_90
        * defensive_factor
        * venue_factor
        * (expected_minutes / 90.0)
    )


def expected_saves(
    shots_faced: float, save_conversion_rate: float = DEFAULT_SAVE_CONVERSION_RATE
) -> float:
    if shots_faced < 0:
        raise ValueError("shots_faced must be non-negative")
    if not 0.0 <= save_conversion_rate <= 1.0:
        raise ValueError("save_conversion_rate must be in [0, 1]")
    return shots_faced * save_conversion_rate


def fit_save_conversion_rate(
    shots_faced: pd.Series, saves: pd.Series, min_shots: float = 30.0
) -> float:
    """Empirical save conversion rate = total saves / total shots faced, refit every gameweek from
    real data (ENGINE_IMPROVEMENTS.md 1.2 — this was a hardcoded placeholder despite the
    regression layer existing to fit exactly this kind of rate). A simple ratio, not a regression
    coefficient — there's no per-position or per-feature structure here to fit, just a
    league-average rate. Falls back to :data:`DEFAULT_SAVE_CONVERSION_RATE` when the training
    sample's total shots faced is too thin to trust an empirical ratio.
    """
    total_shots = float(np.asarray(shots_faced, dtype=float).sum())
    if total_shots < min_shots:
        return DEFAULT_SAVE_CONVERSION_RATE
    total_saves = float(np.asarray(saves, dtype=float).sum())
    return float(np.clip(total_saves / total_shots, 0.0, 1.0))


def fit_away_shot_multiplier(
    home_shots_faced_per_90: pd.Series,
    away_shots_faced_per_90: pd.Series,
    min_rows: int = 20,
) -> float:
    """Empirical ratio of away-venue to home-venue shots faced per 90, refit every gameweek
    (ENGINE_IMPROVEMENTS.md 1.2). Falls back to :data:`DEFAULT_AWAY_SHOT_MULTIPLIER` when either
    side has too few rows to trust the ratio."""
    home = np.asarray(home_shots_faced_per_90, dtype=float)
    away = np.asarray(away_shots_faced_per_90, dtype=float)
    if len(home) < min_rows or len(away) < min_rows:
        return DEFAULT_AWAY_SHOT_MULTIPLIER
    home_mean = home.mean()
    if home_mean <= 0:
        return DEFAULT_AWAY_SHOT_MULTIPLIER
    return float(away.mean() / home_mean)


@dataclass(frozen=True)
class SavesProjection:
    """A goalkeeper's saves-component projection for one gameweek."""

    expected_saves: float
    expected_penalties_faced: float
    penalty_save_rate: float = DEFAULT_PENALTY_SAVE_RATE

    def __post_init__(self) -> None:
        if self.expected_saves < 0:
            raise ValueError("expected_saves must be non-negative")
        if self.expected_penalties_faced < 0:
            raise ValueError("expected_penalties_faced must be non-negative")
        if not 0.0 <= self.penalty_save_rate <= 1.0:
            raise ValueError("penalty_save_rate must be in [0, 1]")

    @property
    def expected_points(self) -> float:
        """1 point per 3 saves (using the full outcome distribution, since floor-division is
        non-linear — BUILD_PLAN scoring.py) plus the low-probability penalty-save bonus."""
        save_points = expected_floor_division(self.expected_saves, SAVES_PER_POINT)
        penalty_save_points = (
            self.expected_penalties_faced * self.penalty_save_rate * PENALTY_SAVE_POINTS
        )
        return save_points + penalty_save_points


def project_saves(
    opponent_shots_on_target_per_90: float,
    team_xga_per_90: float,
    league_avg_xga_per_90: float,
    is_home: bool,
    expected_penalties_faced: float = 0.0,
    penalty_save_rate: float = DEFAULT_PENALTY_SAVE_RATE,
    save_conversion_rate: float = DEFAULT_SAVE_CONVERSION_RATE,
    away_shot_multiplier: float = DEFAULT_AWAY_SHOT_MULTIPLIER,
    expected_minutes: float = 90.0,
) -> SavesProjection:
    """Top-level entry point combining shots faced, save conversion, and the penalty-save bonus.

    Requires a real ``opponent_shots_on_target_per_90`` — still blocked (BUILD_PLAN 3.2 / Tier
    3.2): no opponent shots-on-target data source exists yet. See
    :func:`project_saves_from_own_rate` for the fallback the real backtest driver uses today
    (ENGINE_IMPROVEMENTS_3.md D.1) — this function remains the intended model once that data
    source exists, deliberately kept rather than removed.
    """
    shots_faced = expected_shots_faced(
        opponent_shots_on_target_per_90,
        team_xga_per_90,
        league_avg_xga_per_90,
        is_home,
        away_shot_multiplier,
        expected_minutes,
    )
    return SavesProjection(
        expected_saves=expected_saves(shots_faced, save_conversion_rate),
        expected_penalties_faced=expected_penalties_faced,
        penalty_save_rate=penalty_save_rate,
    )


def project_saves_from_own_rate(
    own_save_rate_per_90: float,
    expected_minutes: float = 90.0,
    *,
    individual_weight: float | None = None,
    league_avg_save_rate_per_90: float | None = None,
    shrinkage_k: float = 0.0,
    expected_penalties_faced: float = 0.0,
    penalty_save_rate: float = DEFAULT_PENALTY_SAVE_RATE,
) -> SavesProjection:
    """Fallback saves projection using a goalkeeper's own per-90 saves EWMA rate rather than an
    opponent-shots-on-target-driven formula (ENGINE_IMPROVEMENTS_3.md D.1).

    :func:`project_saves`'s real blocker (Tier 3.2 — no opponent shots-on-target data source) only
    affects its opponent-adjustment term; a keeper's own saves history is already in the vaastav
    archive, and a real point-in-time measurement found it the better predictor of the two anyway
    (Spearman 0.138 for a keeper's own saves EWMA vs. 0.056 for team xGC-as-of). This is what lets
    goalkeepers be scored at all today rather than excluded from the pool entirely — saves are only
    ~18% of a goalkeeper's scoring, so this unblocks the other ~82% (appearance, clean sheets,
    goals-conceded, bonus) without needing new data.

    Shrinkage toward the league-average rate (same opt-in shape as
    :func:`engine.models.cards.project_cards`) only kicks in when ``individual_weight`` and
    ``league_avg_save_rate_per_90`` are given, ``shrinkage_k`` is positive, and the prior itself is
    a real, positive rate.

    A goalkeeper always makes some saves over any real sample, so a ``league_avg_save_rate_per_90``
    of exactly 0.0 is never a genuine "keepers make no saves" prior. It is the fitted constant's own
    documented too-thin-training-data fallback (see
    ``backtest.run_season._fit_league_avg_rate_by_position``'s ``min_rows`` guard), and real
    measurement against the 2025-26 season found this fallback firing for the first several
    gameweeks of any walk-forward window, before the position accumulates enough rows. Blending a
    keeper's real per-90 rate toward that 0.0 placeholder, rather than skipping shrinkage
    altogether the same as when no prior is supplied at all, silently understated every keeper's
    saves for exactly as long as the fallback stayed active, worth roughly a third of the true rate
    for a keeper with only one or two matches of individual history at that point. Treating a
    non-positive prior the same as "no prior available" avoids anchoring toward a value that was
    never a real estimate of anything.
    """
    if own_save_rate_per_90 < 0:
        raise ValueError("own_save_rate_per_90 must be non-negative")
    if expected_minutes < 0:
        raise ValueError("expected_minutes must be non-negative")
    effective_rate = own_save_rate_per_90
    if (
        individual_weight is not None
        and league_avg_save_rate_per_90 is not None
        and league_avg_save_rate_per_90 > 0
        and shrinkage_k > 0
    ):
        effective_rate = shrink_toward_prior(
            own_save_rate_per_90, individual_weight, league_avg_save_rate_per_90, shrinkage_k
        )
    return SavesProjection(
        expected_saves=effective_rate * (expected_minutes / 90.0),
        expected_penalties_faced=expected_penalties_faced,
        penalty_save_rate=penalty_save_rate,
    )
