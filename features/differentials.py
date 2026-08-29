"""Differentials: players sustainedly outperforming their own position and price bracket, among
low-owned players (DIFFERENTIALS_PLAN). Pure functions over already-loaded state, no I/O, matching
every other module in this package.

**The headline metric is shrunk, never raw (D6).** A player's points per 90 over the window is
blended toward the median for their own position and price bucket, using
:func:`engine.rates.shrink_toward_prior` weighted by :func:`engine.rates.effective_sample_minutes`
-- the same mechanism the goals, assists, saves and cards models already use for exactly this
"how much do we trust this rate" problem. This is what lets the screen show something from GW1
onward without a minimum-sample gate: a one-match performance is heavily shrunk toward the bracket
median, so it self-limits false positives instead of needing a threshold to suppress them, and it
gets more accurate every gameweek as the same estimator leans less on the prior. See
DIFFERENTIALS_PLAN D6/D6a for the full reasoning, and ``SHRINKAGE_K``/the confidence thresholds
below for how the two are tuned to agree with each other.

**Current season only (D5).** The window is resolved from ``latest_played_gameweek``, never from
cross-season history, even though ``engine.data.cross_season`` could supply it -- a summer
transfer or a lost starting place makes prior-season evidence actively misleading for this
specific feature, whose entire claim is "this is verified".

**The one hard exclusion is zero minutes in the window (D7).** Everything else -- thin samples,
missing optional fields, an unpopulated price bracket -- degrades to ``None``/low confidence
rather than dropping the player, so a caller always gets a real answer about who was actually on
the pitch at all.

Layering matches ``features/player_stats.py`` and ``features/transfers.py`` exactly: this returns
the whole matching pool unsorted (D10: no composite score, so there is no single ranking to
apply), the API layer maps it to response rows, and click-to-sort happens in the browser.

**Ownership is lens-dependent (MINI_LEAGUE_PLAN M24, supersedes DIFFERENTIALS_PLAN D1's original
global-only framing).** D1 originally gated "differential" purely on FPL-wide ownership. A player
at 4% globally who three of your league rivals happen to own is not a differential for you; a
player at 25% globally nobody in your league owns is. Both the ownership percentage/count used for
filtering and the ownership columns shown are therefore supplied by the caller as one
:class:`OwnershipLens` -- either the FPL-wide percentage (the ``"global"`` lens, this module's
original and still-supported behaviour) or a specific mini-league's captaincy-weighted effective
ownership (the ``"league"`` lens, sourced from :mod:`features.mini_league`). This module never
learns what a mini-league is; it only ever reads whichever lens it's handed, keeping the "which
lens, and the fallback chain between them" decision entirely in the API layer (``api/main.py``),
matching this module's existing "no I/O, no caller-specific knowledge" stance.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import pandas as pd

from engine.data.cold_start import _price_bucket
from engine.data.live_adapter import DEFAULT_TOTAL_MANAGERS
from engine.data.player_history import PlayerGameweekActual
from engine.rates import effective_sample_minutes, shrink_toward_prior
from engine.scoring import DEF, FWD, GK, MID

__all__ = [
    "DEFAULT_WINDOW_GAMEWEEKS",
    "PRICE_BRACKET_MIN_PLAYERS",
    "SHRINKAGE_K",
    "CONFIDENCE_MEDIUM_THRESHOLD",
    "CONFIDENCE_HIGH_THRESHOLD",
    "PROVEN_SURPLUS_THRESHOLD",
    "LUCK_GAP_THRESHOLD",
    "EMERGING_XGI_THRESHOLD",
    "EMERGING_MIN_STARTS",
    "Confidence",
    "Archetype",
    "DifferentialWindow",
    "PriceBracketBaseline",
    "PriceBracketBaselines",
    "OwnershipLens",
    "PlayerDifferential",
    "resolve_window",
    "fit_price_bracket_baselines",
    "compute_player_differential",
    "build_differentials",
]

# D5: default window length. Trivially overridable per call; 6 gameweeks is DIFFERENTIALS_PLAN's
# proposed default, open for the user to move.
DEFAULT_WINDOW_GAMEWEEKS = 6

# A price bucket with fewer real players than this in the window is too thin to trust as a peer
# median (two players is not "the going rate for this bracket") -- falls back to the
# position-only median, the same bucket-then-position fallback chain
# engine.data.cold_start.baseline_projection already uses.
PRICE_BRACKET_MIN_PLAYERS = 3

# --- Shrinkage and confidence, tuned to agree with each other -----------------------------------
#
# Both are expressed in effective-sample-minutes units (engine.rates.effective_sample_minutes),
# which for a run of ~90-minute matches under that module's default EWMA config is close to a
# plain sum over a short recent window (little decay accumulates inside 6 games) -- appropriate
# here, since a differentials window wants roughly-equal recent weighting, not the long-history
# decay behaviour that config is tuned for over a full season.
#
# SHRINKAGE_K=150 means: at 90 effective minutes (one full match), individual:prior weight is
# 90:150, i.e. the player's own rate gets ~37.5% of the blend, appropriately dominated by the
# bracket prior after a single game (D6's "one match wonder" case). At 450 effective minutes
# (~5 full matches, the HIGH confidence threshold below), the ratio is 450:150 = 3:1, ~75% own
# rate -- confidence and shrinkage cross into "mostly trusting this player's own numbers" together,
# by construction, rather than as two independently-tuned numbers that happen to agree.
SHRINKAGE_K = 150.0
CONFIDENCE_MEDIUM_THRESHOLD = 180.0  # ~2 full matches
CONFIDENCE_HIGH_THRESHOLD = 450.0  # ~5 full matches

# Points per 90, after shrinkage, above the bracket median -- the bar for "genuinely outperforming
# what this price should deliver" rather than noise. Points units, so comparable across positions.
PROVEN_SURPLUS_THRESHOLD = 1.0

# Actual (goals + assists) per 90 minus xGI per 90, over the window's raw (unshrunk) rate -- D3's
# "meaning 1" classifier, never a ranking input. Above this margin, returns are running hotter than
# the underlying process supports and are flagged as unlikely to sustain, not rewarded for it.
LUCK_GAP_THRESHOLD = 0.15

# Minimum xGI/90 to count as "the underlying numbers already say this player should be scoring",
# for the Emerging archetype -- position-dependent because a credible attacking rate for a
# forward and a defender are not the same number. Goalkeepers are excluded from this archetype
# entirely: xGI is not a meaningful signal for a position whose differential value lives elsewhere
# (saves, defensive contribution), and forcing a GK through an attacking-involvement classifier
# would misclassify rather than simply not apply.
EMERGING_XGI_THRESHOLD = {DEF: 0.25, MID: 0.35, FWD: 0.45}

# Starts required in-window before "the underlying numbers look strong" is allowed to read as
# Emerging rather than as one unrepresentative cameo.
EMERGING_MIN_STARTS = 2


class Confidence(Enum):
    """How much evidence sits behind a player's shrunk rate (D6a) -- derived from the same
    effective-sample-minutes value that drove the shrinkage itself, never computed separately, so
    the displayed confidence and the actual blend can never disagree with each other."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Archetype(Enum):
    """D3's three transparent rules. NONE is not a failure state -- most rows, especially early in
    a season, will not clear either the Proven or Emerging bar and are simply shown unclassified,
    per D7's "no hard exclusion beyond zero minutes"."""

    PROVEN = "proven"
    EMERGING = "emerging"
    RIDING_LUCK = "riding_luck"
    NONE = "none"


@dataclass(frozen=True)
class DifferentialWindow:
    """The gameweek range actually used, distinct from what was requested -- D5/D6's clamp to
    however many gameweeks have actually been played this season. ``gameweek_to < gameweek_from``
    is the legitimate preseason case: nothing has been played yet."""

    gameweek_from: int
    gameweek_to: int
    requested_gameweeks: int


@dataclass(frozen=True)
class PriceBracketBaseline:
    """The peer group's own observed rate for one position/price cut -- never shrunk itself, since
    it is the prior that other players get shrunk toward, not a player being evaluated."""

    median_points_per_90: float
    n_players: int


@dataclass(frozen=True)
class PriceBracketBaselines:
    """D2's bucket-then-position fallback chain, mirroring
    :class:`engine.data.cold_start.ColdStartPriors`'s own two-level structure."""

    by_bucket: Mapping[tuple[str, int], PriceBracketBaseline]
    by_position: Mapping[str, PriceBracketBaseline]

    def median_for(self, position: str, price: int) -> PriceBracketBaseline | None:
        bucket = _price_bucket(price)
        baseline = self.by_bucket.get((position, bucket))
        if baseline is not None:
            return baseline
        return self.by_position.get(position)


GLOBAL_LENS = "global"
LEAGUE_LENS = "league"


@dataclass(frozen=True)
class OwnershipLens:
    """Per-player ownership under one lens (MINI_LEAGUE_PLAN M24/M26) -- either
    :data:`GLOBAL_LENS` (FPL-wide ``selected_by_percent``) or :data:`LEAGUE_LENS` (one mini-league's
    captaincy-weighted effective ownership, built from :mod:`features.mini_league`). A player
    missing from any of these mappings is simply not owned/rated under this lens, not an error --
    every lookup degrades to ``None``/``0``/``()``.

    ``owner_count``/``eo_multiplier``/``owner_names`` are only ever populated under the league
    lens: the global lens has no notion of "how many rivals" or "which of them by name", only a
    single FPL-wide percentage.
    """

    source: str  # GLOBAL_LENS | LEAGUE_LENS
    n_rivals: int | None  # number of rivals the league lens was computed over; None under global
    percent: Mapping[int, float | None]
    owner_count: Mapping[int, int]
    eo_multiplier: Mapping[int, float]
    owner_names: Mapping[int, tuple[str, ...]]

    def __post_init__(self) -> None:
        if self.source not in (GLOBAL_LENS, LEAGUE_LENS):
            raise ValueError(f"unknown ownership lens source: {self.source!r}")


@dataclass(frozen=True)
class PlayerDifferential:
    """One player's window, fully computed. Every field here is a real, independently sortable
    column (D10: no composite score) -- the UI ranks on whichever the manager clicks, most often
    ``surplus_vs_bracket``."""

    player_id: int
    position: str
    price: int
    minutes: int
    apps_in_window: int
    starts_in_window: int | None

    points_per_90: float
    shrunk_points_per_90: float
    bracket_median_points_per_90: float
    surplus_vs_bracket: float
    effective_sample_minutes: float
    confidence: Confidence

    xgi_per_90: float
    goals_assists_per_90: float
    luck_gap: float
    defensive_contribution_per_90: float
    bps_per_90: float | None

    return_frequency: float
    points_variance: float | None
    recent_vs_earlier_points_per_90: float | None
    minutes_trend: float | None

    # Ownership under whichever OwnershipLens the caller supplied (M24/M26) -- the FPL-wide
    # percentage under the global lens, or the league's captaincy-weighted EO percentage under the
    # league lens. ``ownership_trend_pct_per_gw``/``net_transfers_per_gw`` are the two exceptions
    # (M27): both are always FPL-wide market-momentum signals, sourced from PlayerGameweekActual's
    # own global ``selected``/``transfers_in``/``transfers_out`` counters, regardless of lens --
    # they measure something a single mini-league is too small to say anything useful about.
    current_ownership_percent: float | None
    ownership_trend_pct_per_gw: float | None
    net_transfers_per_gw: float | None

    archetype: Archetype

    # League-lens-only columns (M28) -- ``None``/``()`` under the global lens, or for a player no
    # rival owns under the league lens.
    league_owner_count: int | None = None
    league_eo_multiplier: float | None = None
    league_owner_names: tuple[str, ...] = ()


def resolve_window(latest_played_gameweek: int, requested_gameweeks: int) -> DifferentialWindow:
    """D5/D6: the last ``requested_gameweeks`` completed gameweeks, clamped to what has actually
    been played. ``latest_played_gameweek <= 0`` (nothing played yet this season) resolves to an
    empty window (``gameweek_to < gameweek_from``) rather than raising -- a real, if unhelpful,
    answer for preseason, matching :func:`build_differentials`'s "no rows, no crash" handling of
    it.
    """
    if requested_gameweeks <= 0:
        raise ValueError("requested_gameweeks must be positive")
    if latest_played_gameweek <= 0:
        return DifferentialWindow(
            gameweek_from=1, gameweek_to=0, requested_gameweeks=requested_gameweeks
        )
    gameweek_from = max(1, latest_played_gameweek - requested_gameweeks + 1)
    return DifferentialWindow(
        gameweek_from=gameweek_from,
        gameweek_to=latest_played_gameweek,
        requested_gameweeks=requested_gameweeks,
    )


def _records_in_window(
    records: Sequence[PlayerGameweekActual], window: DifferentialWindow
) -> list[PlayerGameweekActual]:
    return sorted(
        (r for r in records if window.gameweek_from <= r.gameweek <= window.gameweek_to),
        key=lambda r: r.gameweek,
    )


def _per_90(total: float, minutes: int) -> float:
    return total * 90.0 / minutes


def fit_price_bracket_baselines(
    windowed_records_by_player: Mapping[int, Sequence[PlayerGameweekActual]],
    position_by_player: Mapping[int, str],
    price_by_player: Mapping[int, int],
) -> PriceBracketBaselines:
    """D2: median points per 90 for every (position, price bucket) with real minutes in the
    window, reusing :func:`engine.data.cold_start._price_bucket` rather than a second bucketing
    scheme. Every player with any minutes in the window contributes to the peer pool it is itself
    compared against -- the median is always computed from real, current-window evidence, never
    from a separately-fitted prior season.
    """
    rates_by_bucket: dict[tuple[str, int], list[float]] = defaultdict(list)
    rates_by_position: dict[str, list[float]] = defaultdict(list)

    for player_id, records in windowed_records_by_player.items():
        position = position_by_player.get(player_id)
        price = price_by_player.get(player_id)
        if position is None or price is None:
            continue
        minutes = sum(r.minutes for r in records)
        if minutes <= 0:
            continue
        points_per_90 = _per_90(sum(r.total_points for r in records), minutes)
        rates_by_bucket[(position, _price_bucket(price))].append(points_per_90)
        rates_by_position[position].append(points_per_90)

    by_bucket = {
        key: PriceBracketBaseline(
            median_points_per_90=statistics.median(values), n_players=len(values)
        )
        for key, values in rates_by_bucket.items()
        if len(values) >= PRICE_BRACKET_MIN_PLAYERS
    }
    by_position = {
        position: PriceBracketBaseline(
            median_points_per_90=statistics.median(values), n_players=len(values)
        )
        for position, values in rates_by_position.items()
        if values
    }
    return PriceBracketBaselines(by_bucket=by_bucket, by_position=by_position)


def _confidence_for(weight: float) -> Confidence:
    if weight >= CONFIDENCE_HIGH_THRESHOLD:
        return Confidence.HIGH
    if weight >= CONFIDENCE_MEDIUM_THRESHOLD:
        return Confidence.MEDIUM
    return Confidence.LOW


def _has_return(actual: PlayerGameweekActual, position: str) -> bool:
    attacking = (actual.goals_scored + actual.assists) > 0
    defensive = position in (GK, DEF) and actual.clean_sheets > 0
    return attacking or defensive


def _classify_archetype(
    position: str,
    surplus_vs_bracket: float,
    luck_gap: float,
    xgi_per_90: float,
    starts_in_window: int | None,
) -> Archetype:
    supported = luck_gap <= LUCK_GAP_THRESHOLD
    if surplus_vs_bracket >= PROVEN_SURPLUS_THRESHOLD:
        return Archetype.PROVEN if supported else Archetype.RIDING_LUCK
    if (
        position != GK
        and supported
        and xgi_per_90 >= EMERGING_XGI_THRESHOLD[position]
        and starts_in_window is not None
        and starts_in_window >= EMERGING_MIN_STARTS
    ):
        return Archetype.EMERGING
    return Archetype.NONE


def _optional_sum(records: Sequence[PlayerGameweekActual], attr: str) -> int | None:
    values = [getattr(r, attr) for r in records]
    return None if any(v is None for v in values) else sum(values)


def _recent_vs_earlier_points_per_90(
    earlier: Sequence[PlayerGameweekActual], recent: Sequence[PlayerGameweekActual]
) -> float | None:
    earlier_minutes = sum(r.minutes for r in earlier)
    recent_minutes = sum(r.minutes for r in recent)
    if earlier_minutes <= 0 or recent_minutes <= 0:
        return None
    earlier_rate = _per_90(sum(r.total_points for r in earlier), earlier_minutes)
    recent_rate = _per_90(sum(r.total_points for r in recent), recent_minutes)
    return recent_rate - earlier_rate


def _minutes_trend(
    earlier: Sequence[PlayerGameweekActual], recent: Sequence[PlayerGameweekActual]
) -> float | None:
    if not earlier or not recent:
        return None
    earlier_avg = sum(r.minutes for r in earlier) / len(earlier)
    recent_avg = sum(r.minutes for r in recent) / len(recent)
    return recent_avg - earlier_avg


def compute_player_differential(
    player_id: int,
    position: str,
    price: int,
    records: Sequence[PlayerGameweekActual],
    baselines: PriceBracketBaselines,
    current_ownership_percent: float | None = None,
    shrinkage_k: float = SHRINKAGE_K,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
    league_owner_count: int | None = None,
    league_eo_multiplier: float | None = None,
    league_owner_names: tuple[str, ...] = (),
) -> PlayerDifferential | None:
    """One player's window, already filtered to the resolved :class:`DifferentialWindow` and
    sorted chronologically (:func:`_records_in_window`'s contract). Returns ``None`` for zero
    minutes (D7's only hard exclusion) or when no peer baseline exists at all for this position
    (only possible if literally nobody in the position played this window -- effectively the same
    preseason case :func:`resolve_window` already names, not a second designed gate).

    ``current_ownership_percent`` is whichever :class:`OwnershipLens` the caller is using;
    ``league_owner_count``/``league_eo_multiplier``/``league_owner_names`` (M28) are additionally
    populated only when that lens is the league one -- left at their defaults otherwise, matching
    every other "not applicable under this lens" field on :class:`PlayerDifferential`.
    """
    minutes = sum(r.minutes for r in records)
    if minutes <= 0:
        return None
    baseline = baselines.median_for(position, price)
    if baseline is None:
        return None

    points_per_90 = _per_90(sum(r.total_points for r in records), minutes)
    weight = effective_sample_minutes(
        pd.DataFrame({"minutes": [r.minutes for r in records]}), minutes_col="minutes"
    )
    shrunk_points_per_90 = shrink_toward_prior(
        points_per_90, weight, baseline.median_points_per_90, shrinkage_k
    )
    surplus_vs_bracket = shrunk_points_per_90 - baseline.median_points_per_90

    goals_assists_per_90 = _per_90(sum(r.goals_scored + r.assists for r in records), minutes)
    xgi_per_90 = _per_90(sum(r.expected_goal_involvements for r in records), minutes)
    luck_gap = goals_assists_per_90 - xgi_per_90
    dc_per_90 = _per_90(sum(r.defensive_contribution for r in records), minutes)

    bps_total = _optional_sum(records, "bps")
    bps_per_90 = _per_90(bps_total, minutes) if bps_total is not None else None
    starts_in_window = _optional_sum(records, "starts")

    played = [r for r in records if r.minutes > 0]
    return_frequency = sum(1 for r in played if _has_return(r, position)) / len(played)
    points_variance = (
        statistics.pvariance(r.total_points for r in played) if len(played) >= 2 else None
    )

    mid = len(records) // 2
    earlier, recent = records[:mid], records[mid:]
    recent_vs_earlier = _recent_vs_earlier_points_per_90(earlier, recent)
    minutes_trend = _minutes_trend(earlier, recent)

    ownership_values = [r.selected for r in records]
    ownership_trend_pct_per_gw = None
    if len(records) >= 2 and all(v is not None for v in ownership_values):
        span = records[-1].gameweek - records[0].gameweek
        if span > 0:
            first_pct = ownership_values[0] / total_managers * 100.0
            last_pct = ownership_values[-1] / total_managers * 100.0
            ownership_trend_pct_per_gw = (last_pct - first_pct) / span

    transfers_in_total = _optional_sum(records, "transfers_in")
    transfers_out_total = _optional_sum(records, "transfers_out")
    net_transfers_per_gw = (
        (transfers_in_total - transfers_out_total) / len(records)
        if transfers_in_total is not None and transfers_out_total is not None
        else None
    )

    archetype = _classify_archetype(
        position, surplus_vs_bracket, luck_gap, xgi_per_90, starts_in_window
    )

    return PlayerDifferential(
        player_id=player_id,
        position=position,
        price=price,
        minutes=minutes,
        apps_in_window=len(played),
        starts_in_window=starts_in_window,
        points_per_90=points_per_90,
        shrunk_points_per_90=shrunk_points_per_90,
        bracket_median_points_per_90=baseline.median_points_per_90,
        surplus_vs_bracket=surplus_vs_bracket,
        effective_sample_minutes=weight,
        confidence=_confidence_for(weight),
        xgi_per_90=xgi_per_90,
        goals_assists_per_90=goals_assists_per_90,
        luck_gap=luck_gap,
        defensive_contribution_per_90=dc_per_90,
        bps_per_90=bps_per_90,
        return_frequency=return_frequency,
        points_variance=points_variance,
        recent_vs_earlier_points_per_90=recent_vs_earlier,
        minutes_trend=minutes_trend,
        current_ownership_percent=current_ownership_percent,
        ownership_trend_pct_per_gw=ownership_trend_pct_per_gw,
        net_transfers_per_gw=net_transfers_per_gw,
        archetype=archetype,
        league_owner_count=league_owner_count,
        league_eo_multiplier=league_eo_multiplier,
        league_owner_names=league_owner_names,
    )


_EMPTY_GLOBAL_LENS = OwnershipLens(
    source=GLOBAL_LENS, n_rivals=None, percent={}, owner_count={}, eo_multiplier={}, owner_names={}
)


def build_differentials(
    player_history: Mapping[int, Sequence[PlayerGameweekActual]],
    position_by_player: Mapping[int, str],
    price_by_player: Mapping[int, int],
    latest_played_gameweek: int,
    window_gameweeks: int = DEFAULT_WINDOW_GAMEWEEKS,
    ownership: OwnershipLens | None = None,
    max_ownership_percent: float | None = None,
    max_league_owners: int | None = None,
    total_managers: float = DEFAULT_TOTAL_MANAGERS,
) -> tuple[DifferentialWindow, list[PlayerDifferential]]:
    """Every player with real minutes in the resolved window (D7), unsorted -- D10's "no composite
    score", so there is no single ranking to apply here; the caller sorts on whichever column it
    wants.

    ``ownership`` (M24/M26) supplies both the filter and the display columns; it defaults to an
    empty global lens (no ownership data at all) when omitted, matching this function's original
    behaviour before the league lens existed. Which filter parameter actually applies follows
    ``ownership.source`` (M25): ``max_ownership_percent`` under the global lens (a player with
    unknown percentage is kept rather than dropped, since there is no figure to disprove they
    qualify), ``max_league_owners`` under the league lens (every player has a definite owner count,
    0 if no rival owns them, so there is no "unknown" case to preserve there). Either filter is
    domain logic applied here, per D1, not a display concern left to the API layer.
    """
    ownership = ownership if ownership is not None else _EMPTY_GLOBAL_LENS
    window = resolve_window(latest_played_gameweek, window_gameweeks)
    if window.gameweek_to < window.gameweek_from:
        return window, []

    windowed_records_by_player = {
        player_id: _records_in_window(records, window)
        for player_id, records in player_history.items()
    }
    baselines = fit_price_bracket_baselines(
        windowed_records_by_player, position_by_player, price_by_player
    )

    results: list[PlayerDifferential] = []
    for player_id, records in windowed_records_by_player.items():
        position = position_by_player.get(player_id)
        price = price_by_player.get(player_id)
        if position is None or price is None:
            continue

        percent = ownership.percent.get(player_id)
        if ownership.source == GLOBAL_LENS:
            if max_ownership_percent is not None and percent is not None:
                if percent > max_ownership_percent:
                    continue
        else:
            owner_count = ownership.owner_count.get(player_id, 0)
            if max_league_owners is not None and owner_count > max_league_owners:
                continue

        is_league_lens = ownership.source == LEAGUE_LENS
        differential = compute_player_differential(
            player_id,
            position,
            price,
            records,
            baselines,
            current_ownership_percent=percent,
            total_managers=total_managers,
            league_owner_count=ownership.owner_count.get(player_id, 0) if is_league_lens else None,
            league_eo_multiplier=(
                ownership.eo_multiplier.get(player_id, 0.0) if is_league_lens else None
            ),
            league_owner_names=ownership.owner_names.get(player_id, ()) if is_league_lens else (),
        )
        if differential is not None:
            results.append(differential)
    return window, results
