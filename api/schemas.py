"""Pydantic response/request schemas for the team-selection page's API (Phase 4) — thin,
serialisation-only shapes over the real ``features/`` dataclasses. No FPL rule logic lives here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class RuleViolationOut(BaseModel):
    """The shared error shape for every rejected mutation — rendered directly in the UI."""

    code: str
    message: str
    player_ids: list[int] = []


class SquadPlayerOut(BaseModel):
    player_id: int
    position: str
    price: int


class ComponentBreakdownOut(BaseModel):
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
    total: float


class SimulationOut(BaseModel):
    mean: float
    median: float
    floor: float
    ceiling: float
    prob_big_haul: float


class TeamOut(BaseModel):
    team_id: int
    name: str
    short_name: str


class GameweekOut(BaseModel):
    """The gameweek the app presents as "now", which is the earliest one still open for
    decisions, not the one whose matches happen to be playing."""

    season: str
    # The decision gameweek: what a manager can still change. Advances at a deadline, not when a
    # gameweek's matches finish.
    gameweek: int
    # The gameweek the loaded projection cache was built for. Ordinarily the same as `gameweek`;
    # when it lags behind, the projections are from before the last deadline and a rebuild will
    # produce fresher numbers.
    projections_gameweek: int
    # `gameweek`'s own deadline, and whether it has passed, computed against the clock now rather
    # than served from the value frozen into the cache at build time.
    deadline_time: str
    deadline_passed: bool
    generated_at: str
    model_version: str
    # `gameweek` onward, with any already-locked gameweeks dropped.
    horizon_gameweeks: list[int]


class SquadOut(BaseModel):
    """The one live sandbox squad — 0 to 15 players, no confirm step. ``is_complete`` is true once
    all 15 slots are filled and captain/vice are set."""

    squad: list[SquadPlayerOut]
    starting_xi: list[int]
    bench_order: list[int]
    captain_id: int | None
    vice_captain_id: int | None
    is_complete: bool
    budget_ceiling: int
    budget_remaining: int


class SquadPointsOut(BaseModel):
    total: float
    starting_xi_points: float
    bench_points: float
    captain_bonus: float
    per_player: dict[str, float]
    per_gameweek: dict[str, float]
    missing_player_ids: list[int]


class PlayerDetailOut(BaseModel):
    player_id: int
    name: str
    position: str
    price: int | None
    team_id: int | None
    low_confidence: bool
    gameweek: int
    expected_points: float
    breakdown: ComponentBreakdownOut
    floor: float | None
    ceiling: float | None
    prob_big_haul: float | None


class FixtureCellOut(BaseModel):
    gameweek: int
    opponent_id: int | None
    is_home: bool | None
    expected_points: float | None


class PlayerPanelRowOut(BaseModel):
    player_id: int
    name: str
    team_id: int | None
    position: str
    price: int | None
    low_confidence: bool
    fixtures: list[FixtureCellOut]


class FixtureTickerCellFixtureOut(BaseModel):
    opponent_id: int
    is_home: bool
    difficulty: int
    expected_goals_for: float | None
    expected_goals_against: float | None


class FixtureTickerCellOut(BaseModel):
    gameweek: int
    fixtures: list[FixtureTickerCellFixtureOut]


class FixtureTickerRowOut(BaseModel):
    team_id: int
    gameweeks: list[FixtureTickerCellOut]
    average_difficulty: float | None


class HorizonDifficultyOut(BaseModel):
    attack_rating: int
    defense_rating: int
    mean_attack_factor: float
    mean_defense_factor: float


class TeamSwingRowOut(BaseModel):
    team_id: int
    near: HorizonDifficultyOut | None
    far: HorizonDifficultyOut | None
    attack_swing: float | None
    defense_swing: float | None
    has_owned_player: bool


class FixtureSwingResponseOut(BaseModel):
    near_gameweeks: list[int]
    far_gameweeks: list[int]
    rows: list[TeamSwingRowOut]


# --- Player Stats page (PLAYER_STATS_PLAN) --------------------------------------------------


class RateRatioOut(BaseModel):
    """One actual-vs-expected ratio with its credible interval. An interval containing 1.0 means
    the deviation is not yet distinguishable from chance, which is what stops a thin-sample fluke
    reading as a real signal."""

    ratio: float
    low: float
    high: float
    exposure: float


class ActualStatsOut(BaseModel):
    gameweek_from: int
    gameweek_to: int
    apps: int
    minutes: int
    goals_scored: int
    assists: int
    clean_sheets: int
    goals_conceded: int
    own_goals: int
    penalties_missed: int
    penalties_saved: int
    saves: int
    bonus: int
    defensive_contribution: int
    yellow_cards: int
    red_cards: int
    total_points: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    points_breakdown: ComponentBreakdownOut
    # Ownership under whichever lens the endpoint resolved -- see PlayerStatsResponseOut's
    # ownership_status for whether it is populated at all.
    ownership_percent: float | None
    small_sample: bool
    attacking_ratio: RateRatioOut | None = None
    defensive_ratio: RateRatioOut | None = None
    is_penalty_taker: bool = False


class AvailabilityOut(BaseModel):
    """FPL's own live availability signal, distinct from ``low_confidence`` (the engine's
    cold-start flag)."""

    status: str
    chance_of_playing_next_round: float
    news: str | None


class PlayerStatsRowOut(BaseModel):
    player_id: int
    name: str
    team_id: int | None
    position: str
    price: int | None
    low_confidence: bool
    availability: AvailabilityOut | None = None
    actuals: ActualStatsOut
    fixtures: list[FixtureCellOut]


class PlayerStatsResponseOut(BaseModel):
    """Rows plus why ownership may be missing. This page shows mini-league ownership or nothing
    at all, never FPL's population-wide figure silently substituted under the same heading, so
    the frontend is told the reason rather than left to guess."""

    ownership_status: str  # "ok" | "not_configured" | "fetch_failed" | "no_rivals"
    rows: list[PlayerStatsRowOut]


# --- Differentials page (DIFFERENTIALS_PLAN) -------------------------------------------------


class DifferentialsWindowOut(BaseModel):
    gameweek_from: int
    gameweek_to: int
    requested_gameweeks: int


class SwapCandidateOut(BaseModel):
    incoming_player_id: int
    outgoing_player_id: int
    incoming_swing: float
    outgoing_swing: float
    net_swing_delta: float
    price_delta: int


class DifferentialRowOut(BaseModel):
    player_id: int
    name: str
    team_id: int | None
    position: str
    price: int
    minutes: int
    apps_in_window: int
    starts_in_window: int | None
    points_per_90: float
    shrunk_points_per_90: float
    bracket_median_points_per_90: float
    surplus_vs_bracket: float
    confidence: str
    xgi_per_90: float
    goals_assists_per_90: float
    luck_gap: float
    defensive_contribution_per_90: float
    bps_per_90: float | None
    return_frequency: float
    points_variance: float | None
    recent_vs_earlier_points_per_90: float | None
    minutes_trend: float | None
    current_ownership_percent: float | None
    ownership_trend_pct_per_gw: float | None
    net_transfers_per_gw: float | None
    archetype: str
    fixtures: list[FixtureCellOut]
    # League ownership lens columns (MINI_LEAGUE_PLAN M28) -- None/[] under the global lens, or
    # for a player no rival owns under the league lens.
    league_owner_count: int | None = None
    league_eo_multiplier: float | None = None
    league_owner_names: list[str] = []
    # Prospective swing (M28): what this player would be worth in expected-swing terms if brought
    # into the starting XI, league lens only.
    expected_swing: float | None = None
    # Which of your own starting XI this player would most sensibly replace by expected swing
    # (league lens, complete squad, only). A swing comparison, not a budget or legality check.
    replaces: SwapCandidateOut | None = None


class DifferentialsResponseOut(BaseModel):
    window: DifferentialsWindowOut
    ownership_lens: str  # "global" | "league"
    # The gameweek the league lens's rival picks actually reflect (MINI_LEAGUE_PLAN M1/M31) --
    # None under the global lens, where the concept doesn't apply.
    picks_gameweek: int | None = None
    # Number of rivals the league lens was computed over -- None under the global lens. Lets the
    # UI render "owned by N of M rivals" instead of a bare count.
    n_rivals: int | None = None
    rows: list[DifferentialRowOut]


# --- request bodies ------------------------------------------------------------------------


class AddPlayerIn(BaseModel):
    player_id: int
    position: str
    price: int


class CaptainIn(BaseModel):
    player_id: int
    role: str  # "captain" | "vice"


class BenchOrderIn(BaseModel):
    starting_xi: list[int]
    bench_order: list[int]


class SubstituteIn(BaseModel):
    out_id: int
    in_id: int


class OptimiseIn(BaseModel):
    objective: str = "full_squad"  # "full_squad" | "starting_xi" (bench treated as worthless)
    captain_multiplier: float = 2.0  # 3.0 under Triple Captain


class ImportSquadIn(BaseModel):
    team_id: int = Field(gt=0)


class MiniLeagueSettingsOut(BaseModel):
    fpl_team_id: int | None
    mini_league_ids: list[int]


class MiniLeagueSettingsIn(BaseModel):
    fpl_team_id: int | None = None
    mini_league_ids: list[int] = []


class PlayerOwnershipOut(BaseModel):
    player_id: int
    raw_ownership_percent: float
    eo_multiplier: float
    eo_percent: float
    captain_share_percent: float
    owner_names: list[str]


class PlayerExposureOut(BaseModel):
    player_id: int
    your_multiplier: float
    ownership: PlayerOwnershipOut
    expected_points: float | None
    exposure: float
    expected_swing: float | None


class CaptainOptionOut(BaseModel):
    player_id: int
    expected_points: float | None
    captain_share_percent: float
    eo_multiplier: float
    net_captain_ev: float | None
    net_captain_std: float | None


class DifferentialPickOut(BaseModel):
    player_id: int
    your_multiplier: float
    rival_multiplier: float
    expected_points: float | None
    expected_gap_contribution: float


class HeadToHeadOut(BaseModel):
    rival_entry_id: int
    shared_count: int
    differentials: list[DifferentialPickOut]
    expected_gap: float
    gap_std: float
    p_outscore: float


class RivalChipStateOut(BaseModel):
    entry_id: int
    used_chip_names: list[str]
    remaining_chip_names: list[str]


class RivalPostureOut(BaseModel):
    rival_entry_id: int
    projected_final_gap: float
    p_finish_ahead: float
    variance_preference: str  # "increase" | "decrease" | "neutral"
    sensitivity: float


class MiniLeagueRivalOut(BaseModel):
    entry_id: int
    manager_name: str
    team_name: str
    rank: int
    total_points: int
    gameweek_points: int
    chip_state: RivalChipStateOut
    posture: RivalPostureOut
    head_to_head: HeadToHeadOut


class LeagueInsightOut(BaseModel):
    kind: str  # "edge" | "drag" | "captain"
    player_id: int
    reference_player_id: int | None
    value: float
    owner_count: int
    n_rivals: int


class MiniLeaguePanelOut(BaseModel):
    league_id: int
    league_name: str
    picks_gameweek: int
    gameweek: int
    my_rank: int
    my_total_points: int
    coverage: float
    template_xi: list[int]
    exposures: list[PlayerExposureOut]
    captain_options: list[CaptainOptionOut]
    insights: list[LeagueInsightOut]
    rivals: list[MiniLeagueRivalOut]


# --- Transfer banner (TRANSFER_BANNER) -----------------------------------------------------------


class TransferMoveOut(BaseModel):
    """One sell paired with one buy. ``in_eo_multiplier`` is the field's effective ownership of
    the incoming player, or ``None`` with no league in play, which is what tells a manager at a
    glance whether a suggested buy covers the template or differentiates from it."""

    out_player_id: int
    in_player_id: int
    out_name: str
    in_name: str
    position: str
    price_delta: int
    out_expected_points: float | None
    in_expected_points: float | None
    in_eo_multiplier: float | None


class TransferPlanOut(BaseModel):
    """``expected_final_rank_delta`` is negative when the plan improves your projected finish, so
    the UI must render it as an improvement rather than as a loss. Every league field is ``0.0``
    (or a rank of ``1.0``) when no league is configured, which the response's own
    ``league_id``/``n_rivals`` report."""

    moves: list[TransferMoveOut]
    out_player_ids: list[int]
    in_player_ids: list[int]
    n_transfers: int
    expected_points: float
    expected_points_delta: float
    expected_gap: float
    expected_gap_delta: float
    gap_std: float
    gap_std_delta: float
    expected_final_rank: float
    expected_final_rank_delta: float
    spend_delta: int
    budget_remaining: int


class TransferSuggestionOut(BaseModel):
    """``plans`` is ranked best first at the requested transfer count; ``best_by_transfer_count``
    holds one plan per transfer count from 1 upward, so the banner can show what each extra move
    buys. ``marginal_points_gains`` is that same series already differenced, index 0 being the
    first transfer's own gain."""

    plans: list[TransferPlanOut]
    best_by_transfer_count: list[TransferPlanOut]
    marginal_points_gains: list[float]
    max_transfers: int
    max_transfers_allowed: int
    current_expected_points: float
    current_expected_gap: float
    current_gap_std: float
    current_expected_final_rank: float
    variance_preference: str
    n_rivals: int
    league_id: int | None
    league_name: str
    picks_gameweek: int | None
    gameweeks: list[int]
    league_gameweek: int


class ApplyTransfersIn(BaseModel):
    out_player_ids: list[int] = Field(min_length=1)
    in_player_ids: list[int] = Field(min_length=1)
