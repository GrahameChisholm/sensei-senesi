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
    season: str
    gameweek: int
    deadline_time: str
    deadline_passed: bool
    generated_at: str
    model_version: str
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
    yellow_cards: int
    red_cards: int
    total_points: int
    expected_goals: float
    expected_assists: float
    expected_goal_involvements: float
    expected_goals_conceded: float
    points_breakdown: ComponentBreakdownOut
    selected_by_percent: float | None
    small_sample: bool


class PlayerStatsRowOut(BaseModel):
    player_id: int
    name: str
    team_id: int | None
    position: str
    price: int | None
    low_confidence: bool
    actuals: ActualStatsOut
    fixtures: list[FixtureCellOut]


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
    objective: str = "starting_xi"  # "starting_xi" | "full_squad" (Bench Boost active)
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
