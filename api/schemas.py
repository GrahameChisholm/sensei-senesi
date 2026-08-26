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
    purchase_price: int
    current_price: int
    sell_price: int


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


class TeamStateOut(BaseModel):
    squad: list[SquadPlayerOut]
    starting_xi: list[int]
    bench_order: list[int]
    captain_id: int
    vice_captain_id: int
    bank: int
    free_transfers: int
    chips_remaining: list[str]


class DraftOut(BaseModel):
    base_gameweek: int
    working_state: TeamStateOut
    transfers_made: int
    chip: str | None


class SquadOut(BaseModel):
    is_complete: bool
    committed: TeamStateOut | None
    build_picks: list[SquadPlayerOut] | None
    active_chip: str | None
    active_chip_gameweek: int | None
    chips_available: list[str]
    draft: DraftOut | None
    last_hit_cost: int | None = None


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


class DifferentialsResponseOut(BaseModel):
    window: DifferentialsWindowOut
    rows: list[DifferentialRowOut]


class TransferRecommendationOut(BaseModel):
    sell_player_id: int
    sell_player_name: str
    buy_player_id: int
    buy_player_name: str
    buy_price: int
    position: str
    net_points_gain: float
    hit_cost: int
    is_forced: bool
    reasoning: str


# --- request bodies ------------------------------------------------------------------------


class AddPlayerIn(BaseModel):
    player_id: int
    position: str
    price: int


class SubstituteIn(BaseModel):
    out_id: int
    in_id: int


class TransferIn(BaseModel):
    out_id: int
    in_id: int
    in_price: int
    in_position: str


class CaptainIn(BaseModel):
    player_id: int
    role: str  # "captain" | "vice"


class BenchOrderIn(BaseModel):
    bench_order: list[int]


class ChipIn(BaseModel):
    chip: str | None = None


class ConfirmSquadIn(BaseModel):
    player_ids: list[int]
    starting_xi: list[int]
    bench_order: list[int]
    captain_id: int
    vice_captain_id: int


class ImportSquadIn(BaseModel):
    team_id: int = Field(gt=0)
