"""Pydantic response models — thin views over the ``features/`` dataclasses.

Every model uses ``from_attributes`` so a ``features/`` dataclass instance validates directly
into its matching schema with no hand-written field mapping (and nested dataclass fields, e.g.
``CaptaincyRecommendation.ranked_pool``, validate recursively the same way) -- these schemas
exist purely to give FastAPI a declared response shape (and OpenAPI docs), not to duplicate any
logic that already lives in ``features/``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CaptaincyOptionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: int
    position: str
    expected_points: float
    floor: float | None
    ceiling: float | None
    prob_big_haul: float | None
    is_owned: bool
    is_eligible: bool
    reasoning: str


class CaptaincyRecommendationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    ranked_pool: list[CaptaincyOptionOut]
    top_ev_pick: CaptaincyOptionOut | None
    safe_pick: CaptaincyOptionOut | None
    punt_pick: CaptaincyOptionOut | None


class TransferCandidateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    sell_player_id: int
    buy_player_id: int
    position: str
    sell_price: int
    buy_price: int
    net_spend: int
    horizon_points_sold: float
    horizon_points_bought: float
    points_gain: float
    hit_cost: int
    net_points_gain: float
    is_forced: bool
    reasoning: str


class TransferPlanOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    affordable_candidates: list[TransferCandidateOut]
    recommended: TransferCandidateOut | None


class ChipEvaluationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    chip: str
    target_gameweek: int
    value_now: float
    best_gameweek: int
    best_value: float
    recommendation: str
    reasoning: str


class WildcardEvaluationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    squad_uplift: float
    upgradeable_slots: int
    recommendation: str
    reasoning: str


class FixtureDifficultyOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    team_id: int
    opponent_id: int
    gameweek: int
    is_home: bool
    attack_factor: float
    defense_factor: float
    attack_rating: int
    defense_rating: int
    overall_rating: float


class SquadPlayerOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: int
    position: str
    purchase_price: int
    current_price: int
    sell_price: int


class MyTeamStateOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    squad: list[SquadPlayerOut]
    starting_xi: list[int]
    bench_order: list[int]
    captain_id: int
    vice_captain_id: int
    bank: int
    free_transfers: int
    chips_remaining: list[str]
    total_sell_value: int


class SettingsOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    fpl_team_id: int | None
    mini_league_ids: list[int]
    planning_horizon_gameweeks: int


class SettingsIn(BaseModel):
    fpl_team_id: int | None = None
    mini_league_ids: list[int] = []
    planning_horizon_gameweeks: int = 5


class DataStatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    generated_at: str | None
    is_demo_data: bool


class ModelPerformanceOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    headline: dict | None
    has_live_accuracy: bool


class ComponentBreakdownOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

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


class PlayerSummaryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: int
    name: str
    position: str
    price: int | None
    gameweek: int
    expected_points: float


class PlayerDetailOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    player_id: int
    name: str
    position: str
    price: int | None
    gameweek: int
    expected_points: float
    breakdown: ComponentBreakdownOut
    floor: float | None
    ceiling: float | None
    prob_big_haul: float | None
