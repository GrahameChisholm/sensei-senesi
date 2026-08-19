"""Pydantic response/request schemas for the team-selection page's API (Phase 4) — thin,
serialisation-only shapes over the real ``features/`` dataclasses. No FPL rule logic lives here.
"""

from __future__ import annotations

from pydantic import BaseModel


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


# --- Season Replay --------------------------------------------------------------------------


class SeasonLogEntryOut(BaseModel):
    gameweek: int
    points: float
    running_total: float
    chip_played: str | None


class AdvanceResultOut(BaseModel):
    """``POST /squad/advance``'s response: this gameweek's real, already-decided outcome, plus the
    refreshed squad (now on the next gameweek, unless the season just ended)."""

    gameweek: int
    chip_played: str | None
    effective_xi: list[int]
    effective_captain_id: int
    hit_cost: int
    points: float
    running_total: float
    season_complete: bool
    season_log: list[SeasonLogEntryOut]
    squad: SquadOut
