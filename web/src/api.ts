// Typed fetch client, one function per api/main.py endpoint. No FPL rule logic lives here --
// every mutation just calls the API and returns the recomposed SquadOut, matching the backend's
// own "the server is always the source of truth" design.

import { API_BASE_URL } from "./config";

export interface RuleViolation {
  code: string;
  message: string;
  player_ids: number[];
}

export class ApiError extends Error {
  violation: RuleViolation;
  status: number;
  constructor(status: number, violation: RuleViolation) {
    super(violation.message);
    this.status = status;
    this.violation = violation;
  }
}

export interface TeamOut {
  team_id: number;
  name: string;
  short_name: string;
}

export interface GameweekOut {
  season: string;
  gameweek: number;
  deadline_time: string;
  deadline_passed: boolean;
  generated_at: string;
  model_version: string;
  horizon_gameweeks: number[];
}

export interface SquadPlayerOut {
  player_id: number;
  position: string;
  price: number;
}

// The one live sandbox squad -- 0 to 15 players, no confirm step, no transfer economy.
export interface SquadOut {
  squad: SquadPlayerOut[];
  starting_xi: number[];
  bench_order: number[];
  captain_id: number | null;
  vice_captain_id: number | null;
  is_complete: boolean;
  budget_ceiling: number;
  budget_remaining: number;
}

export interface SquadPointsOut {
  total: number;
  starting_xi_points: number;
  bench_points: number;
  captain_bonus: number;
  per_player: Record<string, number>;
  per_gameweek: Record<string, number>;
  missing_player_ids: number[];
}

export interface ComponentBreakdownOut {
  appearance: number;
  goals: number;
  assists: number;
  clean_sheet: number;
  goals_conceded: number;
  defensive_contribution: number;
  saves: number;
  bonus: number;
  cards: number;
  penalty_misses: number;
  own_goals: number;
  total: number;
}

export interface PlayerDetailOut {
  player_id: number;
  name: string;
  position: string;
  price: number | null;
  team_id: number | null;
  low_confidence: boolean;
  gameweek: number;
  expected_points: number;
  breakdown: ComponentBreakdownOut;
  floor: number | null;
  ceiling: number | null;
  prob_big_haul: number | null;
}

export interface FixtureCellOut {
  gameweek: number;
  opponent_id: number | null;
  is_home: boolean | null;
  expected_points: number | null;
}

export interface PlayerPanelRowOut {
  player_id: number;
  name: string;
  team_id: number | null;
  position: string;
  price: number | null;
  low_confidence: boolean;
  fixtures: FixtureCellOut[];
}

export interface FixtureTickerCellFixtureOut {
  opponent_id: number;
  is_home: boolean;
  difficulty: number;
}

export interface FixtureTickerCellOut {
  gameweek: number;
  fixtures: FixtureTickerCellFixtureOut[];
}

export interface FixtureTickerRowOut {
  team_id: number;
  gameweeks: FixtureTickerCellOut[];
  average_difficulty: number | null;
}

export interface HorizonDifficultyOut {
  attack_rating: number;
  defense_rating: number;
  mean_attack_factor: number;
  mean_defense_factor: number;
}

export interface TeamSwingRowOut {
  team_id: number;
  near: HorizonDifficultyOut | null;
  far: HorizonDifficultyOut | null;
  attack_swing: number | null;
  defense_swing: number | null;
  has_owned_player: boolean;
}

export interface FixtureSwingResponseOut {
  near_gameweeks: number[];
  far_gameweeks: number[];
  rows: TeamSwingRowOut[];
}

// --- Player Stats page ---------------------------------------------------------------------

export interface ActualStatsOut {
  gameweek_from: number;
  gameweek_to: number;
  apps: number;
  minutes: number;
  goals_scored: number;
  assists: number;
  clean_sheets: number;
  goals_conceded: number;
  own_goals: number;
  penalties_missed: number;
  penalties_saved: number;
  saves: number;
  bonus: number;
  yellow_cards: number;
  red_cards: number;
  total_points: number;
  expected_goals: number;
  expected_assists: number;
  expected_goal_involvements: number;
  expected_goals_conceded: number;
  points_breakdown: ComponentBreakdownOut;
  selected_by_percent: number | null;
  small_sample: boolean;
}

export interface PlayerStatsRowOut {
  player_id: number;
  name: string;
  team_id: number | null;
  position: string;
  price: number | null;
  low_confidence: boolean;
  actuals: ActualStatsOut;
  fixtures: FixtureCellOut[];
}

// --- Differentials page ---------------------------------------------------------------------

export interface DifferentialsWindowOut {
  gameweek_from: number;
  gameweek_to: number;
  requested_gameweeks: number;
}

export type Confidence = "low" | "medium" | "high";
export type Archetype = "proven" | "emerging" | "riding_luck" | "none";

export interface DifferentialRowOut {
  player_id: number;
  name: string;
  team_id: number | null;
  position: string;
  price: number;
  minutes: number;
  apps_in_window: number;
  starts_in_window: number | null;
  points_per_90: number;
  shrunk_points_per_90: number;
  bracket_median_points_per_90: number;
  surplus_vs_bracket: number;
  confidence: Confidence;
  xgi_per_90: number;
  goals_assists_per_90: number;
  luck_gap: number;
  defensive_contribution_per_90: number;
  bps_per_90: number | null;
  return_frequency: number;
  points_variance: number | null;
  recent_vs_earlier_points_per_90: number | null;
  minutes_trend: number | null;
  current_ownership_percent: number | null;
  ownership_trend_pct_per_gw: number | null;
  net_transfers_per_gw: number | null;
  archetype: Archetype;
  fixtures: FixtureCellOut[];
}

export interface DifferentialsResponseOut {
  window: DifferentialsWindowOut;
  rows: DifferentialRowOut[];
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!response.ok) {
    const body = (await response.json()) as RuleViolation;
    throw new ApiError(response.status, body);
  }
  return (await response.json()) as T;
}

function query(params: Record<string, string | number | boolean | undefined | null>): string {
  const parts = Object.entries(params)
    .filter(([, value]) => value !== undefined && value !== null)
    .map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(String(value))}`);
  return parts.length ? `?${parts.join("&")}` : "";
}

export const api = {
  getGameweek: () => request<GameweekOut>("/gameweek"),
  getTeams: () => request<TeamOut[]>("/teams"),
  getSquad: () => request<SquadOut>("/squad"),

  addPlayer: (player_id: number, position: string, price: number) =>
    request<SquadOut>("/squad/players", {
      method: "POST",
      body: JSON.stringify({ player_id, position, price }),
    }),
  removePlayer: (playerId: number) =>
    request<SquadOut>(`/squad/players/${playerId}`, { method: "DELETE" }),
  clearSquad: () => request<SquadOut>("/squad/players", { method: "DELETE" }),

  setCaptain: (player_id: number, role: "captain" | "vice") =>
    request<SquadOut>("/squad/captain", {
      method: "POST",
      body: JSON.stringify({ player_id, role }),
    }),
  setBenchOrder: (starting_xi: number[], bench_order: number[]) =>
    request<SquadOut>("/squad/bench-order", {
      method: "POST",
      body: JSON.stringify({ starting_xi, bench_order }),
    }),
  substitute: (out_id: number, in_id: number) =>
    request<SquadOut>("/squad/substitute", {
      method: "POST",
      body: JSON.stringify({ out_id, in_id }),
    }),
  optimiseXi: () => request<SquadOut>("/squad/optimise-xi", { method: "POST" }),
  optimise: (objective: "starting_xi" | "full_squad" = "starting_xi", captainMultiplier = 2.0) =>
    request<SquadOut>("/squad/optimise", {
      method: "POST",
      body: JSON.stringify({ objective, captain_multiplier: captainMultiplier }),
    }),
  importSquad: (teamId: number) =>
    request<SquadOut>("/squad/import", {
      method: "POST",
      body: JSON.stringify({ team_id: teamId }),
    }),

  getSquadPoints: (chip?: string | null, horizon: number = 1, gameweek?: number) =>
    request<SquadPointsOut>(
      `/squad/points${query({ chip: chip ?? undefined, horizon: gameweek === undefined ? horizon : undefined, gameweek })}`,
    ),

  listPlayers: (filters: {
    position?: string;
    min_price?: number;
    max_price?: number;
    search?: string;
  }) => request<PlayerPanelRowOut[]>(`/players${query(filters)}`),
  getPlayer: (playerId: number, gameweek?: number) =>
    request<PlayerDetailOut>(`/players/${playerId}${query({ gameweek })}`),

  getFixtureTicker: (gameweekFrom?: number, gameweekTo?: number) =>
    request<FixtureTickerRowOut[]>(
      `/fixtures${query({ gameweek_from: gameweekFrom, gameweek_to: gameweekTo })}`,
    ),
  getFixtureSwing: (
    nearFrom?: number,
    nearTo?: number,
    farFrom?: number,
    farTo?: number,
  ) =>
    request<FixtureSwingResponseOut>(
      `/teams/fixture-swing${query({
        near_from: nearFrom,
        near_to: nearTo,
        far_from: farFrom,
        far_to: farTo,
      })}`,
    ),
  getPlayerStats: (gameweekFrom: number, gameweekTo: number) =>
    request<PlayerStatsRowOut[]>(
      `/players/stats${query({ gameweek_from: gameweekFrom, gameweek_to: gameweekTo })}`,
    ),
  getDifferentials: (window: number, maxOwnership?: number, hideOwned: boolean = true) =>
    request<DifferentialsResponseOut>(
      `/players/differentials${query({
        window,
        max_ownership: maxOwnership,
        hide_owned: hideOwned,
      })}`,
    ),
};
