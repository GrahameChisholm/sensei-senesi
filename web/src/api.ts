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
}

export interface SquadPlayerOut {
  player_id: number;
  position: string;
  purchase_price: number;
  current_price: number;
  sell_price: number;
}

export interface TeamStateOut {
  squad: SquadPlayerOut[];
  starting_xi: number[];
  bench_order: number[];
  captain_id: number;
  vice_captain_id: number;
  bank: number;
  free_transfers: number;
  chips_remaining: string[];
}

export interface DraftOut {
  base_gameweek: number;
  working_state: TeamStateOut;
  transfers_made: number;
  chip: string | null;
}

export interface SquadOut {
  is_complete: boolean;
  committed: TeamStateOut | null;
  build_picks: SquadPlayerOut[] | null;
  active_chip: string | null;
  active_chip_gameweek: number | null;
  chips_available: string[];
  draft: DraftOut | null;
  last_hit_cost: number | null;
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

export interface TransferRecommendationOut {
  sell_player_id: number;
  sell_player_name: string;
  buy_player_id: number;
  buy_player_name: string;
  buy_price: number;
  position: string;
  net_points_gain: number;
  hit_cost: number;
  is_forced: boolean;
  reasoning: string;
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

  addBuildPlayer: (player_id: number, position: string, price: number) =>
    request<SquadOut>("/squad/build/players", {
      method: "POST",
      body: JSON.stringify({ player_id, position, price }),
    }),
  removeBuildPlayer: (player_id: number) =>
    request<SquadOut>(`/squad/build/players/${player_id}`, { method: "DELETE" }),
  confirmBuild: (body: {
    player_ids: number[];
    starting_xi: number[];
    bench_order: number[];
    captain_id: number;
    vice_captain_id: number;
  }) => request<SquadOut>("/squad/build/confirm", { method: "POST", body: JSON.stringify(body) }),

  openDraft: () => request<SquadOut>("/squad/draft", { method: "POST" }),
  discardDraft: () => request<SquadOut>("/squad/draft", { method: "DELETE" }),
  substitute: (out_id: number, in_id: number) =>
    request<SquadOut>("/squad/draft/substitute", {
      method: "POST",
      body: JSON.stringify({ out_id, in_id }),
    }),
  transfer: (out_id: number, in_id: number, in_price: number, in_position: string) =>
    request<SquadOut>("/squad/draft/transfer", {
      method: "POST",
      body: JSON.stringify({ out_id, in_id, in_price, in_position }),
    }),
  liveTransfer: (out_id: number, in_id: number, in_price: number, in_position: string) =>
    request<SquadOut>("/squad/live-transfer", {
      method: "POST",
      body: JSON.stringify({ out_id, in_id, in_price, in_position }),
    }),
  setCaptain: (player_id: number, role: "captain" | "vice") =>
    request<SquadOut>("/squad/draft/captain", {
      method: "POST",
      body: JSON.stringify({ player_id, role }),
    }),
  liveCaptain: (player_id: number, role: "captain" | "vice") =>
    request<SquadOut>("/squad/live-captain", {
      method: "POST",
      body: JSON.stringify({ player_id, role }),
    }),
  setBenchOrder: (bench_order: number[]) =>
    request<SquadOut>("/squad/draft/bench-order", {
      method: "POST",
      body: JSON.stringify({ bench_order }),
    }),
  setDraftChip: (chip: string | null) =>
    request<SquadOut>("/squad/draft/chip", { method: "POST", body: JSON.stringify({ chip }) }),
  confirmDraft: () => request<SquadOut>("/squad/draft/confirm", { method: "POST" }),
  optimiseXi: () => request<SquadOut>("/squad/optimise-xi", { method: "POST" }),

  getSquadPoints: (chip?: string | null, horizon: number = 1, source: "draft" | "committed" = "draft") =>
    request<SquadPointsOut>(`/squad/points${query({ chip: chip ?? undefined, horizon, source })}`),

  listPlayers: (filters: {
    position?: string;
    min_price?: number;
    max_price?: number;
    search?: string;
  }) => request<PlayerPanelRowOut[]>(`/players${query(filters)}`),
  getPlayer: (playerId: number, gameweek?: number) =>
    request<PlayerDetailOut>(`/players/${playerId}${query({ gameweek })}`),

  getRecommendedTransfer: () => request<TransferRecommendationOut | null>("/transfers/recommended"),

  getFixtureTicker: (horizon?: number) =>
    request<FixtureTickerRowOut[]>(`/fixtures${query({ horizon })}`),
  getPlayerStats: (gameweekFrom: number, gameweekTo: number) =>
    request<PlayerStatsRowOut[]>(
      `/players/stats${query({ gameweek_from: gameweekFrom, gameweek_to: gameweekTo })}`,
    ),
};
