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
  /** The decision gameweek: the earliest one still open, which advances at a deadline rather than
   * when a gameweek's matches finish. */
  gameweek: number;
  /** The gameweek the loaded projection cache was built for. Behind `gameweek` when a deadline has
   * passed since the last build, which means the numbers are fit on older data than a rebuild
   * would use. */
  projections_gameweek: number;
  /** `gameweek`'s own deadline, and whether it has passed, both against the clock now. */
  deadline_time: string;
  deadline_passed: boolean;
  generated_at: string;
  model_version: string;
  /** `gameweek` onward, with any already-locked gameweeks dropped. */
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
  expected_goals_for: number | null;
  expected_goals_against: number | null;
}

export interface FixtureTickerCellOut {
  gameweek: number;
  fixtures: FixtureTickerCellFixtureOut[];
}

export interface FixtureTickerRowOut {
  team_id: number;
  gameweeks: FixtureTickerCellOut[];
  average_difficulty: number | null;
  total_expected_goals_for: number | null;
  total_expected_goals_against: number | null;
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
  defensive_contribution: number;
  yellow_cards: number;
  red_cards: number;
  total_points: number;
  expected_goals: number;
  expected_assists: number;
  expected_goal_involvements: number;
  expected_goals_conceded: number;
  points_breakdown: ComponentBreakdownOut;
  ownership_percent: number | null;
  small_sample: boolean;
  attacking_ratio: RateRatioOut | null;
  defensive_ratio: RateRatioOut | null;
  is_penalty_taker: boolean;
}

/** An actual-vs-expected ratio with its credible interval. An interval containing 1.0 means the
 * deviation is not yet distinguishable from chance. */
export interface RateRatioOut {
  ratio: number;
  low: number;
  high: number;
  exposure: number;
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

/** Why league ownership may be absent. This page shows mini-league ownership or nothing, never
 * FPL's population-wide figure substituted under the same heading. */
export type OwnershipStatus = "ok" | "not_configured" | "fetch_failed" | "no_rivals";

export interface PlayerStatsResponseOut {
  ownership_status: OwnershipStatus;
  rows: PlayerStatsRowOut[];
}

// --- Differentials page ---------------------------------------------------------------------

export interface DifferentialsWindowOut {
  gameweek_from: number;
  gameweek_to: number;
  requested_gameweeks: number;
}

export type Confidence = "low" | "medium" | "high";
export type Archetype = "proven" | "emerging" | "riding_luck" | "none";

export interface SwapCandidateOut {
  incoming_player_id: number;
  outgoing_player_id: number;
  incoming_swing: number;
  outgoing_swing: number;
  net_swing_delta: number;
  price_delta: number;
}

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
  // League ownership lens columns (MINI_LEAGUE_PLAN M28) -- null/[] under the global lens, or for
  // a player no rival owns under the league lens.
  league_owner_count: number | null;
  league_eo_multiplier: number | null;
  league_owner_names: string[];
  // Prospective swing: what this player would be worth in expected-swing terms if brought into
  // the starting XI. League lens only.
  expected_swing: number | null;
  // Which of your own starting XI this player would most sensibly replace by expected swing.
  // League lens with a complete squad only; a swing comparison, not a budget/legality check.
  replaces: SwapCandidateOut | null;
}

export type OwnershipLensSource = "global" | "league";

export interface DifferentialsResponseOut {
  window: DifferentialsWindowOut;
  ownership_lens: OwnershipLensSource;
  picks_gameweek: number | null;
  n_rivals: number | null;
  rows: DifferentialRowOut[];
}

// --- Mini League page -----------------------------------------------------------------------

export interface MiniLeagueSettingsOut {
  fpl_team_id: number | null;
  mini_league_ids: number[];
}

export interface PlayerOwnershipOut {
  player_id: number;
  raw_ownership_percent: number;
  eo_multiplier: number;
  eo_percent: number;
  captain_share_percent: number;
  owner_names: string[];
}

export interface PlayerExposureOut {
  player_id: number;
  your_multiplier: number;
  ownership: PlayerOwnershipOut;
  expected_points: number | null;
  exposure: number;
  expected_swing: number | null;
}

export interface CaptainOptionOut {
  player_id: number;
  expected_points: number | null;
  captain_share_percent: number;
  eo_multiplier: number;
  net_captain_ev: number | null;
  net_captain_std: number | null;
}

export interface DifferentialPickOut {
  player_id: number;
  your_multiplier: number;
  rival_multiplier: number;
  expected_points: number | null;
  expected_gap_contribution: number;
}

export interface HeadToHeadOut {
  rival_entry_id: number;
  shared_count: number;
  differentials: DifferentialPickOut[];
  expected_gap: number;
  gap_std: number;
  p_outscore: number;
}

export interface RivalChipStateOut {
  entry_id: number;
  used_chip_names: string[];
  remaining_chip_names: string[];
}

export type VariancePreference = "increase" | "decrease" | "neutral";

export interface RivalPostureOut {
  rival_entry_id: number;
  projected_final_gap: number;
  p_finish_ahead: number;
  variance_preference: VariancePreference;
  sensitivity: number;
}

export interface MiniLeagueRivalOut {
  entry_id: number;
  manager_name: string;
  team_name: string;
  rank: number;
  total_points: number;
  gameweek_points: number;
  chip_state: RivalChipStateOut;
  posture: RivalPostureOut;
  head_to_head: HeadToHeadOut;
}

export type LeagueInsightKind = "edge" | "drag" | "captain";

export interface LeagueInsightOut {
  kind: LeagueInsightKind;
  player_id: number;
  reference_player_id: number | null;
  value: number;
  owner_count: number;
  n_rivals: number;
}

export interface MiniLeaguePanelOut {
  league_id: number;
  league_name: string;
  picks_gameweek: number;
  gameweek: number;
  my_rank: number;
  my_total_points: number;
  coverage: number;
  template_xi: number[];
  exposures: PlayerExposureOut[];
  captain_options: CaptainOptionOut[];
  insights: LeagueInsightOut[];
  rivals: MiniLeagueRivalOut[];
}

export interface TransferMoveOut {
  out_player_id: number;
  in_player_id: number;
  out_name: string;
  in_name: string;
  position: string;
  /** Incoming price minus outgoing, in tenths of a million. Negative frees money. */
  price_delta: number;
  out_expected_points: number | null;
  in_expected_points: number | null;
  /** The field's effective ownership of the incoming player, or null with no league in play.
   * Above 1 means the league is already captaining him; near 0 means he is a differential. */
  in_eo_multiplier: number | null;
}

export interface TransferPlanOut {
  moves: TransferMoveOut[];
  out_player_ids: number[];
  in_player_ids: number[];
  n_transfers: number;
  expected_points: number;
  expected_points_delta: number;
  expected_gap: number;
  /** Identical to expected_points_delta by construction: the field's ownership cancels out of a
   * transfer, so a plan's gain against the league IS its points gain. Kept as its own field
   * because the banner shows the level (expected_gap), not only the change. */
  expected_gap_delta: number;
  gap_std: number;
  /** Positive widens your gap distribution against the field, negative narrows it. Which of those
   * you want is what variance_preference says. */
  gap_std_delta: number;
  expected_final_rank: number;
  /** New minus current, so NEGATIVE is an improvement in where you finish. */
  expected_final_rank_delta: number;
  spend_delta: number;
  budget_remaining: number;
}

export interface TransferSuggestionOut {
  plans: TransferPlanOut[];
  /** One plan per transfer count from 1 upward, so the banner can show what each extra move buys. */
  best_by_transfer_count: TransferPlanOut[];
  marginal_points_gains: number[];
  max_transfers: number;
  max_transfers_allowed: number;
  current_expected_points: number;
  current_expected_gap: number;
  current_gap_std: number;
  current_expected_final_rank: number;
  variance_preference: "increase" | "decrease" | "neutral";
  n_rivals: number;
  /** Null when no league is in play, in which case plans are ranked on expected points alone. */
  league_id: number | null;
  league_name: string;
  picks_gameweek: number | null;
  gameweeks: number[];
  league_gameweek: number;
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
  optimise: (objective: "full_squad" | "starting_xi" = "full_squad", captainMultiplier = 2.0) =>
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
    request<PlayerStatsResponseOut>(
      `/players/stats${query({ gameweek_from: gameweekFrom, gameweek_to: gameweekTo })}`,
    ),
  getDifferentials: (options: {
    window: number;
    maxOwnership?: number;
    maxLeagueOwners?: number;
    leagueId?: number;
    hideOwned?: boolean;
  }) =>
    request<DifferentialsResponseOut>(
      `/players/differentials${query({
        window: options.window,
        max_ownership: options.maxOwnership,
        max_league_owners: options.maxLeagueOwners,
        league_id: options.leagueId,
        hide_owned: options.hideOwned ?? true,
      })}`,
    ),

  getTransferSuggestions: (options: {
    transfers: number;
    horizon?: number;
    chip?: string | null;
    leagueId?: number;
  }) =>
    request<TransferSuggestionOut>(
      `/squad/transfers${query({
        transfers: options.transfers,
        horizon: options.horizon ?? 1,
        chip: options.chip ?? undefined,
        league_id: options.leagueId,
      })}`,
    ),
  applyTransfers: (outPlayerIds: number[], inPlayerIds: number[]) =>
    request<SquadOut>("/squad/transfers/apply", {
      method: "POST",
      body: JSON.stringify({ out_player_ids: outPlayerIds, in_player_ids: inPlayerIds }),
    }),

  getMiniLeagueSettings: () => request<MiniLeagueSettingsOut>("/mini-league/leagues"),
  setMiniLeagueSettings: (fplTeamId: number | null, miniLeagueIds: number[]) =>
    request<MiniLeagueSettingsOut>("/mini-league/leagues", {
      method: "POST",
      body: JSON.stringify({ fpl_team_id: fplTeamId, mini_league_ids: miniLeagueIds }),
    }),
  getMiniLeague: (leagueId: number, options?: { refresh?: boolean; chip?: string | null }) =>
    request<MiniLeaguePanelOut>(
      `/mini-league/${leagueId}${query({ refresh: options?.refresh, chip: options?.chip })}`,
    ),
};
