// Thin client over the FastAPI backend (api/main.py) -- one function per endpoint, typed to
// match api/schemas.py exactly. No logic lives here beyond the HTTP call itself; every
// recommendation/reasoning is computed server-side by features/, never recomputed in the browser.

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'

class ApiError extends Error {}

async function apiGet<T>(path: string): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`)
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new ApiError(body?.detail ?? `${path} failed with HTTP ${response.status}`)
  }
  return (await response.json()) as T
}

async function apiPut<T>(path: string, body: unknown): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const errorBody = await response.json().catch(() => null)
    throw new ApiError(errorBody?.detail ?? `${path} failed with HTTP ${response.status}`)
  }
  return (await response.json()) as T
}

export interface SquadPlayer {
  player_id: number
  position: string
  purchase_price: number
  current_price: number
  sell_price: number
}

export interface MyTeamState {
  squad: SquadPlayer[]
  starting_xi: number[]
  bench_order: number[]
  captain_id: number
  vice_captain_id: number
  bank: number
  free_transfers: number
  chips_remaining: string[]
  total_sell_value: number
}

export interface FixtureDifficulty {
  team_id: number
  opponent_id: number
  gameweek: number
  is_home: boolean
  attack_factor: number
  defense_factor: number
  attack_rating: number
  defense_rating: number
  overall_rating: number
}

export interface CaptaincyOption {
  player_id: number
  position: string
  expected_points: number
  floor: number | null
  ceiling: number | null
  prob_big_haul: number | null
  is_owned: boolean
  is_eligible: boolean
  reasoning: string
}

export interface CaptaincyRecommendation {
  ranked_pool: CaptaincyOption[]
  top_ev_pick: CaptaincyOption | null
  safe_pick: CaptaincyOption | null
  punt_pick: CaptaincyOption | null
}

export interface TransferCandidate {
  sell_player_id: number
  buy_player_id: number
  position: string
  sell_price: number
  buy_price: number
  net_spend: number
  horizon_points_sold: number
  horizon_points_bought: number
  points_gain: number
  hit_cost: number
  net_points_gain: number
  is_forced: boolean
  reasoning: string
}

export interface TransferPlan {
  affordable_candidates: TransferCandidate[]
  recommended: TransferCandidate | null
}

export interface ChipEvaluation {
  chip: string
  target_gameweek: number
  value_now: number
  best_gameweek: number
  best_value: number
  recommendation: 'play_now' | 'wait'
  reasoning: string
}

export interface WildcardEvaluation {
  squad_uplift: number
  upgradeable_slots: number
  recommendation: 'play_now' | 'hold'
  reasoning: string
}

export interface DataStatus {
  generated_at: string | null
  is_demo_data: boolean
}

export interface Settings {
  fpl_team_id: number | null
  mini_league_ids: number[]
  planning_horizon_gameweeks: number
}

export interface ModelPerformanceHeadline {
  overall_mae: number
  overall_rmse: number
  pooled_spearman: number
  top_n_mean_actual: Record<string, number>
  clean_sheet_mace: number
  minutes_played_at_all_mace: number
  minutes_60_plus_mace: number
  defensive_contribution_mace: number
  mean_calibrations_played: Record<
    string,
    { predicted: number; actual: number; relative_gap: number }
  >
  captaincy_hit_rate: number | null
  gate: {
    beats_baselines: boolean
    no_severe_bias: boolean
    calibration_acceptable: boolean
    predictions_logged: boolean
    trusted_by_user: boolean
    passed: boolean
  }
}

export interface ModelPerformance {
  headline: ModelPerformanceHeadline | null
  has_live_accuracy: boolean
}

export interface PlayerSummary {
  player_id: number
  name: string
  position: string
  price: number | null
  gameweek: number
  expected_points: number
}

export interface ComponentBreakdown {
  appearance: number
  goals: number
  assists: number
  clean_sheet: number
  goals_conceded: number
  defensive_contribution: number
  saves: number
  bonus: number
  cards: number
  penalty_misses: number
  own_goals: number
}

export interface PlayerDetail {
  player_id: number
  name: string
  position: string
  price: number | null
  gameweek: number
  expected_points: number
  breakdown: ComponentBreakdown
  floor: number | null
  ceiling: number | null
  prob_big_haul: number | null
}

export interface PlayerSearchParams {
  search?: string
  position?: string
  maxPrice?: number
}

function playerSearchQuery(params: PlayerSearchParams): string {
  const query = new URLSearchParams()
  if (params.search) query.set('search', params.search)
  if (params.position) query.set('position', params.position)
  if (params.maxPrice !== undefined) query.set('max_price', String(params.maxPrice))
  const qs = query.toString()
  return qs ? `?${qs}` : ''
}

export const api = {
  dataStatus: () => apiGet<DataStatus>('/data-status'),
  team: () => apiGet<MyTeamState>('/team'),
  settings: () => apiGet<Settings>('/settings'),
  updateSettings: (settings: Settings) => apiPut<Settings>('/settings', settings),
  modelPerformance: () => apiGet<ModelPerformance>('/model-performance'),
  players: (params: PlayerSearchParams = {}) =>
    apiGet<PlayerSummary[]>(`/players${playerSearchQuery(params)}`),
  player: (playerId: number) => apiGet<PlayerDetail>(`/players/${playerId}`),
  fixtures: (gameweek?: number) =>
    apiGet<FixtureDifficulty[]>(`/fixtures${gameweek ? `?gameweek=${gameweek}` : ''}`),
  captaincy: (gameweek: number) =>
    apiGet<CaptaincyRecommendation>(`/captaincy?gameweek=${gameweek}`),
  transfers: () => apiGet<TransferPlan>('/transfers'),
  benchBoost: (gameweek: number) =>
    apiGet<ChipEvaluation>(`/chips/bench-boost?gameweek=${gameweek}`),
  tripleCaptain: (gameweek: number) =>
    apiGet<ChipEvaluation>(`/chips/triple-captain?gameweek=${gameweek}`),
  freeHit: (gameweek: number) => apiGet<ChipEvaluation>(`/chips/free-hit?gameweek=${gameweek}`),
  wildcard: () => apiGet<WildcardEvaluation>('/chips/wildcard'),
}
