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

export const api = {
  team: () => apiGet<MyTeamState>('/team'),
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
