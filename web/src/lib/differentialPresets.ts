import { OwnershipLensSource } from "../api";

export type DifferentialSortKey =
  | "name"
  | "price"
  | "current_ownership_percent"
  | "league_owner_count"
  | "ownership_trend_pct_per_gw"
  | "shrunk_points_per_90"
  | "surplus_vs_bracket"
  | "expected_swing"
  | "xgi_per_90"
  | "defensive_contribution_per_90"
  | "confidence";

export type DifferentialIntent = "attack" | "consolidate" | "best_available" | "custom";

export interface DifferentialPreset {
  key: Exclude<DifferentialIntent, "custom">;
  label: string;
  description: string;
  windowGameweeks: number;
  maxOwnershipPercent: number;
  maxLeagueOwners: number | undefined;
}

/** Three fixed starting points replacing three independently-tuned knobs (the window length, the
 * ownership ceiling, and hide-owned) -- each bundles a window and an ownership ceiling under both
 * lenses so switching leagues on or off never leaves a preset half-configured. "Best available"
 * sends `maxLeagueOwners: undefined` so the server applies no ceiling at all, keeping it
 * independent of how many rivals turn out to be in the league. */
export const DIFFERENTIAL_PRESETS: DifferentialPreset[] = [
  {
    key: "attack",
    label: "Attack",
    description: "Near-exclusive picks almost nobody else has, over a short window.",
    windowGameweeks: 6,
    maxOwnershipPercent: 5,
    maxLeagueOwners: 1,
  },
  {
    key: "consolidate",
    label: "Consolidate",
    description: "A wider evidence check before committing a transfer.",
    windowGameweeks: 8,
    maxOwnershipPercent: 25,
    maxLeagueOwners: 4,
  },
  {
    key: "best_available",
    label: "Best available",
    description: "Every qualifying player, ranked purely by output, no ownership ceiling.",
    windowGameweeks: 8,
    maxOwnershipPercent: 50,
    maxLeagueOwners: undefined,
  },
];

/** The decision metric each preset should sort by, once the active ownership lens is known
 * (MINI_LEAGUE_PLAN-adjacent, item 3): expected swing under the league lens where it exists,
 * surplus-vs-bracket otherwise -- "Best available" always sorts by output regardless of lens,
 * since a swing figure that's about to be null under the global lens defeats its own purpose. */
export function presetSortKey(
  preset: DifferentialPreset,
  ownershipLens: OwnershipLensSource,
): DifferentialSortKey {
  if (preset.key === "best_available") return "surplus_vs_bracket";
  if (preset.key === "consolidate") return "confidence";
  return ownershipLens === "league" ? "expected_swing" : "surplus_vs_bracket";
}
