import { PlayerStatsRowOut } from "../api";

export type StatKey =
  | "goals_scored"
  | "assists"
  | "clean_sheets"
  | "goals_conceded"
  | "own_goals"
  | "penalties_missed"
  | "penalties_saved"
  | "saves"
  | "bonus"
  | "yellow_cards"
  | "red_cards"
  | "total_points"
  | "expected_goals"
  | "expected_assists"
  | "expected_goal_involvements"
  | "expected_goals_conceded";

export interface StatColumn {
  key: StatKey;
  label: string;
  decimals: number;
  perNinetyEligible: boolean;
}

export const STAT_COLUMNS: StatColumn[] = [
  { key: "goals_scored", label: "Goals", decimals: 0, perNinetyEligible: true },
  { key: "assists", label: "Assists", decimals: 0, perNinetyEligible: true },
  { key: "clean_sheets", label: "CS", decimals: 0, perNinetyEligible: true },
  { key: "goals_conceded", label: "GC", decimals: 0, perNinetyEligible: true },
  { key: "own_goals", label: "OG", decimals: 0, perNinetyEligible: false },
  { key: "penalties_missed", label: "Pen miss", decimals: 0, perNinetyEligible: false },
  { key: "penalties_saved", label: "Pen saved", decimals: 0, perNinetyEligible: false },
  { key: "saves", label: "Saves", decimals: 0, perNinetyEligible: true },
  { key: "bonus", label: "Bonus", decimals: 0, perNinetyEligible: true },
  { key: "yellow_cards", label: "YC", decimals: 0, perNinetyEligible: false },
  { key: "red_cards", label: "RC", decimals: 0, perNinetyEligible: false },
  { key: "total_points", label: "Pts", decimals: 0, perNinetyEligible: true },
  { key: "expected_goals", label: "xG", decimals: 2, perNinetyEligible: true },
  { key: "expected_assists", label: "xA", decimals: 2, perNinetyEligible: true },
  { key: "expected_goal_involvements", label: "xGI", decimals: 2, perNinetyEligible: true },
  { key: "expected_goals_conceded", label: "xGC", decimals: 2, perNinetyEligible: true },
];

// Average minutes per match played, not total minutes across the range -- two players who
// played 90 minutes in one game out of five should read the same here, regardless of how wide
// the selected gameweek range is. row.actuals.minutes stays the *total* everywhere (statValue's
// per-90 rate denominator below needs the total, not this average).
export function averageMinutesPerMatch(row: PlayerStatsRowOut): number {
  if (row.actuals.apps <= 0) return 0;
  return row.actuals.minutes / row.actuals.apps;
}

export function statValue(row: PlayerStatsRowOut, column: StatColumn, perNinety: boolean): number {
  const raw = row.actuals[column.key];
  if (!perNinety || !column.perNinetyEligible) return raw;
  if (row.actuals.minutes <= 0) return 0;
  return (raw * 90) / row.actuals.minutes;
}

export function formatStat(value: number, column: StatColumn, perNinety: boolean): string {
  // A per-90 rate (e.g. 1.5 goals/90) needs more precision than the raw integer count it's
  // derived from -- toFixed(column.decimals) would round it to a whole number otherwise.
  const decimals = perNinety && column.perNinetyEligible ? Math.max(column.decimals, 2) : column.decimals;
  return value.toFixed(decimals);
}

export function horizonPoints(row: PlayerStatsRowOut): number {
  return row.fixtures.reduce((total, fixture) => total + (fixture.expected_points ?? 0), 0);
}
