import { OwnershipStatus, PlayerStatsRowOut, RateRatioOut } from "../api";

export type StatKey =
  | "goals_scored"
  | "assists"
  | "defensive_contribution"
  | "clean_sheets"
  | "goals_conceded"
  | "bonus"
  | "total_points"
  | "expected_goals"
  | "expected_assists"
  | "expected_goal_involvements"
  | "expected_goals_conceded";

/** Which colour-tinted header band (PLAYER_STATS_PLAN restructure) a column belongs to. Meta and
 * the ratio/fixture columns aren't driven by STAT_COLUMNS at all (they're hardcoded cells in
 * PlayerStatsTable), so their band membership lives there instead of here. */
export type StatGroupKey = "historic" | "points" | "expected";

export interface StatColumn {
  key: StatKey;
  label: string;
  decimals: number;
  perNinetyEligible: boolean;
  group: StatGroupKey;
  /** Explains what the column is and, for a derived stat, roughly how it's calculated -- shown as
   * the header's tooltip so a number is never on screen with no explanation behind it. */
  tooltip: string;
}

export const STAT_COLUMNS: StatColumn[] = [
  {
    key: "goals_scored",
    label: "Goals",
    decimals: 0,
    perNinetyEligible: true,
    group: "historic",
    tooltip: "Goals scored, summed over the selected gameweek range.",
  },
  {
    key: "assists",
    label: "Assists",
    decimals: 0,
    perNinetyEligible: true,
    group: "historic",
    tooltip: "Assists, summed over the selected gameweek range.",
  },
  {
    key: "defensive_contribution",
    label: "DC",
    decimals: 0,
    perNinetyEligible: true,
    group: "historic",
    tooltip:
      "Defensive contribution points: 2 points in a gameweek a player clears the combined tackles/clearances/blocks/interceptions/recoveries threshold for their position, summed over the range.",
  },
  {
    key: "clean_sheets",
    label: "CS",
    decimals: 0,
    perNinetyEligible: true,
    group: "historic",
    tooltip:
      "Clean sheets: gameweeks the player's team conceded 0 goals while they played 60+ minutes.",
  },
  {
    key: "goals_conceded",
    label: "GC",
    decimals: 0,
    perNinetyEligible: true,
    group: "historic",
    tooltip: "Goals conceded by the player's team while they were on the pitch.",
  },
  {
    key: "bonus",
    label: "Bonus",
    decimals: 0,
    perNinetyEligible: true,
    group: "historic",
    tooltip:
      "Bonus points, awarded to the top 3 Bonus Points System (BPS) scorers in each match.",
  },
  {
    key: "total_points",
    label: "Pts",
    decimals: 0,
    perNinetyEligible: true,
    group: "points",
    tooltip:
      "Actual FPL points scored, summed over the range. Click a value to see the breakdown by component (goals, assists, clean sheets, bonus, ...).",
  },
  {
    key: "expected_goals",
    label: "xG",
    decimals: 2,
    perNinetyEligible: true,
    group: "expected",
    tooltip:
      "Expected goals (Opta), summed over the range: the quality of chances taken, independent of whether they actually went in.",
  },
  {
    key: "expected_assists",
    label: "xA",
    decimals: 2,
    perNinetyEligible: true,
    group: "expected",
    tooltip:
      "Expected assists (Opta), summed over the range: the quality of chances created for teammates, independent of whether they actually went in.",
  },
  {
    key: "expected_goal_involvements",
    label: "xGI",
    decimals: 2,
    perNinetyEligible: true,
    group: "expected",
    tooltip: "Expected goal involvements: xG plus xA, summed over the range.",
  },
  {
    key: "expected_goals_conceded",
    label: "xGC",
    decimals: 2,
    perNinetyEligible: true,
    group: "expected",
    tooltip:
      "Expected goals conceded by the player's team over the range, from the team's own underlying defensive numbers rather than the goals actually let in.",
  },
];

/** Label and CSS class for each colour-tinted header band, in table order. Meta has no entry --
 * it's deliberately left untinted (see PlayerStatsTable), and the ratio/fixture bands are declared
 * directly in PlayerStatsTable since they aren't backed by STAT_COLUMNS. */
export const STAT_GROUP_META: Record<StatGroupKey, { label: string; className: string }> = {
  historic: { label: "Historic Stats", className: "stats-group-band--historic" },
  points: { label: "Points", className: "stats-group-band--points" },
  expected: { label: "Expected", className: "stats-group-band--expected" },
};

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

/** Total points scored per £m of price, over the selected range -- a cumulative value-for-money
 * figure, not a rate, so unlike every STAT_COLUMNS entry it is deliberately unaffected by the
 * per-90 toggle. `price` is null for a player with no current price on record; 0 there or a
 * (never expected, but zero-guarded) zero price reads as "no value to compute" rather than
 * dividing by zero. */
export function pointsPerMillion(row: PlayerStatsRowOut): number {
  if (row.price === null || row.price === 0) return 0;
  return row.actuals.total_points / (row.price / 10);
}

/** Why the Own% column is empty, worded for the column header's tooltip. This page shows
 * mini-league ownership or nothing at all, so the reason is always stated rather than the number
 * being quietly replaced by FPL's population-wide figure. */
export const OWNERSHIP_STATUS_TOOLTIP: Record<OwnershipStatus, string> = {
  ok: "Percent of your mini-league rivals who own this player",
  not_configured: "No mini league configured. Set one up on the Mini League page.",
  fetch_failed: "Could not reach FPL to load your mini league.",
  no_rivals: "Your mini league has no other members to compare against.",
};

// --- Actual vs expected ------------------------------------------------------------------------
//
// Both ratio columns show a shrunk posterior mean (see engine/rates.py) rather than a raw
// difference of per-90 rates, which is what makes them safe to sort: a thin-sample fluke lands
// near 1.0 with a wide interval instead of at the top of the table.

/** A ratio is only a *signal* when its whole credible interval clears 1.0. Anything else is still
 * consistent with chance, however far the point estimate happens to sit from 1.0. */
export type RatioVerdict = "hot" | "cold" | "inconclusive";

export function ratioVerdict(ratio: RateRatioOut | null): RatioVerdict {
  if (ratio === null) return "inconclusive";
  if (ratio.low > 1.0) return "hot";
  if (ratio.high < 1.0) return "cold";
  return "inconclusive";
}

export function formatRatio(ratio: RateRatioOut | null): string {
  return ratio === null ? "—" : `${ratio.ratio.toFixed(2)}×`;
}

/** What each ratio column is, for a column/row header's tooltip -- distinct from
 * {@link ratioTooltip}, which explains one specific cell's value. */
export const RATIO_COLUMN_TOOLTIP: Record<"attacking" | "defensive", string> = {
  attacking:
    "Goals and assists actually scored, against expected goal involvements (xGI). Shrunk toward 1.0 by how much evidence backs it, so a thin sample cannot top the sort. Above 1x means running hot (expect regression); below 1x means running cold (a positive-regression candidate).",
  defensive:
    "Clean sheets actually kept, against expected clean sheets (from each match's expected goals conceded). Shown only for positions a clean sheet pays (GK/DEF/MID). Same shrinkage and above/below-1x reading as vs xGI.",
};

/** Sort key for a ratio column. Nulls sort to the bottom, and an inconclusive ratio sorts by its
 * point estimate like any other, since the column is "how hot", not "how certain". */
export function ratioSortValue(ratio: RateRatioOut | null): number {
  return ratio === null ? Number.NEGATIVE_INFINITY : ratio.ratio;
}

/** A position whose players show no spread beyond Poisson noise gets an infinite prior, collapsing
 * every ratio there to exactly 1.0. The zero-width interval that produces would otherwise read as
 * total certainty when it means the opposite: nothing is distinguishable yet. */
function isDegenerate(ratio: RateRatioOut): boolean {
  return ratio.low === ratio.high;
}

function describeInterval(ratio: RateRatioOut): string {
  return `90% credible interval ${ratio.low.toFixed(2)}× to ${ratio.high.toFixed(2)}×`;
}

/** Tooltip for one ratio cell. Always states the interval, then what it does or does not support,
 * so the number is never read as more certain than it is. */
export function ratioTooltip(ratio: RateRatioOut | null, kind: "attacking" | "defensive"): string {
  if (ratio === null) {
    return kind === "attacking"
      ? "No expected goal involvements in this range to compare against."
      : "No clean sheet points at stake for this position.";
  }
  const subject = kind === "attacking" ? "Goals and assists" : "Clean sheets";
  const against = kind === "attacking" ? "expected goal involvements" : "expected clean sheets";
  if (isDegenerate(ratio)) {
    return `${subject} versus ${against}. Over this range, nothing separates players in this position beyond chance, so everyone sits at expectation.`;
  }
  const verdict = ratioVerdict(ratio);
  const reading =
    verdict === "hot"
      ? "Running ahead of the underlying numbers by more than chance explains, so expect regression."
      : verdict === "cold"
        ? "Running behind the underlying numbers by more than chance explains, a candidate for positive regression."
        : "Still consistent with chance, so not yet evidence of anything.";
  return `${subject} versus ${against}. ${describeInterval(ratio)}. ${reading}`;
}

/** The at-a-glance badge, deliberately worded differently from the Differentials page's
 * Proven/Emerging/Riding Luck so the two vocabularies cannot be confused: this one is derived
 * from the posterior interval, that one from a raw threshold. */
export function overperformanceBadge(
  row: PlayerStatsRowOut,
): { label: string; className: string; title: string } | null {
  const candidates: Array<{ ratio: RateRatioOut | null; kind: "attacking" | "defensive" }> = [
    { ratio: row.actuals.attacking_ratio, kind: "attacking" },
    { ratio: row.actuals.defensive_ratio, kind: "defensive" },
  ];
  // Where both axes have a verdict, lead with whichever sits further from expectation.
  const decisive = candidates
    .filter((c) => c.ratio !== null && ratioVerdict(c.ratio) !== "inconclusive")
    .sort((a, b) => Math.abs((b.ratio?.ratio ?? 1) - 1) - Math.abs((a.ratio?.ratio ?? 1) - 1))[0];
  if (decisive === undefined || decisive.ratio === null) return null;

  const hot = ratioVerdict(decisive.ratio) === "hot";
  const axis = decisive.kind === "attacking" ? "finishing" : "clean sheets";
  return {
    label: hot ? "Running hot" : "Running cold",
    className: hot ? "overperf-hot" : "overperf-cold",
    title: `${hot ? "Overperforming" : "Underperforming"} on ${axis}. ${ratioTooltip(
      decisive.ratio,
      decisive.kind,
    )}`,
  };
}
