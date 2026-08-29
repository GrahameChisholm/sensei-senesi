// EP -> colour scale (D12), read directly off the reference screenshots' own values rather than
// invented: Anderson 3.0 -> pink, Groß 3.7 -> tan, João Pedro 4.0 -> tan, N.Williams 4.6 -> green,
// Mbeumo 4.7 -> green. Applied per-gameweek (never a horizon total) so the palette is stable when
// toggling Next GW / Next 3 GWs.
//
// Bg colours interpolate continuously across the low/mid/high range; the paired text colour is
// bucketed into the same three tiers (a continuously-interpolated text colour reads muddy against
// a continuously-interpolated background, and the badge only needs to signal which tier it's in).

const LOW = { bg: { r: 0xf8, g: 0xdc, b: 0xda }, text: "#a8382c" }; // low EP
const MID = { bg: { r: 0xf7, g: 0xec, b: 0xc8 }, text: "#8a6a1f" }; // mid EP
const HIGH = { bg: { r: 0xdc, g: 0xf0, b: 0xe2 }, text: "#1c7a3f" }; // high EP
const BLANK_BG = "#ececE7";
const BLANK_TEXT = "#767a80";

const LOW_BREAKPOINT = 3.0;
const HIGH_BREAKPOINT = 4.3;

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function mixBg(from: typeof LOW.bg, to: typeof LOW.bg, t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const r = lerp(from.r, to.r, clamped);
  const g = lerp(from.g, to.g, clamped);
  const b = lerp(from.b, to.b, clamped);
  return `rgb(${r}, ${g}, ${b})`;
}

function epMidpoint(): number {
  return (LOW_BREAKPOINT + HIGH_BREAKPOINT) / 2;
}

/** Background colour for one player's expected-points cell, for one specific gameweek's value. */
export function expectedPointsColour(expectedPoints: number | null): string {
  if (expectedPoints === null) return BLANK_BG; // blank gameweek -- visually distinct grey
  if (expectedPoints <= LOW_BREAKPOINT) return mixBg(LOW.bg, LOW.bg, 0);
  if (expectedPoints >= HIGH_BREAKPOINT) return mixBg(HIGH.bg, HIGH.bg, 0);
  const mid = epMidpoint();
  if (expectedPoints <= mid) {
    return mixBg(LOW.bg, MID.bg, (expectedPoints - LOW_BREAKPOINT) / (mid - LOW_BREAKPOINT));
  }
  return mixBg(MID.bg, HIGH.bg, (expectedPoints - mid) / (HIGH_BREAKPOINT - mid));
}

/** Text colour paired with {@link expectedPointsColour} for the same value -- bucketed (not
 * interpolated) into the same low/mid/high tiers so the number reads clearly against its cell. */
export function expectedPointsTextColour(expectedPoints: number | null): string {
  if (expectedPoints === null) return BLANK_TEXT;
  if (expectedPoints <= LOW_BREAKPOINT) return LOW.text;
  if (expectedPoints >= HIGH_BREAKPOINT) return HIGH.text;
  return MID.text;
}

export function lowConfidenceBorder(): string {
  return "2px dashed #b5892b";
}

// Fixture difficulty rating (1 easiest to 5 hardest, FPL's own team_h_difficulty/
// team_a_difficulty scale) -> colour, the same green-to-red convention Fantasy Football Hub's own
// fixture ticker uses, recast as the app's pastel-bg/bold-text pairing so it sits on the same
// visual language as the EP heatmap. A discrete lookup, not a continuous mix, since the rating is
// already a whole number 1 through 5.
const DIFFICULTY_COLOURS: Record<number, { bg: string; text: string }> = {
  1: { bg: "#dcf0e2", text: "#1c7a3f" },
  2: { bg: "#e8eed7", text: "#4f7a3a" },
  3: { bg: "#f7ecc8", text: "#8a6a1f" },
  4: { bg: "#f5ddd0", text: "#a8532c" },
  5: { bg: "#f8dcda", text: "#a8382c" },
};

function difficultyBucket(difficulty: number): number {
  return Math.min(5, Math.max(1, Math.round(difficulty)));
}

/** Background colour for one fixture cell. Null means a blank gameweek, shaded the same neutral
 * grey expectedPointsColour already uses for a blank. */
export function fixtureDifficultyColour(difficulty: number | null): string {
  if (difficulty === null) return BLANK_BG;
  return DIFFICULTY_COLOURS[difficultyBucket(difficulty)].bg;
}

/** Text colour paired with {@link fixtureDifficultyColour} for the same value. */
export function fixtureDifficultyTextColour(difficulty: number | null): string {
  if (difficulty === null) return BLANK_TEXT;
  return DIFFICULTY_COLOURS[difficultyBucket(difficulty)].text;
}

// Mini League exposure/swing (MINI_LEAGUE_PLAN M20): a signed value needs a diverging scale, not
// the one-directional low/mid/high EP heatmap above -- reuses the same LOW (red)/HIGH (green)
// tokens from that palette rather than inventing a new one, applied symmetrically around zero.
const EXPOSURE_MAGNITUDE_BREAKPOINT = 6.0; // |expected_swing| at or beyond which the colour is fully saturated
const NEUTRAL_BG_RGB = { r: 0xf0, g: 0xef, b: 0xec }; // --surface-muted
const NEUTRAL_TEXT = "#767a80"; // --text-muted

/** Background colour for one player's signed exposure/expected-swing cell. Symmetric around
 * zero: strongly negative is as saturated red as strongly positive is saturated green. */
export function exposureColour(value: number | null): string {
  if (value === null) return BLANK_BG;
  if (value === 0) return mixBg(NEUTRAL_BG_RGB, NEUTRAL_BG_RGB, 0);
  const t = Math.min(1, Math.abs(value) / EXPOSURE_MAGNITUDE_BREAKPOINT);
  return value > 0 ? mixBg(NEUTRAL_BG_RGB, HIGH.bg, t) : mixBg(NEUTRAL_BG_RGB, LOW.bg, t);
}

/** Text colour paired with {@link exposureColour} for the same value. */
export function exposureTextColour(value: number | null): string {
  if (value === null) return BLANK_TEXT;
  if (value === 0) return NEUTRAL_TEXT;
  return value > 0 ? HIGH.text : LOW.text;
}
