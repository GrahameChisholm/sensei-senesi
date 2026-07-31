// EP -> colour scale (D12), read directly off the reference screenshots' own values rather than
// invented: Anderson 3.0 -> pink, Groß 3.7 -> tan, João Pedro 4.0 -> tan, N.Williams 4.6 -> green,
// Mbeumo 4.7 -> green. Applied per-gameweek (never a horizon total) so the palette is stable when
// toggling Next GW / Next 3 GWs.

const PINK = { r: 0xf7, g: 0xd7, b: 0xda }; // low EP
const TAN = { r: 0xf3, g: 0xe6, b: 0xc9 }; // mid EP
const GREEN = { r: 0xcd, g: 0xe9, b: 0xd7 }; // high EP

const LOW_BREAKPOINT = 3.0;
const HIGH_BREAKPOINT = 4.3;

function lerp(a: number, b: number, t: number): number {
  return Math.round(a + (b - a) * t);
}

function mix(from: typeof PINK, to: typeof PINK, t: number): string {
  const clamped = Math.max(0, Math.min(1, t));
  const r = lerp(from.r, to.r, clamped);
  const g = lerp(from.g, to.g, clamped);
  const b = lerp(from.b, to.b, clamped);
  return `rgb(${r}, ${g}, ${b})`;
}

/** Background colour for one player's expected-points cell, for one specific gameweek's value. */
export function expectedPointsColour(expectedPoints: number | null): string {
  if (expectedPoints === null) return "#e5e5e5"; // blank gameweek -- visually distinct grey
  if (expectedPoints <= LOW_BREAKPOINT) return mix(PINK, PINK, 0);
  if (expectedPoints >= HIGH_BREAKPOINT) return mix(GREEN, GREEN, 0);
  const mid = (LOW_BREAKPOINT + HIGH_BREAKPOINT) / 2;
  if (expectedPoints <= mid) {
    return mix(PINK, TAN, (expectedPoints - LOW_BREAKPOINT) / (mid - LOW_BREAKPOINT));
  }
  return mix(TAN, GREEN, (expectedPoints - mid) / (HIGH_BREAKPOINT - mid));
}

export function lowConfidenceBorder(): string {
  return "2px dashed #9a6b00";
}
