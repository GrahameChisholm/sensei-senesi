/** Bump chart colour assignment for up to 12 equally-weighted lines (no self-highlight, per the
 * Roundup design). The eight hues below are a validated categorical palette (adjacent-pair CVD
 * Delta E >= 8, normal-vision floor >= 15 against a white surface -- checked, not eyeballed) in
 * a fixed order that must never be re-cycled or extended past 8 with a new hue: past that, hue
 * alone stops reliably carrying identity even for readers without a colour vision deficiency.
 *
 * A 12-manager league needs 4 more lines than that. Rather than invent unvalidated colours 9-12,
 * slots 9-12 reuse hues 1-4 with a dashed stroke as a second, composite channel -- a manager on
 * a dashed blue line is never confused with the solid blue line, and every line also ends in a
 * direct name label, which is the chart's real disambiguator when two lines are near each other.
 */

const HUES = [
  "#2a78d6", // blue
  "#eb6834", // orange
  "#1baf7a", // aqua
  "#eda100", // yellow
  "#e87ba4", // magenta
  "#008300", // green
  "#4a3aa7", // violet
  "#e34948", // red
];

export interface LineStyle {
  colour: string;
  dashed: boolean;
}

/** One style per line, assigned by a stable index (e.g. sort order of entry_id), not by rank or
 * any other value that can change week to week -- colour must follow the entity. */
export function bumpChartLineStyle(index: number): LineStyle {
  const hue = HUES[index % HUES.length];
  const dashed = index >= HUES.length;
  return { colour: hue, dashed };
}
