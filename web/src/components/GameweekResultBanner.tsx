import { AdvanceResultOut } from "../api";

/** Season Replay's post-Advance reveal: what actually happened in the gameweek just scored. */
export function GameweekResultBanner({
  result,
  onDismiss,
}: {
  result: AdvanceResultOut | null;
  onDismiss: () => void;
}) {
  if (!result) return null;

  return (
    <div className="gameweek-result-banner" role="status">
      <span>
        GW{result.gameweek}: scored <strong>{result.points.toFixed(1)}</strong> points
        {result.chip_played && ` (${result.chip_played} played)`}
        {result.hit_cost > 0 && ` (−${result.hit_cost} hit)`} — running total{" "}
        {result.running_total.toFixed(1)}
        {result.season_complete && " — season complete!"}
      </span>
      <button onClick={onDismiss}>×</button>
    </div>
  );
}
