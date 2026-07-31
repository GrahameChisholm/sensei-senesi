import { useSquadPoints } from "../hooks/useProjections";

/** D19: while a Wildcard/Free Hit draft is open, shows the committed squad's points next to the
 * draft's recalculated total, so the swing is visible before spending the chip. */
export function DraftCompareBar({ draftChip, refreshKey }: { draftChip: string | null; refreshKey: unknown }) {
  const committedPoints = useSquadPoints(null, 1, refreshKey, "committed");
  const draftPoints = useSquadPoints(draftChip, 1, refreshKey, "draft");

  if (!committedPoints || !draftPoints) return null;

  const delta = draftPoints.total - committedPoints.total;

  return (
    <div className="draft-compare-bar">
      <span>Current squad: {committedPoints.total.toFixed(1)} pts</span>
      <span className="arrow">→</span>
      <span>Draft: {draftPoints.total.toFixed(1)} pts</span>
      <span className={delta >= 0 ? "delta-positive" : "delta-negative"}>
        ({delta >= 0 ? "+" : ""}
        {delta.toFixed(1)})
      </span>
    </div>
  );
}
