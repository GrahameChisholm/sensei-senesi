import { MiniLeaguePanelOut } from "../api";

const VARIANCE_VERDICT: Record<string, string> = {
  increase: "Add variance",
  decrease: "Kill variance",
  neutral: "Hold steady",
};

function projectedRank(panel: MiniLeaguePanelOut): number {
  // MINI_LEAGUE_PLAN M12: summing each rival's own probability of finishing ahead of you gives
  // the expected number of rivals who end up above you, hence the expected final position.
  const expectedRivalsAhead = panel.rivals.reduce(
    (sum, rival) => sum + (1 - rival.posture.p_finish_ahead),
    0,
  );
  return 1 + expectedRivalsAhead;
}

interface MiniLeagueHeaderProps {
  panel: MiniLeaguePanelOut;
  targetRivalId: number | null;
  onTargetRivalChange: (entryId: number) => void;
  onRefresh: () => void;
}

/** The page's "where do I stand, what should I be doing" strip (MINI_LEAGUE_PLAN M22 zone 2).
 * Carries a target-rival selector because a single league-wide posture verdict is wrong whenever
 * a manager is sandwiched between two rivals -- which is the normal case, not the exception -- so
 * the verdict shown here is always for one named rival, defaulting to whoever sits directly above. */
export function MiniLeagueHeader({
  panel,
  targetRivalId,
  onTargetRivalChange,
  onRefresh,
}: MiniLeagueHeaderProps) {
  const target = panel.rivals.find((rival) => rival.entry_id === targetRivalId) ?? null;
  const expectedSwing = panel.exposures.reduce((sum, e) => sum + (e.expected_swing ?? 0), 0);
  const rankedRivals = [...panel.rivals].sort((a, b) => a.rank - b.rank);

  return (
    <div className="gameweek-header">
      <div className="gameweek-header-row">
        <div className="gameweek-title">
          <h2>{panel.league_name}</h2>
          <p className="deadline">GW{panel.gameweek}</p>
        </div>

        {rankedRivals.length > 0 && (
          <label className="stats-range-label">
            Target
            <select
              value={targetRivalId ?? ""}
              onChange={(event) => onTargetRivalChange(Number(event.target.value))}
            >
              {rankedRivals.map((rival) => (
                <option key={rival.entry_id} value={rival.entry_id}>
                  {rival.manager_name} (#{rival.rank})
                </option>
              ))}
            </select>
          </label>
        )}

        <div className="header-tiles">
          <div className="tile">
            <div className="tile-value">#{panel.my_rank}</div>
            <div className="tile-label">Rank</div>
          </div>
          <div className="tile">
            <div className="tile-value">#{projectedRank(panel).toFixed(1)}</div>
            <div className="tile-label">Projected finish</div>
          </div>
          <div className="tile">
            <div className="tile-value">{(panel.coverage * 100).toFixed(0)}%</div>
            <div className="tile-label">Template coverage</div>
          </div>
          <div className="tile">
            <div className="tile-value">
              {expectedSwing >= 0 ? "+" : ""}
              {expectedSwing.toFixed(1)}
            </div>
            <div className="tile-label">Expected swing (GW{panel.gameweek})</div>
          </div>
        </div>
      </div>

      <div className="gameweek-header-row">
        <div>
          {target ? (
            <p className="previewing-badge">
              vs {target.manager_name}: {VARIANCE_VERDICT[target.posture.variance_preference]} (
              {(target.posture.p_finish_ahead * 100).toFixed(0)}% to finish ahead)
            </p>
          ) : (
            <p className="deadline">No rivals fetched for this league yet.</p>
          )}
        </div>
        <div className="header-actions">
          <button onClick={onRefresh} title="Bypass the 10-minute cache and re-fetch live">
            Refresh
          </button>
        </div>
      </div>
    </div>
  );
}
