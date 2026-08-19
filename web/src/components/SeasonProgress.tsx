import { useState } from "react";
import { GameweekOut, SeasonLogEntryOut } from "../api";

interface SeasonProgressProps {
  gameweek: GameweekOut | null;
  seasonLog: SeasonLogEntryOut[];
  seasonComplete: boolean;
  editing: boolean;
  onAdvance: () => void;
}

/** Season Replay only: the running total, the "Advance to GW N+1" button, and a collapsible
 * gameweek-by-gameweek history. Advancing requires the draft to be confirmed first (`editing`
 * disables the button) -- otherwise the just-confirmed decisions for this gameweek would never
 * actually get scored. */
export function SeasonProgress({
  gameweek,
  seasonLog,
  seasonComplete,
  editing,
  onAdvance,
}: SeasonProgressProps) {
  const [showLog, setShowLog] = useState(false);
  const runningTotal = seasonLog.length ? seasonLog[seasonLog.length - 1].running_total : 0;

  return (
    <div className="season-progress">
      <div className="season-progress-summary">
        <span className="season-label">Season Replay — {gameweek?.season ?? "—"}</span>
        <span className="running-total">Running total: {runningTotal.toFixed(1)} pts</span>
        {seasonComplete ? (
          <span className="season-complete-badge">Season complete</span>
        ) : (
          <button
            disabled={editing}
            title={editing ? "Confirm your team before advancing" : undefined}
            onClick={onAdvance}
          >
            Advance to GW {(gameweek?.gameweek ?? 0) + 1}
          </button>
        )}
        {seasonLog.length > 0 && (
          <button className="link-button" onClick={() => setShowLog((v) => !v)}>
            {showLog ? "Hide" : "Show"} gameweek log
          </button>
        )}
      </div>

      {showLog && (
        <table className="season-log-table">
          <thead>
            <tr>
              <th>GW</th>
              <th>Points</th>
              <th>Running total</th>
              <th>Chip</th>
            </tr>
          </thead>
          <tbody>
            {seasonLog.map((entry) => (
              <tr key={entry.gameweek}>
                <td>{entry.gameweek}</td>
                <td>{entry.points.toFixed(1)}</td>
                <td>{entry.running_total.toFixed(1)}</td>
                <td>{entry.chip_played ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
