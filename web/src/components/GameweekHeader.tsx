import { GameweekOut, SquadOut, SquadPointsOut } from "../api";
import { GameweekSelector } from "./GameweekSelector";
import { ImportTeamForm } from "./ImportTeamForm";

interface GameweekHeaderProps {
  gameweek: GameweekOut | null;
  squad: SquadOut;
  points: SquadPointsOut | null;
  horizon: "next" | "three";
  onHorizonChange: (horizon: "next" | "three") => void;
  /** The future gameweek currently being previewed, or null when showing the real current
   * gameweek, shared with the Fixtures page so the two stay in sync. */
  viewGameweek: number | null;
  onViewGameweekChange: (gameweek: number | null) => void;
  onAutoBuild: () => void;
  /** Empties the squad and resets the personal budget ceiling back to the classic £100m -- a
   * sandbox reset, not a real transfer: no sell prices involved, just the classic budget/quota/
   * club-limit rules a fresh squad has to satisfy on rebuild. */
  onClearSquad: () => void;
  /** Re-syncs the squad from a real FPL Team ID, overwriting whatever squad state currently
   * exists -- usable any time, not just once at onboarding, since a manager's real team also
   * changes over the season via their own transfers in the real FPL app. */
  onImportSquad: (teamId: number) => void;
}

export function GameweekHeader({
  gameweek,
  squad,
  points,
  horizon,
  onHorizonChange,
  viewGameweek,
  onViewGameweekChange,
  onAutoBuild,
  onClearSquad,
  onImportSquad,
}: GameweekHeaderProps) {
  const isPreviewing = viewGameweek !== null;
  // The server advanced the decision gameweek past the one the cache was built for, which it does
  // on its own once a deadline passes. Worth saying out loud: the projections still work, they
  // were just fit before that deadline, and a rebuild produces fresher ones.
  const projectionsAreStale =
    gameweek !== null && gameweek.projections_gameweek !== gameweek.gameweek;

  return (
    <div className="gameweek-header">
      <div className="gameweek-header-row">
        <div className="gameweek-title">
          <h2>Gameweek {gameweek?.gameweek ?? "—"}</h2>
          {isPreviewing && <p className="previewing-badge">Previewing GW{viewGameweek}</p>}
          {gameweek && !isPreviewing && !gameweek.deadline_passed && (
            <p className="deadline">
              Deadline:{" "}
              {new Date(gameweek.deadline_time).toLocaleString(undefined, {
                weekday: "short",
                day: "numeric",
                month: "short",
                hour: "2-digit",
                minute: "2-digit",
              })}
            </p>
          )}
          {gameweek && !isPreviewing && gameweek.deadline_passed && (
            <p
              className="deadline-passed"
              title="Every gameweek this cache projects has already kicked off, so nothing here can still be changed. Rebuild projections to plan the next one."
            >
              Deadline passed, nothing here can be changed
            </p>
          )}
          {gameweek && !isPreviewing && projectionsAreStale && (
            <p
              className="stale-projections"
              title={`These projections were built for GW${gameweek.projections_gameweek}, before the last deadline. Rebuild them for numbers fit on everything since.`}
            >
              Projections built for GW{gameweek.projections_gameweek}
            </p>
          )}
        </div>

        {gameweek && gameweek.horizon_gameweeks.length > 1 && (
          <GameweekSelector
            gameweeks={gameweek.horizon_gameweeks}
            currentGameweek={gameweek.gameweek}
            selected={viewGameweek}
            onSelect={onViewGameweekChange}
          />
        )}

        <div className="header-tiles">
          <div className="tile">
            <div className="tile-value">{points ? points.total.toFixed(1) : "—"}</div>
            <div className="tile-label">
              {isPreviewing ? `Predicted (GW${viewGameweek})` : "Predicted"}
            </div>
          </div>
          <div className="tile">
            <div className="tile-value">£{(squad.budget_remaining / 10).toFixed(1)}m</div>
            <div className="tile-label">
              Budget remaining
              {squad.budget_ceiling !== 1000 &&
                ` (of £${(squad.budget_ceiling / 10).toFixed(1)}m)`}
            </div>
          </div>
        </div>
      </div>

      <div className="gameweek-header-row">
        <div className="horizon-toggle" style={{ visibility: isPreviewing ? "hidden" : "visible" }}>
          <button
            className={horizon === "next" ? "active" : ""}
            onClick={() => onHorizonChange("next")}
          >
            Next GW
          </button>
          <button
            className={horizon === "three" ? "active" : ""}
            onClick={() => onHorizonChange("three")}
          >
            Next 3 GWs
          </button>
        </div>

        <div className="header-actions">
          <button className="danger" onClick={onClearSquad} title="Clear the squad and start over with a fresh £100m budget">
            Clear squad
          </button>
          <ImportTeamForm
            onImport={onImportSquad}
            confirmMessage="Import your real FPL team? This replaces your current squad. This can't be undone."
          />
          <button className="btn-primary" onClick={onAutoBuild}>
            <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18" />
            </svg>
            Auto-build best squad
          </button>
        </div>
      </div>
    </div>
  );
}
