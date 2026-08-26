import { GameweekOut, SquadOut, SquadPointsOut } from "../api";

interface GameweekHeaderProps {
  gameweek: GameweekOut | null;
  squad: SquadOut;
  points: SquadPointsOut | null;
  horizon: "next" | "three";
  onHorizonChange: (horizon: "next" | "three") => void;
  /** True only while a Season Replay edit is mid-flight (a draft is open, pending Confirm/Reset)
   * -- the live season never opens a draft, so this (and the buttons it gates) never shows there. */
  editing: boolean;
  onOptimise: () => void;
  onResetTeam: () => void;
  /** Discards the squad entirely and drops back to the empty-£100m build screen (POST
   * /squad/wipe) -- a sandbox reset, not a real transfer: no sell prices involved, just the
   * classic budget/quota/club-limit rules a fresh squad has to satisfy on rebuild. Immediate and
   * irreversible from here (unlike a marked-for-removal slot, there's no "Cancel all" once this
   * has been sent). */
  onWipeSquad: () => void;
}

export function GameweekHeader({
  gameweek,
  squad,
  points,
  horizon,
  onHorizonChange,
  editing,
  onOptimise,
  onResetTeam,
  onWipeSquad,
}: GameweekHeaderProps) {
  const teamState = squad.draft?.working_state ?? squad.committed;
  const transfersMade = squad.draft?.transfers_made ?? 0;

  return (
    <div className="gameweek-header">
      <div className="gameweek-title">
        <h2>Gameweek {gameweek?.gameweek ?? "—"}</h2>
        {gameweek && (
          <p className="deadline">
            Deadline: {new Date(gameweek.deadline_time).toLocaleString(undefined, { weekday: "short", day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" })}
          </p>
        )}
      </div>

      <div className="header-tiles">
        <div className="tile">
          <div className="tile-value">{points ? points.total.toFixed(1) : "—"}</div>
          <div className="tile-label">Predicted</div>
        </div>
        <div className="tile">
          <div className="tile-value">£{teamState ? (teamState.bank / 10).toFixed(1) : "0.0"}m</div>
          <div className="tile-label">In the bank</div>
        </div>
        <div className="tile">
          <div className="tile-value">{transfersMade}</div>
          <div className="tile-label">Transfers</div>
        </div>
      </div>

      <div className="horizon-toggle">
        <button className={horizon === "next" ? "active" : ""} onClick={() => onHorizonChange("next")}>
          Next GW
        </button>
        <button className={horizon === "three" ? "active" : ""} onClick={() => onHorizonChange("three")}>
          Next 3 GWs
        </button>
      </div>

      <div className="header-actions">
        {editing && (
          <button className="danger" onClick={onResetTeam}>
            Reset team
          </button>
        )}
        <button className="danger" onClick={onWipeSquad} title="Mark every player for removal so the squad can be rebuilt from scratch">
          Wipe squad
        </button>
        <button className="btn-primary" onClick={onOptimise}>
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M12 3v4M12 17v4M3 12h4M17 12h4M6 6l2.5 2.5M15.5 15.5L18 18M18 6l-2.5 2.5M8.5 15.5L6 18" />
          </svg>
          Optimise lineup
        </button>
      </div>
    </div>
  );
}
