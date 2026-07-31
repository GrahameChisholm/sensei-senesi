import { GameweekOut, SquadOut, SquadPointsOut } from "../api";

interface GameweekHeaderProps {
  gameweek: GameweekOut | null;
  squad: SquadOut;
  points: SquadPointsOut | null;
  horizon: "next" | "three";
  onHorizonChange: (horizon: "next" | "three") => void;
  editing: boolean;
  onEditTeam: () => void;
  onOptimise: () => void;
  onResetTeam: () => void;
}

export function GameweekHeader({
  gameweek,
  squad,
  points,
  horizon,
  onHorizonChange,
  editing,
  onEditTeam,
  onOptimise,
  onResetTeam,
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
        {!editing ? (
          <button onClick={onEditTeam}>Edit team</button>
        ) : (
          <button className="danger" onClick={onResetTeam}>
            Reset team
          </button>
        )}
        <button onClick={onOptimise}>Optimise lineup</button>
      </div>
    </div>
  );
}
