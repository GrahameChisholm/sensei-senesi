import { PlayerPanelRowOut, SquadPlayerOut, TeamOut } from "../api";
import { QUOTA } from "../lib/squadBuild";
import { cardFixtures, POSITION_ORDER } from "./Pitch";
import { PlayerCard } from "./PlayerCard";

interface BuildPitchProps {
  picks: SquadPlayerOut[];
  directory: Record<number, PlayerPanelRowOut>;
  teams: Record<number, TeamOut>;
  horizon: "next" | "three";
  onRemove: (playerId: number) => void;
}

/** The same pitch layout as the live squad view, but for the 0-15 players picked before a squad
 * is complete: no starting XI/bench split yet (that's decided once the squad is confirmed), just
 * one row per position with a card for each pick and an empty slot for each spot still open. */
export function BuildPitch({ picks, directory, teams, horizon, onRemove }: BuildPitchProps) {
  const byPosition: Record<string, SquadPlayerOut[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const pick of picks) byPosition[pick.position]?.push(pick);

  return (
    <div className="pitch">
      {POSITION_ORDER.map((position) => {
        const filled = byPosition[position];
        const emptyCount = QUOTA[position] - filled.length;
        return (
          <div className="pitch-row" key={position}>
            {filled.map((pick) => {
              const row = directory[pick.player_id];
              return (
                <PlayerCard
                  key={pick.player_id}
                  playerId={pick.player_id}
                  name={row?.name ?? `#${pick.player_id}`}
                  price={pick.current_price}
                  fixtures={cardFixtures(row, teams)}
                  horizon={horizon}
                  lowConfidence={row?.low_confidence}
                  onRemove={() => onRemove(pick.player_id)}
                />
              );
            })}
            {Array.from({ length: Math.max(emptyCount, 0) }, (_, index) => (
              <div key={`empty-${position}-${index}`} className="empty-slot">
                <span>{position}</span>
                <span className="empty-slot-hint">Pick a {position} from the panel</span>
              </div>
            ))}
          </div>
        );
      })}
    </div>
  );
}
