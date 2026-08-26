import { PlayerPanelRowOut, SquadOut, TeamOut } from "../api";
import { CardFixture, PlayerCard } from "./PlayerCard";

export const POSITION_ORDER = ["GK", "DEF", "MID", "FWD"];
export const QUOTA: Record<string, number> = { GK: 2, DEF: 5, MID: 5, FWD: 3 };

export function cardFixtures(
  row: PlayerPanelRowOut | undefined,
  teams: Record<number, TeamOut>,
): CardFixture[] {
  if (!row) return [];
  return row.fixtures.map((fixture) => ({
    gameweek: fixture.gameweek,
    opponentShortName:
      fixture.opponent_id !== null ? teams[fixture.opponent_id]?.short_name ?? null : null,
    isHome: fixture.is_home,
    expectedPoints: fixture.expected_points,
  }));
}

interface PitchProps {
  squad: SquadOut;
  directory: Record<number, PlayerPanelRowOut>;
  teams: Record<number, TeamOut>;
  horizon: "next" | "three";
  onRemove: (playerId: number) => void;
  onSetCaptain: (playerId: number, role: "captain" | "vice") => void;
}

/** One pitch for every squad size from empty to a complete 15. Once a starting XI/bench split
 * exists (15/15 reached and auto-arranged), it renders the classic formation shape: one row per
 * position for the starting XI, plus a separate bench strip, matching real FPL. Before that, it
 * falls back to one row per position with an empty "pick a {position}" placeholder per still-open
 * slot, since there's no arrangement yet to render a formation from. Removing a squad player is
 * instant (no draft, no confirm) -- the vacated slot just renders empty on the next render.
 * Captain/vice pills only ever show on starting-XI cards, since only they're eligible for the
 * armband. */
export function Pitch({ squad, directory, teams, horizon, onRemove, onSetCaptain }: PitchProps) {
  const byId = Object.fromEntries(squad.squad.map((p) => [p.player_id, p]));

  function renderCard(playerId: number, isBench: boolean) {
    const pick = byId[playerId];
    const row = directory[playerId];
    return (
      <PlayerCard
        key={playerId}
        playerId={playerId}
        name={row?.name ?? `#${playerId}`}
        price={pick?.price ?? row?.price ?? null}
        fixtures={cardFixtures(row, teams)}
        horizon={horizon}
        isCaptain={squad.captain_id === playerId}
        isVice={squad.vice_captain_id === playerId}
        lowConfidence={row?.low_confidence}
        onRemove={() => onRemove(playerId)}
        onCaptain={isBench ? undefined : () => onSetCaptain(playerId, "captain")}
        onVice={isBench ? undefined : () => onSetCaptain(playerId, "vice")}
      />
    );
  }

  if (squad.starting_xi.length > 0) {
    const positionById = Object.fromEntries(squad.squad.map((p) => [p.player_id, p.position]));
    const rowsByPosition: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
    for (const playerId of squad.starting_xi) {
      rowsByPosition[positionById[playerId]]?.push(playerId);
    }
    return (
      <div className="pitch">
        {POSITION_ORDER.map((position) => (
          <div className="pitch-row" key={position}>
            {rowsByPosition[position].map((playerId) => renderCard(playerId, false))}
          </div>
        ))}
        <div className="pitch-row bench-row">
          <div className="bench-label">Bench</div>
          {squad.bench_order.map((playerId) => renderCard(playerId, true))}
        </div>
      </div>
    );
  }

  const byPosition: Record<string, typeof squad.squad> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const pick of squad.squad) byPosition[pick.position]?.push(pick);

  return (
    <div className="pitch">
      {POSITION_ORDER.map((position) => {
        const picks = byPosition[position];
        const emptyCount = Math.max(QUOTA[position] - picks.length, 0);
        return (
          <div className="pitch-row" key={position}>
            {picks.map((pick) => renderCard(pick.player_id, false))}
            {Array.from({ length: emptyCount }, (_, index) => (
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
