import { PlayerPanelRowOut, TeamOut, TeamStateOut } from "../api";
import { CardFixture, PlayerCard } from "./PlayerCard";

export const POSITION_ORDER = ["GK", "DEF", "MID", "FWD"];

export function cardFixtures(row: PlayerPanelRowOut | undefined, teams: Record<number, TeamOut>): CardFixture[] {
  if (!row) return [];
  return row.fixtures.map((fixture) => ({
    gameweek: fixture.gameweek,
    opponentShortName: fixture.opponent_id !== null ? teams[fixture.opponent_id]?.short_name ?? null : null,
    isHome: fixture.is_home,
    expectedPoints: fixture.expected_points,
  }));
}

interface PitchProps {
  teamState: TeamStateOut;
  directory: Record<number, PlayerPanelRowOut>;
  teams: Record<number, TeamOut>;
  horizon: "next" | "three";
  /** Every squad member currently marked for removal (via a card's hover "x"), if any, a
   * purely local, not-yet-applied intent: nothing is removed from the real squad until a
   * replacement is picked in the Player Panel (a squad can never actually hold fewer than 15
   * players, see features.team_state.MyTeamState). Any number of players can be marked at once,
   * each rendered as an empty slot instead of that player's card, and filled independently. */
  removingIds: number[];
  onStartRemove: (playerId: number) => void;
  onCancelRemove: (playerId: number) => void;
  onSetCaptain: (playerId: number, role: "captain" | "vice") => void;
}

/** No more "edit team" mode: the pitch is always live. Hovering any squad player reveals a small
 * "x"; clicking it marks that slot empty (still just local UI state) until the Player Panel picks
 * a same-position replacement, which is what actually calls the API (TeamSelection.tsx). Any
 * number of slots can be emptied at once, so a manager can clear their whole squad and rebuild it
 * from the panel. Hovering a starting-XI player also reveals "C"/"VC" pills; captain/vice must
 * be in the XI, so bench cards never get them. */
export function Pitch({
  teamState,
  directory,
  teams,
  horizon,
  removingIds,
  onStartRemove,
  onCancelRemove,
  onSetCaptain,
}: PitchProps) {
  const positionById = Object.fromEntries(teamState.squad.map((p) => [p.player_id, p.position]));
  const priceById = Object.fromEntries(teamState.squad.map((p) => [p.player_id, p.current_price]));

  const rowsByPosition: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const playerId of teamState.starting_xi) {
    rowsByPosition[positionById[playerId]]?.push(playerId);
  }

  function renderSlot(playerId: number, isBench: boolean) {
    if (removingIds.includes(playerId)) {
      return (
        <button
          key={playerId}
          type="button"
          className="empty-slot"
          onClick={() => onCancelRemove(playerId)}
          title="Cancel"
        >
          <span>{positionById[playerId]}</span>
          <span className="empty-slot-hint">Pick a replacement, or click to cancel</span>
        </button>
      );
    }
    const row = directory[playerId];
    return (
      <PlayerCard
        key={playerId}
        playerId={playerId}
        name={row?.name ?? `#${playerId}`}
        price={priceById[playerId] ?? row?.price ?? null}
        fixtures={cardFixtures(row, teams)}
        horizon={horizon}
        isCaptain={teamState.captain_id === playerId}
        isVice={teamState.vice_captain_id === playerId}
        lowConfidence={row?.low_confidence}
        onRemove={() => onStartRemove(playerId)}
        onCaptain={isBench ? undefined : () => onSetCaptain(playerId, "captain")}
        onVice={isBench ? undefined : () => onSetCaptain(playerId, "vice")}
      />
    );
  }

  return (
    <div className="pitch">
      {POSITION_ORDER.map((position) => (
        <div className="pitch-row" key={position}>
          {rowsByPosition[position].map((playerId) => renderSlot(playerId, false))}
        </div>
      ))}
      <div className="pitch-row bench-row">
        <div className="bench-label">Bench</div>
        {teamState.bench_order.map((playerId) => renderSlot(playerId, true))}
      </div>
    </div>
  );
}
