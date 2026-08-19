import { PlayerPanelRowOut, TeamOut, TeamStateOut } from "../api";
import { CardFixture, PlayerCard } from "./PlayerCard";

const POSITION_ORDER = ["GK", "DEF", "MID", "FWD"];

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
  /** The squad member currently marked for removal (via the card's hover "x"), if any -- a
   * purely local, not-yet-applied intent: nothing is removed from the real squad until a
   * replacement is picked in the Player Panel (a squad can never actually hold 14 players, see
   * features.team_state.MyTeamState). Rendered as an empty slot instead of that player's card. */
  removing: number | null;
  onStartRemove: (playerId: number) => void;
  onCancelRemove: () => void;
}

/** No more "edit team" mode: the pitch is always live. Hovering any squad player reveals a small
 * "x"; clicking it marks that slot empty (still just local UI state) until the Player Panel picks
 * a same-position replacement, which is what actually calls the API (TeamSelection.tsx). */
export function Pitch({ teamState, directory, teams, horizon, removing, onStartRemove, onCancelRemove }: PitchProps) {
  const positionById = Object.fromEntries(teamState.squad.map((p) => [p.player_id, p.position]));
  const priceById = Object.fromEntries(teamState.squad.map((p) => [p.player_id, p.current_price]));

  const rowsByPosition: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const playerId of teamState.starting_xi) {
    rowsByPosition[positionById[playerId]]?.push(playerId);
  }

  function renderSlot(playerId: number) {
    if (playerId === removing) {
      return (
        <button
          key={playerId}
          type="button"
          className="empty-slot"
          onClick={onCancelRemove}
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
      />
    );
  }

  return (
    <div className="pitch">
      {POSITION_ORDER.map((position) => (
        <div className="pitch-row" key={position}>
          {rowsByPosition[position].map((playerId) => renderSlot(playerId))}
        </div>
      ))}
      <div className="pitch-row bench-row">
        <div className="bench-label">Bench</div>
        {teamState.bench_order.map((playerId) => renderSlot(playerId))}
      </div>
    </div>
  );
}
