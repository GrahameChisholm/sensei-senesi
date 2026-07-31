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
  editable: boolean;
  selected: number | null;
  onSelect: (playerId: number | null) => void;
  onSubstitute: (outId: number, inId: number) => void;
}

/** Selection lives in the parent page (not locally) so the player panel can also react to "which
 * squad member is currently selected" -- clicking a bench/XI player selects them; clicking the
 * other group's player while one is selected substitutes; clicking a panel row while a squad
 * player is selected instead triggers a transfer (handled by the parent, TeamSelection.tsx). */
export function Pitch({ teamState, directory, teams, horizon, editable, selected, onSelect, onSubstitute }: PitchProps) {
  const positionById = Object.fromEntries(teamState.squad.map((p) => [p.player_id, p.position]));
  const priceById = Object.fromEntries(teamState.squad.map((p) => [p.player_id, p.current_price]));

  const rowsByPosition: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const playerId of teamState.starting_xi) {
    rowsByPosition[positionById[playerId]]?.push(playerId);
  }

  function handleClick(playerId: number, isBench: boolean) {
    if (!editable) return;
    if (selected === null) {
      onSelect(playerId);
      return;
    }
    if (selected === playerId) {
      onSelect(null);
      return;
    }
    const selectedIsBench = teamState.bench_order.includes(selected);
    if (selectedIsBench === isBench) {
      onSelect(playerId); // same group -- just change the selection
      return;
    }
    const outId = isBench ? selected : playerId;
    const inId = isBench ? playerId : selected;
    onSubstitute(outId, inId);
    onSelect(null);
  }

  function renderCard(playerId: number, isBench: boolean) {
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
        clickable={editable}
        selected={selected === playerId}
        onClick={() => handleClick(playerId, isBench)}
      />
    );
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
        {teamState.bench_order.map((playerId) => renderCard(playerId, true))}
      </div>
    </div>
  );
}
