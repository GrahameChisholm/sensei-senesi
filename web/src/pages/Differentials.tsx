import { useState } from "react";
import { DifferentialsFilters } from "../components/DifferentialsFilters";
import { DifferentialsTable } from "../components/DifferentialsTable";
import { useGameweek, useTeams } from "../hooks/useProjections";
import { useDifferentials } from "../hooks/useDifferentials";

const DEFAULT_WINDOW_GAMEWEEKS = 6;
const DEFAULT_MAX_OWNERSHIP = 10;
const DEFAULT_MAX_LEAGUE_OWNERS = 1;

export function Differentials() {
  const teams = useTeams();
  const [gameweek] = useGameweek();

  const [windowGameweeks, setWindowGameweeks] = useState(DEFAULT_WINDOW_GAMEWEEKS);
  const [maxOwnership, setMaxOwnership] = useState(DEFAULT_MAX_OWNERSHIP);
  const [maxLeagueOwners, setMaxLeagueOwners] = useState(DEFAULT_MAX_LEAGUE_OWNERS);
  const [hideOwned, setHideOwned] = useState(true);

  const {
    window: resolvedWindow,
    ownershipLens,
    picksGameweek,
    nRivals,
    rows,
    loading,
  } = useDifferentials(
    windowGameweeks,
    maxOwnership,
    maxLeagueOwners,
    undefined,
    hideOwned,
  );

  const hasPlayedGameweeks = resolvedWindow.gameweek_to >= resolvedWindow.gameweek_from;
  // MINI_LEAGUE_PLAN M31: the staleness caveat follows the data -- it only applies once the
  // league lens is actually in force, never under the global lens where it doesn't mean anything.
  const isStale =
    ownershipLens === "league" &&
    picksGameweek !== null &&
    gameweek !== null &&
    picksGameweek < gameweek.gameweek;

  return (
    <div className="player-stats">
      <h2>Differentials</h2>
      {isStale && (
        <p className="transfer-hint">
          League ownership is as of GW{picksGameweek}. GW{gameweek?.gameweek} picks go public once
          this gameweek's deadline passes.
        </p>
      )}
      <DifferentialsFilters
        windowGameweeks={windowGameweeks}
        onWindowChange={setWindowGameweeks}
        ownershipLens={ownershipLens}
        maxOwnership={maxOwnership}
        onMaxOwnershipChange={setMaxOwnership}
        maxLeagueOwners={maxLeagueOwners}
        onMaxLeagueOwnersChange={setMaxLeagueOwners}
        hideOwned={hideOwned}
        onHideOwnedChange={setHideOwned}
      />
      {loading ? (
        <p>Loading…</p>
      ) : !hasPlayedGameweeks ? (
        <div className="differentials-empty">
          Nothing has been played yet this season -- Differentials looks only at real, verified
          gameweek results (D5), so there is nothing to show until the first gameweek is played.
        </div>
      ) : (
        <>
          <p className="stats-row-count">
            Based on GW{resolvedWindow.gameweek_from} to GW{resolvedWindow.gameweek_to}, ranked by{" "}
            {ownershipLens === "league" ? "your mini-league's" : "global FPL"} ownership.{" "}
            {rows.length} player{rows.length === 1 ? "" : "s"} qualif
            {rows.length === 1 ? "ies" : "y"}.
          </p>
          <DifferentialsTable
            rows={rows}
            teams={teams}
            ownershipLens={ownershipLens}
            nRivals={nRivals}
          />
        </>
      )}
    </div>
  );
}
