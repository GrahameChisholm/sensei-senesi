import { useState } from "react";
import { DifferentialsFilters } from "../components/DifferentialsFilters";
import { DifferentialsTable } from "../components/DifferentialsTable";
import { useTeams } from "../hooks/useProjections";
import { useDifferentials } from "../hooks/useDifferentials";

const DEFAULT_WINDOW_GAMEWEEKS = 6;
const DEFAULT_MAX_OWNERSHIP = 10;

export function Differentials() {
  const teams = useTeams();

  const [windowGameweeks, setWindowGameweeks] = useState(DEFAULT_WINDOW_GAMEWEEKS);
  const [maxOwnership, setMaxOwnership] = useState(DEFAULT_MAX_OWNERSHIP);
  const [hideOwned, setHideOwned] = useState(true);

  const { window: resolvedWindow, rows, loading } = useDifferentials(
    windowGameweeks,
    maxOwnership,
    hideOwned,
  );

  const hasPlayedGameweeks = resolvedWindow.gameweek_to >= resolvedWindow.gameweek_from;

  return (
    <div className="player-stats">
      <h2>Differentials</h2>
      <DifferentialsFilters
        windowGameweeks={windowGameweeks}
        onWindowChange={setWindowGameweeks}
        maxOwnership={maxOwnership}
        onMaxOwnershipChange={setMaxOwnership}
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
            Based on GW{resolvedWindow.gameweek_from} to GW{resolvedWindow.gameweek_to}. {rows.length}{" "}
            player{rows.length === 1 ? "" : "s"} qualif{rows.length === 1 ? "ies" : "y"}.
          </p>
          <DifferentialsTable rows={rows} teams={teams} />
        </>
      )}
    </div>
  );
}
