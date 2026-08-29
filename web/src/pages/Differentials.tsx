import { useEffect } from "react";
import { DifferentialsFilters } from "../components/DifferentialsFilters";
import { DifferentialsTable } from "../components/DifferentialsTable";
import { useGameweek, usePlayerDirectory, useTeams } from "../hooks/useProjections";
import { useDifferentials } from "../hooks/useDifferentials";
import { useStoredState } from "../hooks/useStoredState";
import {
  DIFFERENTIAL_PRESETS,
  DifferentialIntent,
  DifferentialSortKey,
  presetSortKey,
} from "../lib/differentialPresets";

const BEST_AVAILABLE = DIFFERENTIAL_PRESETS.find((preset) => preset.key === "best_available")!;

export function Differentials() {
  const teams = useTeams();
  const [directory] = usePlayerDirectory();
  const [gameweek] = useGameweek();

  const [intent, setIntent] = useStoredState<DifferentialIntent>(
    "differentials.intent",
    "best_available",
  );
  const [windowGameweeks, setWindowGameweeksRaw] = useStoredState(
    "differentials.window",
    BEST_AVAILABLE.windowGameweeks,
  );
  const [maxOwnership, setMaxOwnershipRaw] = useStoredState(
    "differentials.maxOwnership",
    BEST_AVAILABLE.maxOwnershipPercent,
  );
  const [maxLeagueOwners, setMaxLeagueOwnersRaw] = useStoredState<number | undefined>(
    "differentials.maxLeagueOwners",
    BEST_AVAILABLE.maxLeagueOwners,
  );
  const [hideOwned, setHideOwnedRaw] = useStoredState("differentials.hideOwned", true);
  const [sortKey, setSortKey] = useStoredState<DifferentialSortKey>(
    "differentials.sortKey",
    "surplus_vs_bracket",
  );
  const [sortDescending, setSortDescending] = useStoredState("differentials.sortDescending", true);

  const {
    window: resolvedWindow,
    ownershipLens,
    picksGameweek,
    nRivals,
    rows,
    loading,
  } = useDifferentials(windowGameweeks, maxOwnership, maxLeagueOwners, undefined, hideOwned);

  // Keep a preset's sort aligned with the lens actually in force (MINI_LEAGUE_PLAN-adjacent, item
  // 3): the lens can flip from global to league the moment a configured league's first live fetch
  // succeeds, and a preset's sort should follow that switch rather than freeze at whatever it
  // resolved to on the very first render.
  useEffect(() => {
    if (intent === "custom") return;
    const preset = DIFFERENTIAL_PRESETS.find((candidate) => candidate.key === intent);
    if (!preset) return;
    const nextSortKey = presetSortKey(preset, ownershipLens);
    setSortKey((current) => (current === nextSortKey ? current : nextSortKey));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [intent, ownershipLens]);

  function handleIntentChange(nextIntent: DifferentialIntent) {
    setIntent(nextIntent);
    if (nextIntent === "custom") return;
    const preset = DIFFERENTIAL_PRESETS.find((candidate) => candidate.key === nextIntent);
    if (!preset) return;
    setWindowGameweeksRaw(preset.windowGameweeks);
    setMaxOwnershipRaw(preset.maxOwnershipPercent);
    setMaxLeagueOwnersRaw(preset.maxLeagueOwners);
    setSortKey(presetSortKey(preset, ownershipLens));
    setSortDescending(true);
  }

  function markCustom() {
    if (intent !== "custom") setIntent("custom");
  }

  function handleSort(key: DifferentialSortKey) {
    if (key === sortKey) {
      setSortDescending((prev) => !prev);
    } else {
      setSortKey(key);
      setSortDescending(true);
    }
  }

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
        intent={intent}
        onIntentChange={handleIntentChange}
        windowGameweeks={windowGameweeks}
        onWindowChange={(value) => {
          markCustom();
          setWindowGameweeksRaw(value);
        }}
        ownershipLens={ownershipLens}
        maxOwnership={maxOwnership}
        onMaxOwnershipChange={(value) => {
          markCustom();
          setMaxOwnershipRaw(value);
        }}
        maxLeagueOwners={maxLeagueOwners}
        onMaxLeagueOwnersChange={(value) => {
          markCustom();
          setMaxLeagueOwnersRaw(value);
        }}
        hideOwned={hideOwned}
        onHideOwnedChange={(value) => {
          markCustom();
          setHideOwnedRaw(value);
        }}
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
            directory={directory}
            ownershipLens={ownershipLens}
            nRivals={nRivals}
            sortKey={sortKey}
            sortDescending={sortDescending}
            onSort={handleSort}
          />
        </>
      )}
    </div>
  );
}
