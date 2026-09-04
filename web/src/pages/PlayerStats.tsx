import { useEffect, useMemo, useState } from "react";
import { PlayerStatsFilters } from "../components/PlayerStatsFilters";
import { PlayerStatsTable } from "../components/PlayerStatsTable";
import { useGameweek, useTeams } from "../hooks/useProjections";
import { usePlayerStats } from "../hooks/usePlayerStats";
import { isDefinitelyUnavailable } from "../lib/playerStats";

export function PlayerStats() {
  const [gameweek] = useGameweek();
  const teams = useTeams();

  const [search, setSearch] = useState("");
  const [selectedTeamIds, setSelectedTeamIds] = useState<Set<number>>(new Set());
  const [selectedPositions, setSelectedPositions] = useState<Set<string>>(new Set());
  const [minPrice, setMinPrice] = useState(40);
  const [maxPrice, setMaxPrice] = useState(155);
  const [perNinety, setPerNinety] = useState(false);
  const [hideUnavailable, setHideUnavailable] = useState(false);
  const [gameweekFrom, setGameweekFrom] = useState(1);
  const [gameweekTo, setGameweekTo] = useState(1);
  const [rangeInitialized, setRangeInitialized] = useState(false);
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [comparing, setComparing] = useState(false);

  useEffect(() => {
    if (gameweek !== null && !rangeInitialized) {
      // The cache's own gameweek, not the decision one: this page reads actual performance out of
      // the same snapshot the cache was built from, so it knows nothing about any gameweek played
      // since, and following the decision gameweek would only widen the range over empty rows.
      setGameweekTo(gameweek.projections_gameweek);
      setRangeInitialized(true);
    }
  }, [gameweek, rangeInitialized]);

  const { rows, ownershipStatus, loading } = usePlayerStats(gameweekFrom, gameweekTo);

  function toggleSelected(playerId: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(playerId)) next.delete(playerId);
      else next.add(playerId);
      return next;
    });
  }

  function resetComparison() {
    setComparing(false);
    setSelectedIds(new Set());
  }

  function toggleTeam(teamId: number) {
    setSelectedTeamIds((prev) => {
      const next = new Set(prev);
      if (next.has(teamId)) next.delete(teamId);
      else next.add(teamId);
      return next;
    });
  }

  function togglePosition(position: string) {
    setSelectedPositions((prev) => {
      const next = new Set(prev);
      if (next.has(position)) next.delete(position);
      else next.add(position);
      return next;
    });
  }

  const filteredRows = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return rows.filter((row) => {
      if (selectedPositions.size > 0 && !selectedPositions.has(row.position)) return false;
      if (selectedTeamIds.size > 0 && (row.team_id === null || !selectedTeamIds.has(row.team_id))) {
        return false;
      }
      if (row.price !== null && (row.price < minPrice || row.price > maxPrice)) return false;
      if (hideUnavailable && isDefinitelyUnavailable(row.availability)) return false;
      if (normalizedSearch) {
        const teamName = row.team_id !== null ? teams[row.team_id]?.name ?? "" : "";
        const haystack = `${row.name} ${teamName}`.toLowerCase();
        if (!haystack.includes(normalizedSearch)) return false;
      }
      return true;
    });
  }, [rows, search, selectedPositions, selectedTeamIds, minPrice, maxPrice, hideUnavailable, teams]);

  // Comparing shows exactly the checked players, off the full unfiltered pool -- a player checked
  // under one search/team/position filter should still show up in the comparison even after the
  // filter changes, rather than the comparison quietly shrinking to whatever's currently visible.
  const displayedRows = useMemo(() => {
    if (!comparing) return filteredRows;
    return rows.filter((row) => selectedIds.has(row.player_id));
  }, [comparing, filteredRows, rows, selectedIds]);

  return (
    <div className="player-stats">
      <h2>Player Stats</h2>
      <PlayerStatsFilters
        teams={teams}
        search={search}
        onSearchChange={setSearch}
        selectedTeamIds={selectedTeamIds}
        onToggleTeam={toggleTeam}
        selectedPositions={selectedPositions}
        onTogglePosition={togglePosition}
        minPrice={minPrice}
        maxPrice={maxPrice}
        onPriceChange={(min, max) => {
          setMinPrice(min);
          setMaxPrice(max);
        }}
        gameweekFrom={gameweekFrom}
        gameweekTo={gameweekTo}
        maxGameweek={gameweek?.gameweek ?? 1}
        onGameweekRangeChange={(from, to) => {
          setGameweekFrom(from);
          setGameweekTo(to);
        }}
        perNinety={perNinety}
        onPerNinetyChange={setPerNinety}
        hideUnavailable={hideUnavailable}
        onHideUnavailableChange={setHideUnavailable}
      />
      {loading ? (
        <p>Loading…</p>
      ) : (
        <>
          <div className="compare-toggle-row">
            <p className="stats-row-count">
              {comparing
                ? `Comparing ${displayedRows.length} player${displayedRows.length === 1 ? "" : "s"}`
                : `${filteredRows.length} of ${rows.length} players`}
            </p>
            {comparing ? (
              <button type="button" className="btn-primary" onClick={resetComparison}>
                Reset
              </button>
            ) : (
              <button
                type="button"
                className="btn-primary"
                disabled={selectedIds.size === 0}
                onClick={() => setComparing(true)}
              >
                Compare ({selectedIds.size})
              </button>
            )}
          </div>
          <PlayerStatsTable
            rows={displayedRows}
            teams={teams}
            perNinety={perNinety}
            ownershipStatus={ownershipStatus}
            selectedIds={selectedIds}
            onToggleSelected={toggleSelected}
            horizonGameweeks={gameweek?.horizon_gameweeks ?? []}
          />
        </>
      )}
    </div>
  );
}
