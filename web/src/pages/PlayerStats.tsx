import { useEffect, useMemo, useState } from "react";
import { PlayerCompareSelect, MAX_COMPARE_PLAYERS } from "../components/PlayerCompareSelect";
import { PlayerCompareTable } from "../components/PlayerCompareTable";
import { PlayerStatsFilters } from "../components/PlayerStatsFilters";
import { PlayerStatsTable } from "../components/PlayerStatsTable";
import { useGameweek, useTeams } from "../hooks/useProjections";
import { usePlayerStats } from "../hooks/usePlayerStats";

export function PlayerStats() {
  const [gameweek] = useGameweek();
  const teams = useTeams();

  const [search, setSearch] = useState("");
  const [selectedTeamIds, setSelectedTeamIds] = useState<Set<number>>(new Set());
  const [selectedPositions, setSelectedPositions] = useState<Set<string>>(new Set());
  const [minPrice, setMinPrice] = useState(40);
  const [maxPrice, setMaxPrice] = useState(155);
  const [perNinety, setPerNinety] = useState(false);
  const [gameweekFrom, setGameweekFrom] = useState(1);
  const [gameweekTo, setGameweekTo] = useState(1);
  const [rangeInitialized, setRangeInitialized] = useState(false);
  const [compareIds, setCompareIds] = useState<number[]>([]);

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

  function addCompare(playerId: number) {
    setCompareIds((prev) =>
      prev.includes(playerId) || prev.length >= MAX_COMPARE_PLAYERS ? prev : [...prev, playerId],
    );
  }

  function removeCompare(playerId: number) {
    setCompareIds((prev) => prev.filter((id) => id !== playerId));
  }

  const compareRows = useMemo(() => {
    const byId = new Map(rows.map((row) => [row.player_id, row]));
    return compareIds.map((id) => byId.get(id)).filter((row) => row !== undefined);
  }, [rows, compareIds]);

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
      if (normalizedSearch) {
        const teamName = row.team_id !== null ? teams[row.team_id]?.name ?? "" : "";
        const haystack = `${row.name} ${teamName}`.toLowerCase();
        if (!haystack.includes(normalizedSearch)) return false;
      }
      return true;
    });
  }, [rows, search, selectedPositions, selectedTeamIds, minPrice, maxPrice, teams]);

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
      />
      {!loading && (
        <PlayerCompareSelect
          rows={rows}
          teams={teams}
          selectedIds={compareIds}
          onAdd={addCompare}
          onRemove={removeCompare}
        />
      )}
      {compareRows.length > 0 && (
        <PlayerCompareTable
          rows={compareRows}
          teams={teams}
          perNinety={perNinety}
          ownershipStatus={ownershipStatus}
          onRemove={removeCompare}
        />
      )}
      {loading ? (
        <p>Loading…</p>
      ) : (
        <>
          <p className="stats-row-count">
            {filteredRows.length} of {rows.length} players
          </p>
          <PlayerStatsTable
            rows={filteredRows}
            teams={teams}
            perNinety={perNinety}
            ownershipStatus={ownershipStatus}
          />
        </>
      )}
    </div>
  );
}
