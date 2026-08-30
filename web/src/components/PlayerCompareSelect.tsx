import { useMemo, useState } from "react";
import { PlayerStatsRowOut, TeamOut } from "../api";

export const MAX_COMPARE_PLAYERS = 6;
const MAX_SUGGESTIONS = 8;

interface PlayerCompareSelectProps {
  rows: PlayerStatsRowOut[];
  teams: Record<number, TeamOut>;
  selectedIds: number[];
  onAdd: (playerId: number) => void;
  onRemove: (playerId: number) => void;
}

/** A typeahead multiselect for building an ad hoc comparison set (up to
 * {@link MAX_COMPARE_PLAYERS}): type to search the page's full player list regardless of the
 * active filters, click a suggestion to add it as a removable chip. Selection order is
 * preserved -- it becomes column order in PlayerCompareTable. */
export function PlayerCompareSelect({
  rows,
  teams,
  selectedIds,
  onAdd,
  onRemove,
}: PlayerCompareSelectProps) {
  const [query, setQuery] = useState("");

  const rowById = useMemo(() => new Map(rows.map((row) => [row.player_id, row])), [rows]);
  const atLimit = selectedIds.length >= MAX_COMPARE_PLAYERS;

  const suggestions = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    if (!normalized || atLimit) return [];
    const selected = new Set(selectedIds);
    const matches: PlayerStatsRowOut[] = [];
    for (const row of rows) {
      if (selected.has(row.player_id)) continue;
      const teamName = row.team_id !== null ? teams[row.team_id]?.name ?? "" : "";
      const haystack = `${row.name} ${teamName}`.toLowerCase();
      if (haystack.includes(normalized)) {
        matches.push(row);
        if (matches.length >= MAX_SUGGESTIONS) break;
      }
    }
    return matches;
  }, [query, rows, teams, selectedIds, atLimit]);

  function handleAdd(playerId: number) {
    onAdd(playerId);
    setQuery("");
  }

  return (
    <div className="stats-filter-group compare-select">
      <span className="stats-filter-label">
        Compare{selectedIds.length > 0 ? ` (${selectedIds.length}/${MAX_COMPARE_PLAYERS})` : ""}
      </span>
      <div className="compare-select-body">
        {selectedIds.length > 0 && (
          <div className="compare-chip-list">
            {selectedIds.map((playerId) => {
              const row = rowById.get(playerId);
              return (
                <span key={playerId} className="compare-chip">
                  {row?.name ?? `#${playerId}`}
                  <button
                    type="button"
                    aria-label={`Remove ${row?.name ?? "player"} from comparison`}
                    onClick={() => onRemove(playerId)}
                  >
                    ×
                  </button>
                </span>
              );
            })}
          </div>
        )}
        <div className="compare-select-input">
          <input
            type="search"
            placeholder={atLimit ? `Up to ${MAX_COMPARE_PLAYERS} players` : "Add a player to compare"}
            value={query}
            disabled={atLimit}
            onChange={(e) => setQuery(e.target.value)}
          />
          {suggestions.length > 0 && (
            <ul className="compare-suggestions">
              {suggestions.map((row) => (
                <li key={row.player_id}>
                  <button type="button" onClick={() => handleAdd(row.player_id)}>
                    {row.name}
                    <span className="panel-team">
                      {row.team_id !== null ? teams[row.team_id]?.short_name : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
}
