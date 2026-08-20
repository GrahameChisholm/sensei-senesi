import { useMemo, useState } from "react";
import { PlayerPanelRowOut, TeamOut } from "../api";
import { usePlayerPanel } from "../hooks/useProjections";
import { expectedPointsColour, expectedPointsTextColour } from "../lib/colours";

const POSITIONS = ["All", "GK", "DEF", "MID", "FWD"];

type SortKey = "ep" | "price" | "ep_per_million";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "ep", label: "Expected points" },
  { key: "price", label: "Price" },
  { key: "ep_per_million", label: "EP per £m" },
];

function firstGameweekEp(row: PlayerPanelRowOut): number {
  return row.fixtures[0]?.expected_points ?? 0;
}

function sortRows(rows: PlayerPanelRowOut[], sortKey: SortKey): PlayerPanelRowOut[] {
  const withKey = rows.map((row) => {
    const ep = firstGameweekEp(row);
    const price = row.price ?? 0;
    const value = sortKey === "ep" ? ep : sortKey === "price" ? price : price > 0 ? ep / (price / 10) : 0;
    return { row, value };
  });
  withKey.sort((a, b) => b.value - a.value);
  return withKey.map((entry) => entry.row);
}

interface PlayerPanelProps {
  teams: Record<number, TeamOut>;
  /** True while at least one squad slot is open and waiting to be filled, either from a
   * marked-for-removal player on the live pitch, or an unfilled position while building a squad
   * from scratch. */
  fillMode: boolean;
  /** Positions with at least one open slot right now. Ignored when `fillMode` is false. A row
   * whose position isn't in this list can't be added yet (adding it wouldn't match any open
   * slot), so its Add action is disabled rather than sent to the API to fail. */
  fillablePositions: string[];
  /** Every player_id currently in the squad (including any marked for removal). While a slot is
   * being filled, none of these are valid add targets (a marked-for-removal player because the
   * backend hasn't actually let go of them yet, everyone else because they're already owned). */
  squadPlayerIds: number[];
  /** Bank plus the sell price of every player currently marked for removal, a rough ceiling for
   * the "Affordable only" filter, not a per-slot guarantee (a specific swap's own budget is
   * enforced by the API). Null when nothing is open, so the filter has nothing to check against. */
  affordableBudget: number | null;
  onAdd: (playerId: number, position: string, price: number) => void;
  /** Cancels every currently open slot at once. Omit where there's nothing to cancel (e.g.
   * building a squad from scratch, where an "empty slot" is just not-yet-picked, not a pending
   * removal). */
  onCancel?: () => void;
}

export function PlayerPanel({
  teams,
  fillMode,
  fillablePositions,
  squadPlayerIds,
  affordableBudget,
  onAdd,
  onCancel,
}: PlayerPanelProps) {
  const [position, setPosition] = useState("All");
  const [search, setSearch] = useState("");
  const [maxPrice, setMaxPrice] = useState<number | undefined>(undefined);
  const [sortKey, setSortKey] = useState<SortKey>("ep");
  const [affordableOnly, setAffordableOnly] = useState(false);

  const { rows: unsortedRows, loading } = usePlayerPanel({
    position: position === "All" ? undefined : position,
    search: search || undefined,
    max_price: maxPrice,
  });

  const filteredRows = useMemo(() => {
    let result = unsortedRows;
    if (fillMode) {
      const owned = new Set(squadPlayerIds);
      result = result.filter((row) => !owned.has(row.player_id));
    }
    if (affordableOnly && affordableBudget !== null) {
      result = result.filter((row) => row.price !== null && row.price <= affordableBudget);
    }
    return result;
  }, [unsortedRows, affordableOnly, affordableBudget, fillMode, squadPlayerIds]);

  const rows = useMemo(() => sortRows(filteredRows, sortKey), [filteredRows, sortKey]);

  return (
    <div className="player-panel">
      <h3>Players</h3>

      <div className="panel-filters">
        <div className="position-tabs">
          {POSITIONS.map((p) => (
            <button
              key={p}
              className={position === p ? "active" : ""}
              onClick={() => setPosition(p)}
            >
              {p}
            </button>
          ))}
        </div>
        <input
          type="search"
          placeholder="Search by player or team name"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <label>
          Max price
          <input
            type="range"
            min={40}
            max={155}
            step={5}
            value={maxPrice ?? 155}
            onChange={(e) => setMaxPrice(Number(e.target.value))}
          />
          £{((maxPrice ?? 155) / 10).toFixed(1)}m
        </label>
        <label>
          Sort by
          <select value={sortKey} onChange={(e) => setSortKey(e.target.value as SortKey)}>
            {SORT_OPTIONS.map((option) => (
              <option key={option.key} value={option.key}>
                {option.label}
              </option>
            ))}
          </select>
        </label>
        {fillMode && affordableBudget !== null && (
          <label>
            <input
              type="checkbox"
              checked={affordableOnly}
              onChange={(e) => setAffordableOnly(e.target.checked)}
            />
            Affordable only (£{(affordableBudget / 10).toFixed(1)}m available)
          </label>
        )}
      </div>

      {fillMode && (
        <p className="transfer-hint">
          Open slots: {fillablePositions.join(", ")}.{" "}
          {onCancel && (
            <button className="link-button" onClick={onCancel}>
              Cancel all
            </button>
          )}
        </p>
      )}

      {loading ? (
        <p>Loading…</p>
      ) : (
        <table className="panel-table">
          <thead>
            <tr>
              <th>Player</th>
              <th>Price</th>
              {rows[0]?.fixtures.map((f) => (
                <th key={f.gameweek}>GW{f.gameweek}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const fillable = fillMode && row.price !== null && fillablePositions.includes(row.position);
              return (
              <tr
                key={row.player_id}
                className={fillMode ? (fillable ? "clickable-row" : "unfillable-row") : undefined}
                onClick={fillable ? () => onAdd(row.player_id, row.position, row.price as number) : undefined}
              >
                <td>
                  {row.name}
                  {row.low_confidence && <span className="badge low-confidence-badge">!</span>}
                  <span className="panel-team">{row.team_id !== null ? teams[row.team_id]?.short_name : ""}</span>
                </td>
                <td>{row.price !== null ? `£${(row.price / 10).toFixed(1)}m` : "—"}</td>
                {row.fixtures.map((f) => (
                  <td
                    key={f.gameweek}
                    style={{ background: expectedPointsColour(f.expected_points), color: expectedPointsTextColour(f.expected_points) }}
                  >
                    <div>{f.expected_points !== null ? f.expected_points.toFixed(1) : "—"}</div>
                    <div className="fixture-label">
                      {f.opponent_id !== null && f.is_home !== null
                        ? `${teams[f.opponent_id]?.short_name ?? f.opponent_id} (${f.is_home ? "H" : "A"})`
                        : "—"}
                    </div>
                  </td>
                ))}
              </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
