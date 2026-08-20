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
  transferOutSelected: number | null;
  /** The removed slot's position (GK/DEF/MID/FWD) -- while set, the position filter is locked to
   * it, since any other position would fail the squad's own quota check anyway. */
  fillPosition: string | null;
  /** Every player_id currently in the squad (including the one being removed) -- while a slot is
   * being filled, none of these are valid transfer-in targets (the removed player because the
   * backend hasn't actually let go of them yet, everyone else because they're already owned). */
  squadPlayerIds: number[];
  /** Bank + the selected player's own sell price (D6/D2: what a transfer could actually afford) --
   * null when nothing is selected, so "Affordable only" has nothing to filter against yet. */
  affordableBudget: number | null;
  onTransferIn: (playerId: number, position: string, price: number) => void;
  onCancel: () => void;
}

export function PlayerPanel({
  teams,
  transferOutSelected,
  fillPosition,
  squadPlayerIds,
  affordableBudget,
  onTransferIn,
  onCancel,
}: PlayerPanelProps) {
  const [position, setPosition] = useState("All");
  const [search, setSearch] = useState("");
  const [maxPrice, setMaxPrice] = useState<number | undefined>(undefined);
  const [sortKey, setSortKey] = useState<SortKey>("ep");
  const [affordableOnly, setAffordableOnly] = useState(false);

  const effectivePosition = fillPosition ?? position;

  const { rows: unsortedRows, loading } = usePlayerPanel({
    position: effectivePosition === "All" ? undefined : effectivePosition,
    search: search || undefined,
    max_price: maxPrice,
  });

  const filteredRows = useMemo(() => {
    let result = unsortedRows;
    if (transferOutSelected !== null) {
      const owned = new Set(squadPlayerIds);
      result = result.filter((row) => !owned.has(row.player_id));
    }
    if (affordableOnly && affordableBudget !== null) {
      result = result.filter((row) => row.price !== null && row.price <= affordableBudget);
    }
    return result;
  }, [unsortedRows, affordableOnly, affordableBudget, transferOutSelected, squadPlayerIds]);

  const rows = useMemo(() => sortRows(filteredRows, sortKey), [filteredRows, sortKey]);

  return (
    <div className="player-panel">
      <h3>Players</h3>

      <div className="panel-filters">
        <div className="position-tabs">
          {POSITIONS.map((p) => (
            <button
              key={p}
              className={effectivePosition === p ? "active" : ""}
              disabled={fillPosition !== null}
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
        {transferOutSelected !== null && affordableBudget !== null && (
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

      {transferOutSelected !== null && (
        <p className="transfer-hint">
          Pick a {fillPosition ?? ""} to fill the empty slot.{" "}
          <button className="link-button" onClick={onCancel}>
            Cancel
          </button>
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
            {rows.map((row) => (
              <tr
                key={row.player_id}
                className={transferOutSelected !== null ? "clickable-row" : undefined}
                onClick={
                  transferOutSelected !== null && row.price !== null
                    ? () => onTransferIn(row.player_id, row.position, row.price as number)
                    : undefined
                }
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
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
