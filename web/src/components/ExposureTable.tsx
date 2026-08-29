import { useMemo } from "react";
import { PlayerExposureOut, PlayerPanelRowOut, TeamOut } from "../api";
import { useStoredState } from "../hooks/useStoredState";
import { exposureColour, exposureTextColour } from "../lib/colours";

type SortKey = "name" | "your_multiplier" | "eo_multiplier" | "exposure" | "expected_points" | "expected_swing";

function sortValue(row: PlayerExposureOut, directory: Record<number, PlayerPanelRowOut>, key: SortKey) {
  if (key === "name") return (directory[row.player_id]?.name ?? "").toLowerCase();
  if (key === "eo_multiplier") return row.ownership.eo_multiplier;
  const value = row[key];
  return value ?? Number.NEGATIVE_INFINITY;
}

function sortRows(
  rows: PlayerExposureOut[],
  directory: Record<number, PlayerPanelRowOut>,
  sortKey: SortKey,
  sortDescending: boolean,
): PlayerExposureOut[] {
  const withValue = rows.map((row) => ({ row, value: sortValue(row, directory, sortKey) }));
  withValue.sort((a, b) => {
    if (typeof a.value === "string" || typeof b.value === "string") {
      return String(a.value).localeCompare(String(b.value)) * (sortDescending ? -1 : 1);
    }
    return (a.value - b.value) * (sortDescending ? -1 : 1);
  });
  return withValue.map((entry) => entry.row);
}

interface ExposureTableProps {
  rows: PlayerExposureOut[];
  directory: Record<number, PlayerPanelRowOut>;
  teams: Record<number, TeamOut>;
  ownedPlayerIds: Set<number>;
}

/** The Mini League page's centrepiece (MINI_LEAGUE_PLAN M7/M21 zone 3), split into "Your edges"
 * and "Your holes" (item 4): an edge and a hole call for a different action, so rather than one
 * long list a manager has to scan to find the boundary, positive-swing rows sort into their own
 * table and negative-swing rows into another. A row whose swing is exactly zero or unknown is
 * omitted from both -- it carries no signal either way. The "your picks | all" toggle collapses
 * both back to the 15 when the full league-wide list is too long to scan. Sort/toggle state is
 * persisted per browser view (item 7), since it's a durable reading preference, not per-request
 * filter state. */
export function ExposureTable({ rows, directory, teams, ownedPlayerIds }: ExposureTableProps) {
  const [sortKey, setSortKey] = useStoredState<SortKey>(
    "mini-league.exposure.sortKey",
    "expected_swing",
  );
  const [sortDescending, setSortDescending] = useStoredState(
    "mini-league.exposure.sortDescending",
    true,
  );
  const [ownedOnly, setOwnedOnly] = useStoredState("mini-league.exposure.ownedOnly", false);

  const visibleRows = ownedOnly ? rows.filter((row) => ownedPlayerIds.has(row.player_id)) : rows;

  const edges = useMemo(
    () => visibleRows.filter((row) => row.expected_swing !== null && row.expected_swing > 0),
    [visibleRows],
  );
  const holes = useMemo(
    () => visibleRows.filter((row) => row.expected_swing !== null && row.expected_swing < 0),
    [visibleRows],
  );

  const sortedEdges = useMemo(
    () => sortRows(edges, directory, sortKey, sortDescending),
    [edges, directory, sortKey, sortDescending],
  );
  const sortedHoles = useMemo(
    () => sortRows(holes, directory, sortKey, sortDescending),
    [holes, directory, sortKey, sortDescending],
  );

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDescending((prev) => !prev);
    } else {
      setSortKey(key);
      setSortDescending(true);
    }
  }

  function sortIndicator(key: SortKey): string {
    if (key !== sortKey) return "";
    return sortDescending ? " ▼" : " ▲";
  }

  function renderSection(title: string, sectionRows: PlayerExposureOut[]) {
    return (
      <div style={{ marginBottom: "0.85rem" }}>
        <p className="stats-filter-label" style={{ marginBottom: "0.35rem" }}>
          {title} ({sectionRows.length})
        </p>
        {sectionRows.length === 0 ? (
          <p className="stats-row-count">Nothing here right now.</p>
        ) : (
          <div className="stats-table-scroll">
            <table className="stats-table">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => toggleSort("name")} title="Player name and club">
                    Player{sortIndicator("name")}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("your_multiplier")}
                    title="Your own points multiplier for this player: 0 benched, 1 started, 2 captained, 3 triple captained"
                  >
                    You{sortIndicator("your_multiplier")}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("eo_multiplier")}
                    title="League effective ownership: the average points multiplier for this player across every rival in your league"
                  >
                    League EO{sortIndicator("eo_multiplier")}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("exposure")}
                    title="Your multiplier minus the league's effective ownership. Positive means you're more exposed to this player than the field, negative means you're exposed to him working against you."
                  >
                    Exposure{sortIndicator("exposure")}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("expected_points")}
                    title="Engine projected points for this gameweek"
                  >
                    xP{sortIndicator("expected_points")}
                  </th>
                  <th
                    className="sortable"
                    onClick={() => toggleSort("expected_swing")}
                    title="Exposure times expected points: how much this player is expected to move your rank relative to the field this gameweek"
                  >
                    Swing{sortIndicator("expected_swing")}
                  </th>
                </tr>
              </thead>
              <tbody>
                {sectionRows.map((row) => {
                  const player = directory[row.player_id];
                  return (
                    <tr key={row.player_id}>
                      <td>
                        {player?.name ?? `#${row.player_id}`}
                        <span className="panel-team">
                          {player?.team_id !== null && player?.team_id !== undefined
                            ? teams[player.team_id]?.short_name
                            : ""}
                        </span>
                      </td>
                      <td>{row.your_multiplier.toFixed(0)}</td>
                      <td title={row.ownership.owner_names.join(", ")}>
                        {row.ownership.eo_multiplier.toFixed(2)}
                      </td>
                      <td
                        style={{
                          background: exposureColour(row.exposure),
                          color: exposureTextColour(row.exposure),
                        }}
                      >
                        {row.exposure >= 0 ? "+" : ""}
                        {row.exposure.toFixed(2)}
                      </td>
                      <td>{row.expected_points !== null ? row.expected_points.toFixed(1) : "—"}</td>
                      <td
                        style={{
                          background: exposureColour(row.expected_swing),
                          color: exposureTextColour(row.expected_swing),
                          fontWeight: 700,
                        }}
                      >
                        {row.expected_swing !== null
                          ? `${row.expected_swing >= 0 ? "+" : ""}${row.expected_swing.toFixed(1)}`
                          : "—"}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="fixture-swing">
      <div className="stats-filters-row" style={{ marginBottom: "0.5rem" }}>
        <p className="stats-filter-label">Exposure</p>
        <div className="stats-team-picker">
          <button className={!ownedOnly ? "active" : ""} onClick={() => setOwnedOnly(false)}>
            All
          </button>
          <button className={ownedOnly ? "active" : ""} onClick={() => setOwnedOnly(true)}>
            Your picks
          </button>
        </div>
      </div>
      {renderSection("Your edges", sortedEdges)}
      {renderSection("Your holes", sortedHoles)}
    </div>
  );
}
