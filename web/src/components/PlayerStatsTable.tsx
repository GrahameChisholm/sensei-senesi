import { useMemo, useRef, useState } from "react";
import { ComponentBreakdownOut, PlayerStatsRowOut, TeamOut } from "../api";
import { expectedPointsColour, expectedPointsTextColour } from "../lib/colours";
import {
  STAT_COLUMNS,
  StatColumn,
  StatKey,
  averageMinutesPerMatch,
  formatStat,
  horizonPoints,
  statValue,
} from "../lib/playerStats";
import { BreakdownPopover } from "./BreakdownPopover";

const ROW_HEIGHT = 44;
const VISIBLE_ROWS = 14;
const CONTAINER_HEIGHT = ROW_HEIGHT * VISIBLE_ROWS;
const OVERSCAN = 4;

const ACTUAL_COMPONENT_LABELS: Record<string, string> = {
  appearance: "Appearance",
  goals: "Goals",
  assists: "Assists",
  clean_sheet: "Clean sheet",
  goals_conceded: "Goals conceded",
  defensive_contribution: "Defensive contribution",
  saves: "Saves",
  bonus: "Bonus",
  cards: "Cards",
  penalty_misses: "Penalty misses",
  own_goals: "Own goals",
};

function ActualBreakdownPopover({ breakdown }: { breakdown: ComponentBreakdownOut }) {
  const lines = Object.entries(breakdown).filter(
    ([key, value]) => key !== "total" && Math.abs(value) > 0.001,
  );
  return (
    <div className="breakdown-popover">
      <table>
        <tbody>
          {lines.map(([key, value]) => (
            <tr key={key}>
              <td>{ACTUAL_COMPONENT_LABELS[key] ?? key}</td>
              <td className={value < 0 ? "negative" : undefined}>{value.toFixed(1)}</td>
            </tr>
          ))}
          <tr className="total-row">
            <td>Total (actual)</td>
            <td>{breakdown.total.toFixed(1)}</td>
          </tr>
        </tbody>
      </table>
    </div>
  );
}

type SortKey = "name" | "price" | "apps" | "minutes" | "selected_by_percent" | "horizon_points" | StatKey;

function sortValue(row: PlayerStatsRowOut, sortKey: SortKey, perNinety: boolean): number | string {
  if (sortKey === "name") return row.name.toLowerCase();
  if (sortKey === "price") return row.price ?? 0;
  if (sortKey === "apps") return row.actuals.apps;
  if (sortKey === "minutes") return averageMinutesPerMatch(row);
  if (sortKey === "selected_by_percent") return row.actuals.selected_by_percent ?? 0;
  if (sortKey === "horizon_points") return horizonPoints(row);
  const column = STAT_COLUMNS.find((c) => c.key === sortKey);
  return column ? statValue(row, column, perNinety) : 0;
}

interface PlayerStatsTableProps {
  rows: PlayerStatsRowOut[];
  teams: Record<number, TeamOut>;
  perNinety: boolean;
}

export function PlayerStatsTable({ rows, teams, perNinety }: PlayerStatsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("total_points");
  const [sortDescending, setSortDescending] = useState(true);
  const [openBreakdown, setOpenBreakdown] = useState<{ playerId: number; gameweek: number } | null>(
    null,
  );
  const [openActualBreakdown, setOpenActualBreakdown] = useState<number | null>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const scrollRef = useRef<HTMLDivElement>(null);

  const sortedRows = useMemo(() => {
    const withValue = rows.map((row) => ({ row, value: sortValue(row, sortKey, perNinety) }));
    withValue.sort((a, b) => {
      if (typeof a.value === "string" || typeof b.value === "string") {
        return String(a.value).localeCompare(String(b.value)) * (sortDescending ? -1 : 1);
      }
      return (a.value - b.value) * (sortDescending ? -1 : 1);
    });
    return withValue.map((entry) => entry.row);
  }, [rows, sortKey, sortDescending, perNinety]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setSortDescending((prev) => !prev);
    } else {
      setSortKey(key);
      setSortDescending(true);
    }
  }

  const startIndex = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - OVERSCAN);
  const endIndex = Math.min(
    sortedRows.length,
    Math.ceil((scrollTop + CONTAINER_HEIGHT) / ROW_HEIGHT) + OVERSCAN,
  );
  const visibleRows = sortedRows.slice(startIndex, endIndex);
  const topPadding = startIndex * ROW_HEIGHT;
  const bottomPadding = (sortedRows.length - endIndex) * ROW_HEIGHT;

  function sortIndicator(key: SortKey): string {
    if (key !== sortKey) return "";
    return sortDescending ? " ▼" : " ▲";
  }

  return (
    <div
      className="stats-table-scroll"
      ref={scrollRef}
      style={{ height: CONTAINER_HEIGHT, overflowY: "auto", overflowX: "auto" }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
    >
      <table className="stats-table">
        <thead>
          <tr>
            <th className="sortable" onClick={() => toggleSort("name")}>
              Player{sortIndicator("name")}
            </th>
            <th className="sortable" onClick={() => toggleSort("price")}>
              Price{sortIndicator("price")}
            </th>
            <th className="sortable" onClick={() => toggleSort("apps")}>
              Apps{sortIndicator("apps")}
            </th>
            <th
              className="sortable"
              onClick={() => toggleSort("minutes")}
              title="Average minutes played per match started or appeared in, not total minutes across the selected range"
            >
              Mins/Match{sortIndicator("minutes")}
            </th>
            {STAT_COLUMNS.map((column) => (
              <th key={column.key} className="sortable" onClick={() => toggleSort(column.key)}>
                {column.label}
                {sortIndicator(column.key)}
              </th>
            ))}
            <th className="sortable" onClick={() => toggleSort("selected_by_percent")}>
              Own%{sortIndicator("selected_by_percent")}
            </th>
            <th
              colSpan={3}
              className="sortable"
              onClick={() => toggleSort("horizon_points")}
            >
              Next 3 GWs{sortIndicator("horizon_points")}
            </th>
          </tr>
        </thead>
        <tbody>
          {topPadding > 0 && (
            <tr style={{ height: topPadding }}>
              <td colSpan={5 + STAT_COLUMNS.length} />
            </tr>
          )}
          {visibleRows.map((row) => (
            <tr key={row.player_id} style={{ height: ROW_HEIGHT }}>
              <td>
                {row.name}
                {row.low_confidence && <span className="badge low-confidence-badge">!</span>}
                {row.actuals.small_sample && (
                  <span className="badge small-sample-badge" title="Small sample this range">
                    n
                  </span>
                )}
                <span className="panel-team">
                  {row.team_id !== null ? teams[row.team_id]?.short_name : ""}
                </span>
              </td>
              <td>{row.price !== null ? `£${(row.price / 10).toFixed(1)}m` : "—"}</td>
              <td>{row.actuals.apps}</td>
              <td>{averageMinutesPerMatch(row).toFixed(1)}</td>
              {STAT_COLUMNS.map((column) => {
                const isTotalPoints = column.key === "total_points";
                return (
                  <td
                    key={column.key}
                    className={isTotalPoints ? "clickable-cell" : undefined}
                    onClick={
                      isTotalPoints
                        ? () =>
                            setOpenActualBreakdown((prev) =>
                              prev === row.player_id ? null : row.player_id,
                            )
                        : undefined
                    }
                    style={{ position: isTotalPoints ? "relative" : undefined }}
                  >
                    {formatStat(statValue(row, column, perNinety), column, perNinety)}
                    {isTotalPoints && openActualBreakdown === row.player_id && (
                      <div className="popover-anchor">
                        <ActualBreakdownPopover breakdown={row.actuals.points_breakdown} />
                      </div>
                    )}
                  </td>
                );
              })}
              <td>
                {row.actuals.selected_by_percent !== null
                  ? `${row.actuals.selected_by_percent.toFixed(1)}%`
                  : "—"}
              </td>
              {row.fixtures.map((fixture) => (
                <td
                  key={fixture.gameweek}
                  className="clickable-cell"
                  style={{
                    background: expectedPointsColour(fixture.expected_points),
                    color: expectedPointsTextColour(fixture.expected_points),
                    position: "relative",
                  }}
                  onClick={() =>
                    setOpenBreakdown((prev) =>
                      prev?.playerId === row.player_id && prev.gameweek === fixture.gameweek
                        ? null
                        : { playerId: row.player_id, gameweek: fixture.gameweek },
                    )
                  }
                >
                  <div>{fixture.expected_points !== null ? fixture.expected_points.toFixed(1) : "—"}</div>
                  <div className="fixture-label">
                    {fixture.opponent_id !== null && fixture.is_home !== null
                      ? `${teams[fixture.opponent_id]?.short_name ?? fixture.opponent_id} (${fixture.is_home ? "H" : "A"})`
                      : "—"}
                  </div>
                  {openBreakdown?.playerId === row.player_id &&
                    openBreakdown.gameweek === fixture.gameweek && (
                      <div className="popover-anchor">
                        <BreakdownPopover playerId={row.player_id} />
                      </div>
                    )}
                </td>
              ))}
            </tr>
          ))}
          {bottomPadding > 0 && (
            <tr style={{ height: bottomPadding }}>
              <td colSpan={5 + STAT_COLUMNS.length} />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
