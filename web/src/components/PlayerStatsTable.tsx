import { Fragment, useMemo, useRef, useState } from "react";
import { ComponentBreakdownOut, OwnershipStatus, PlayerStatsRowOut, TeamOut } from "../api";
import {
  expectedPointsColour,
  expectedPointsTextColour,
  ratioColour,
  ratioTextColour,
} from "../lib/colours";
import {
  OWNERSHIP_STATUS_TOOLTIP,
  RATIO_COLUMN_TOOLTIP,
  STAT_COLUMNS,
  STAT_GROUP_META,
  StatColumn,
  StatGroupKey,
  StatKey,
  averageMinutesPerMatch,
  formatRatio,
  formatStat,
  horizonPoints,
  overperformanceBadge,
  pointsPerMillion,
  ratioSortValue,
  ratioTooltip,
  ratioVerdict,
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

type SortKey =
  | "name"
  | "price"
  | "position"
  | "apps"
  | "minutes"
  | "ownership_percent"
  | "points_per_million"
  | "attacking_ratio"
  | "defensive_ratio"
  | "horizon_points"
  | StatKey;

function columnsForGroup(group: StatGroupKey): StatColumn[] {
  return STAT_COLUMNS.filter((column) => column.group === group);
}

// 2 pinned (checkbox, Player) + 5 Meta (Price, Position, Apps, Mins/Match, Own%) + every
// STAT_COLUMNS entry + 1 (Points/£m, not a STAT_COLUMNS entry) + 2 ratio columns + 3 fixtures.
const TOTAL_COLUMNS = 2 + 5 + STAT_COLUMNS.length + 1 + 2 + 3;

function sortValue(row: PlayerStatsRowOut, sortKey: SortKey, perNinety: boolean): number | string {
  if (sortKey === "name") return row.name.toLowerCase();
  if (sortKey === "price") return row.price ?? 0;
  if (sortKey === "position") return row.position;
  if (sortKey === "apps") return row.actuals.apps;
  if (sortKey === "minutes") return averageMinutesPerMatch(row);
  if (sortKey === "ownership_percent") return row.actuals.ownership_percent ?? 0;
  if (sortKey === "points_per_million") return pointsPerMillion(row);
  if (sortKey === "attacking_ratio") return ratioSortValue(row.actuals.attacking_ratio);
  if (sortKey === "defensive_ratio") return ratioSortValue(row.actuals.defensive_ratio);
  if (sortKey === "horizon_points") return horizonPoints(row);
  const column = STAT_COLUMNS.find((c) => c.key === sortKey);
  return column ? statValue(row, column, perNinety) : 0;
}

interface PlayerStatsTableProps {
  rows: PlayerStatsRowOut[];
  teams: Record<number, TeamOut>;
  perNinety: boolean;
  ownershipStatus: OwnershipStatus;
  selectedIds: Set<number>;
  onToggleSelected: (playerId: number) => void;
  horizonGameweeks: number[];
}

export function PlayerStatsTable({
  rows,
  teams,
  perNinety,
  ownershipStatus,
  selectedIds,
  onToggleSelected,
  horizonGameweeks,
}: PlayerStatsTableProps) {
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
            <th
              rowSpan={2}
              className="compare-checkbox-cell"
              title="Check a player to add them to the comparison set."
            />
            <th
              rowSpan={2}
              className="sortable stats-name-header"
              onClick={() => toggleSort("name")}
              title="Player name and team. Badges flag low confidence in the projection, the designated penalty taker, and running hot/cold on their underlying numbers."
            >
              Player{sortIndicator("name")}
            </th>
            <th colSpan={5} title="Price, position, appearances, minutes, and mini-league ownership.">
              Meta
            </th>
            <th
              colSpan={columnsForGroup("historic").length}
              className={STAT_GROUP_META.historic.className}
              title="Actual output over the selected gameweek range."
            >
              {STAT_GROUP_META.historic.label}
            </th>
            <th
              colSpan={columnsForGroup("points").length + 1}
              className={STAT_GROUP_META.points.className}
              title="Points scored, and points per £m of price."
            >
              {STAT_GROUP_META.points.label}
            </th>
            <th
              colSpan={columnsForGroup("expected").length}
              className={STAT_GROUP_META.expected.className}
              title="The underlying process (Opta xG/xA/xGC) behind what actually happened."
            >
              {STAT_GROUP_META.expected.label}
            </th>
            <th
              colSpan={2}
              className="stats-group-band--ratio"
              title="Actual output against the underlying expected numbers -- running hot, cold, or in line with the process."
            >
              Expected v Performance
            </th>
            <th
              colSpan={3}
              className="sortable stats-group-band--fixtures"
              onClick={() => toggleSort("horizon_points")}
              title="Predicted expected points for each of the next 3 gameweeks, from the engine's projections. Sorts by the 3-gameweek total; click a cell to see that gameweek's breakdown."
            >
              Next 3 GWs{sortIndicator("horizon_points")}
            </th>
          </tr>
          <tr>
            <th
              className="sortable"
              onClick={() => toggleSort("price")}
              title="Current FPL price in £m."
            >
              Price{sortIndicator("price")}
            </th>
            <th
              className="sortable"
              onClick={() => toggleSort("position")}
              title="Playing position."
            >
              Pos{sortIndicator("position")}
            </th>
            <th
              className="sortable"
              onClick={() => toggleSort("apps")}
              title="Appearances: gameweeks in the selected range with any minutes played."
            >
              Apps{sortIndicator("apps")}
            </th>
            <th
              className="sortable"
              onClick={() => toggleSort("minutes")}
              title="Average minutes played per match started or appeared in, not total minutes across the selected range"
            >
              Mins/Match{sortIndicator("minutes")}
            </th>
            <th
              className="sortable"
              onClick={() => toggleSort("ownership_percent")}
              title={OWNERSHIP_STATUS_TOOLTIP[ownershipStatus]}
            >
              Own%{sortIndicator("ownership_percent")}
            </th>
            {columnsForGroup("historic").map((column) => (
              <th
                key={column.key}
                className={`sortable ${STAT_GROUP_META.historic.className}`}
                onClick={() => toggleSort(column.key)}
                title={column.tooltip}
              >
                {column.label}
                {sortIndicator(column.key)}
              </th>
            ))}
            {columnsForGroup("points").map((column) => (
              <th
                key={column.key}
                className={`sortable ${STAT_GROUP_META.points.className}`}
                onClick={() => toggleSort(column.key)}
                title={column.tooltip}
              >
                {column.label}
                {sortIndicator(column.key)}
              </th>
            ))}
            <th
              className={`sortable ${STAT_GROUP_META.points.className}`}
              onClick={() => toggleSort("points_per_million")}
              title="Total points scored per £1m of current price, summed over the selected range -- a cumulative value-for-money figure, unaffected by the Per 90 toggle."
            >
              Pts/£m{sortIndicator("points_per_million")}
            </th>
            {columnsForGroup("expected").map((column) => (
              <th
                key={column.key}
                className={`sortable ${STAT_GROUP_META.expected.className}`}
                onClick={() => toggleSort(column.key)}
                title={column.tooltip}
              >
                {column.label}
                {sortIndicator(column.key)}
              </th>
            ))}
            <th
              className="sortable stats-group-band--ratio"
              onClick={() => toggleSort("attacking_ratio")}
              title={RATIO_COLUMN_TOOLTIP.attacking}
            >
              vs xGI{sortIndicator("attacking_ratio")}
            </th>
            <th
              className="sortable stats-group-band--ratio"
              onClick={() => toggleSort("defensive_ratio")}
              title={RATIO_COLUMN_TOOLTIP.defensive}
            >
              vs xCS{sortIndicator("defensive_ratio")}
            </th>
            {[0, 1, 2].map((i) => (
              <th key={i} className="stats-group-band--fixtures">
                {horizonGameweeks[i] !== undefined ? `GW ${horizonGameweeks[i]}` : ""}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {topPadding > 0 && (
            <tr style={{ height: topPadding }}>
              <td colSpan={TOTAL_COLUMNS} />
            </tr>
          )}
          {visibleRows.map((row) => (
            <tr key={row.player_id} style={{ height: ROW_HEIGHT }}>
              <td className="compare-checkbox-cell">
                <input
                  type="checkbox"
                  className="compare-checkbox"
                  checked={selectedIds.has(row.player_id)}
                  onChange={() => onToggleSelected(row.player_id)}
                  aria-label={`Add ${row.name} to comparison`}
                />
              </td>
              <td>
                {row.name}
                {row.low_confidence && <span className="badge low-confidence-badge">!</span>}
                {row.actuals.is_penalty_taker && (
                  <span
                    className="badge penalty-taker-badge"
                    title="On penalties. Penalty volume persists but conversion regresses hard toward ~79%, so a hot run here means less than the same run from open play."
                  >
                    P
                  </span>
                )}
                {(() => {
                  const badge = overperformanceBadge(row);
                  return badge === null ? null : (
                    <span className={`overperf-badge ${badge.className}`} title={badge.title}>
                      {badge.label}
                    </span>
                  );
                })()}
                <span className="panel-team">
                  {row.team_id !== null ? teams[row.team_id]?.short_name : ""}
                </span>
              </td>
              <td>{row.price !== null ? `£${(row.price / 10).toFixed(1)}m` : "—"}</td>
              <td>{row.position}</td>
              <td>{row.actuals.apps}</td>
              <td>{averageMinutesPerMatch(row).toFixed(1)}</td>
              <td>
                {row.actuals.ownership_percent !== null
                  ? `${row.actuals.ownership_percent.toFixed(1)}%`
                  : "—"}
              </td>
              {STAT_COLUMNS.map((column) => {
                const isTotalPoints = column.key === "total_points";
                return (
                  <Fragment key={column.key}>
                    <td
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
                    {isTotalPoints && <td>{pointsPerMillion(row).toFixed(1)}</td>}
                  </Fragment>
                );
              })}
              {(["attacking", "defensive"] as const).map((kind) => {
                const ratio =
                  kind === "attacking"
                    ? row.actuals.attacking_ratio
                    : row.actuals.defensive_ratio;
                const conclusive = ratioVerdict(ratio) !== "inconclusive";
                return (
                  <td
                    key={kind}
                    title={ratioTooltip(ratio, kind)}
                    style={{
                      background: ratioColour(ratio?.ratio ?? null, conclusive),
                      color: ratioTextColour(ratio?.ratio ?? null, conclusive),
                    }}
                  >
                    {formatRatio(ratio)}
                  </td>
                );
              })}
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
              <td colSpan={TOTAL_COLUMNS} />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
