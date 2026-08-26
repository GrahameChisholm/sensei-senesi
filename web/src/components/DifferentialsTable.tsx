import { Fragment, useMemo, useState } from "react";
import { Archetype, Confidence, DifferentialRowOut, TeamOut } from "../api";
import { expectedPointsColour, expectedPointsTextColour } from "../lib/colours";
import { BreakdownPopover } from "./BreakdownPopover";

type SortKey =
  | "name"
  | "price"
  | "current_ownership_percent"
  | "ownership_trend_pct_per_gw"
  | "shrunk_points_per_90"
  | "surplus_vs_bracket"
  | "xgi_per_90"
  | "defensive_contribution_per_90"
  | "confidence";

const CONFIDENCE_RANK: Record<Confidence, number> = { low: 0, medium: 1, high: 2 };

const ARCHETYPE_LABEL: Record<Archetype, string> = {
  proven: "Proven",
  emerging: "Emerging",
  riding_luck: "Riding luck",
  none: "",
};

function confidenceDots(confidence: Confidence): string {
  const filled = CONFIDENCE_RANK[confidence] + 1;
  return "●●●".slice(0, filled) + "○○○".slice(0, 3 - filled);
}

function thesis(row: DifferentialRowOut): string {
  const returns = Math.round(row.return_frequency * row.apps_in_window);
  const starts = row.starts_in_window ?? row.apps_in_window;
  switch (row.archetype) {
    case "proven":
      return `Proven. ${starts} starts, returns in ${returns} of ${row.apps_in_window}, underlying numbers back the output.`;
    case "emerging":
      return "Emerging. xGI says the returns are coming; ownership has not noticed yet.";
    case "riding_luck":
      return "Riding luck. Output is running well ahead of xGI, so this is more likely to regress than repeat.";
    default:
      return "";
  }
}

function sortValue(row: DifferentialRowOut, sortKey: SortKey): number | string {
  if (sortKey === "name") return row.name.toLowerCase();
  if (sortKey === "confidence") return CONFIDENCE_RANK[row.confidence];
  const value = row[sortKey];
  return value ?? Number.NEGATIVE_INFINITY;
}

interface DifferentialsTableProps {
  rows: DifferentialRowOut[];
  teams: Record<number, TeamOut>;
}

export function DifferentialsTable({ rows, teams }: DifferentialsTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("surplus_vs_bracket");
  const [sortDescending, setSortDescending] = useState(true);
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [openBreakdown, setOpenBreakdown] = useState<{ playerId: number; gameweek: number } | null>(
    null,
  );

  const sortedRows = useMemo(() => {
    const withValue = rows.map((row) => ({ row, value: sortValue(row, sortKey) }));
    withValue.sort((a, b) => {
      if (typeof a.value === "string" || typeof b.value === "string") {
        return String(a.value).localeCompare(String(b.value)) * (sortDescending ? -1 : 1);
      }
      return (a.value - b.value) * (sortDescending ? -1 : 1);
    });
    return withValue.map((entry) => entry.row);
  }, [rows, sortKey, sortDescending]);

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

  if (rows.length === 0) {
    return (
      <div className="differentials-empty">
        No player currently clears these thresholds. Try widening the ownership ceiling or the
        window.
      </div>
    );
  }

  return (
    <div className="stats-table-scroll">
      <table className="stats-table">
        <thead>
          <tr>
            <th className="sortable" onClick={() => toggleSort("name")}>
              Player{sortIndicator("name")}
            </th>
            <th className="sortable" onClick={() => toggleSort("price")}>
              Price{sortIndicator("price")}
            </th>
            <th className="sortable" onClick={() => toggleSort("current_ownership_percent")}>
              Own%{sortIndicator("current_ownership_percent")}
            </th>
            <th className="sortable" onClick={() => toggleSort("ownership_trend_pct_per_gw")}>
              Trend{sortIndicator("ownership_trend_pct_per_gw")}
            </th>
            <th className="sortable" onClick={() => toggleSort("shrunk_points_per_90")}>
              Pts/90{sortIndicator("shrunk_points_per_90")}
            </th>
            <th className="sortable" onClick={() => toggleSort("surplus_vs_bracket")}>
              vs Bracket{sortIndicator("surplus_vs_bracket")}
            </th>
            <th className="sortable" onClick={() => toggleSort("xgi_per_90")}>
              xGI/90{sortIndicator("xgi_per_90")}
            </th>
            <th className="sortable" onClick={() => toggleSort("defensive_contribution_per_90")}>
              DC/90{sortIndicator("defensive_contribution_per_90")}
            </th>
            <th className="sortable" onClick={() => toggleSort("confidence")}>
              Conf{sortIndicator("confidence")}
            </th>
            <th colSpan={3} className="differentials-muted-header" title="Display only -- not part of the ranking (D4)">
              Next 3
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row) => {
            const hasThesis = row.archetype !== "none";
            const isExpanded = expandedId === row.player_id;
            return (
              <Fragment key={row.player_id}>
                <tr>
                  <td>
                    <button
                      className={`differentials-name${hasThesis ? " clickable-cell" : ""}`}
                      onClick={hasThesis ? () => setExpandedId(isExpanded ? null : row.player_id) : undefined}
                      disabled={!hasThesis}
                    >
                      {row.name}
                      {hasThesis && (
                        <span className={`archetype-badge archetype-${row.archetype}`}>
                          {ARCHETYPE_LABEL[row.archetype]}
                        </span>
                      )}
                    </button>
                    <span className="panel-team">
                      {row.team_id !== null ? teams[row.team_id]?.short_name : ""}
                    </span>
                  </td>
                  <td>£{(row.price / 10).toFixed(1)}m</td>
                  <td>
                    {row.current_ownership_percent !== null
                      ? `${row.current_ownership_percent.toFixed(1)}%`
                      : "—"}
                  </td>
                  <td>
                    {row.ownership_trend_pct_per_gw !== null
                      ? `${row.ownership_trend_pct_per_gw >= 0 ? "▲" : "▼"}${Math.abs(
                          row.ownership_trend_pct_per_gw,
                        ).toFixed(2)}`
                      : "—"}
                  </td>
                  <td>{row.shrunk_points_per_90.toFixed(1)}</td>
                  <td className={row.surplus_vs_bracket >= 0 ? "positive" : "negative"}>
                    {row.surplus_vs_bracket >= 0 ? "+" : ""}
                    {row.surplus_vs_bracket.toFixed(1)}
                  </td>
                  <td>{row.xgi_per_90.toFixed(2)}</td>
                  <td>{row.defensive_contribution_per_90.toFixed(1)}</td>
                  <td className="confidence-dots" title={row.confidence}>
                    {confidenceDots(row.confidence)}
                  </td>
                  {row.fixtures.map((fixture) => (
                    <td
                      key={fixture.gameweek}
                      className="clickable-cell differentials-muted-cell"
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
                      <div>
                        {fixture.expected_points !== null ? fixture.expected_points.toFixed(1) : "—"}
                      </div>
                      <div className="fixture-label">
                        {fixture.opponent_id !== null && fixture.is_home !== null
                          ? `${teams[fixture.opponent_id]?.short_name ?? fixture.opponent_id} (${
                              fixture.is_home ? "H" : "A"
                            })`
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
                {isExpanded && hasThesis && (
                  <tr className="differentials-thesis-row">
                    <td colSpan={9 + 3}>{thesis(row)}</td>
                  </tr>
                )}
              </Fragment>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
