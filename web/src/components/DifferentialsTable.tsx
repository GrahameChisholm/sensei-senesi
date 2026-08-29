import { Fragment, useMemo, useState } from "react";
import {
  Archetype,
  Confidence,
  DifferentialRowOut,
  OwnershipLensSource,
  PlayerPanelRowOut,
  TeamOut,
} from "../api";
import { useStoredState } from "../hooks/useStoredState";
import { DifferentialSortKey } from "../lib/differentialPresets";
import { expectedPointsColour, expectedPointsTextColour, exposureTextColour } from "../lib/colours";
import { BreakdownPopover } from "./BreakdownPopover";

const CONFIDENCE_RANK: Record<Confidence, number> = { low: 0, medium: 1, high: 2 };

const ARCHETYPE_LABEL: Record<Archetype, string> = {
  proven: "Proven",
  emerging: "Emerging",
  riding_luck: "Riding luck",
  none: "",
};

const INLINE_THESIS_COUNT = 5;

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

function sortValue(row: DifferentialRowOut, sortKey: DifferentialSortKey): number | string {
  if (sortKey === "name") return row.name.toLowerCase();
  if (sortKey === "confidence") return CONFIDENCE_RANK[row.confidence];
  const value = row[sortKey];
  return value ?? Number.NEGATIVE_INFINITY;
}

interface DifferentialsTableProps {
  rows: DifferentialRowOut[];
  teams: Record<number, TeamOut>;
  directory: Record<number, PlayerPanelRowOut>;
  ownershipLens: OwnershipLensSource;
  nRivals: number | null;
  sortKey: DifferentialSortKey;
  sortDescending: boolean;
  onSort: (key: DifferentialSortKey) => void;
}

export function DifferentialsTable({
  rows,
  teams,
  directory,
  ownershipLens,
  nRivals,
  sortKey,
  sortDescending,
  onSort,
}: DifferentialsTableProps) {
  const isLeagueLens = ownershipLens === "league";
  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [openBreakdown, setOpenBreakdown] = useState<{ playerId: number; gameweek: number } | null>(
    null,
  );
  // Progressive columns (item 5): the decision-facing columns stay visible by default, the
  // underlying stats a manager checks only after a candidate has already caught their eye move
  // behind this toggle. Persisted per browser view since it's a durable reading preference, not
  // per-request state.
  const [showUnderlying, setShowUnderlying] = useStoredState("differentials.showUnderlying", false);

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

  const inlineThesisIds = useMemo(
    () => new Set(sortedRows.slice(0, INLINE_THESIS_COUNT).map((row) => row.player_id)),
    [sortedRows],
  );

  function sortIndicator(key: DifferentialSortKey): string {
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

  const baseColumnCount =
    2 + // Player, vs Bracket
    (isLeagueLens ? 2 : 1) + // Owned/Own%, plus Swing and Replaces under the league lens
    1 + // Conf
    (showUnderlying ? 5 : 0); // Price, Market Trend, Pts/90, xGI/90, DC/90

  return (
    <div className="stats-table-scroll">
      <table className="stats-table">
        <thead>
          <tr>
            <th className="sortable" onClick={() => onSort("name")} title="Player name and club">
              Player{sortIndicator("name")}
            </th>
            {isLeagueLens ? (
              <th
                className="sortable"
                onClick={() => onSort("league_owner_count")}
                title="How many of your league rivals own this player, out of the total tracked. Hover a value to see who."
              >
                Owned{sortIndicator("league_owner_count")}
              </th>
            ) : (
              <th
                className="sortable"
                onClick={() => onSort("current_ownership_percent")}
                title="Percentage of all FPL managers who currently own this player"
              >
                Own%{sortIndicator("current_ownership_percent")}
              </th>
            )}
            {isLeagueLens && (
              <th
                className="sortable"
                onClick={() => onSort("expected_swing")}
                title="What this player would be worth in expected swing points if brought into your starting XI: (1 minus league effective ownership) times expected points"
              >
                Swing{sortIndicator("expected_swing")}
              </th>
            )}
            <th
              className="sortable"
              onClick={() => onSort("surplus_vs_bracket")}
              title="Shrunk points per 90 minus the median for the same position and price bracket. Positive means outperforming what this price should deliver."
            >
              vs Bracket{sortIndicator("surplus_vs_bracket")}
            </th>
            {isLeagueLens && (
              <th title="Which of your own starting XI this player would most sensibly replace by expected swing. A swing comparison only, not a budget or squad-legality check.">
                Replaces
              </th>
            )}
            <th
              className="sortable"
              onClick={() => onSort("confidence")}
              title="How much evidence backs this player's rate, based on minutes played in the window"
            >
              Conf{sortIndicator("confidence")}
            </th>
            {showUnderlying && (
              <>
                <th className="sortable" onClick={() => onSort("price")} title="Current price">
                  Price{sortIndicator("price")}
                </th>
                <th
                  className="sortable differentials-muted-header"
                  onClick={() => onSort("ownership_trend_pct_per_gw")}
                  title="How fast this player's ownership is changing per gameweek, from FPL's own wide transfer data. Always global, regardless of the ownership view selected above."
                >
                  Market Trend{sortIndicator("ownership_trend_pct_per_gw")}
                </th>
                <th
                  className="sortable"
                  onClick={() => onSort("shrunk_points_per_90")}
                  title="Points per 90 minutes, shrunk toward the median for players at the same position and price so a small sample doesn't overstate it"
                >
                  Pts/90{sortIndicator("shrunk_points_per_90")}
                </th>
                <th
                  className="sortable"
                  onClick={() => onSort("xgi_per_90")}
                  title="Expected goals plus expected assists per 90 minutes, the underlying attacking involvement rate"
                >
                  xGI/90{sortIndicator("xgi_per_90")}
                </th>
                <th
                  className="sortable"
                  onClick={() => onSort("defensive_contribution_per_90")}
                  title="Defensive contribution points per 90 minutes, from tackles, interceptions, clearances and recoveries"
                >
                  DC/90{sortIndicator("defensive_contribution_per_90")}
                </th>
              </>
            )}
            <th
              colSpan={3}
              className="differentials-muted-header clickable-cell"
              title="Display only, not part of the ranking (D4): each gameweek's expected points and opponent. Click to show or hide the underlying rate stats (Price, Market Trend, Pts/90, xGI/90, DC/90)."
              onClick={() => setShowUnderlying((prev) => !prev)}
            >
              Next 3 {showUnderlying ? "· hide underlying ▲" : "· show underlying ▼"}
            </th>
          </tr>
        </thead>
        <tbody>
          {sortedRows.map((row) => {
            const hasThesis = row.archetype !== "none";
            const showInlineThesis = hasThesis && inlineThesisIds.has(row.player_id);
            const isExpanded = expandedId === row.player_id;
            return (
              <Fragment key={row.player_id}>
                <tr>
                  <td>
                    <button
                      className={`differentials-name${
                        hasThesis && !showInlineThesis ? " clickable-cell" : ""
                      }`}
                      onClick={
                        hasThesis && !showInlineThesis
                          ? () => setExpandedId(isExpanded ? null : row.player_id)
                          : undefined
                      }
                      disabled={!hasThesis || showInlineThesis}
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
                    {showInlineThesis && <div className="differentials-inline-thesis">{thesis(row)}</div>}
                  </td>
                  {isLeagueLens ? (
                    <td title={row.league_owner_names.join(", ") || "Owned by nobody in the league"}>
                      {row.league_owner_count ?? 0}
                      {nRivals !== null ? ` of ${nRivals}` : ""}
                    </td>
                  ) : (
                    <td>
                      {row.current_ownership_percent !== null
                        ? `${row.current_ownership_percent.toFixed(1)}%`
                        : "—"}
                    </td>
                  )}
                  {isLeagueLens && (
                    <td style={{ color: exposureTextColour(row.expected_swing), fontWeight: 700 }}>
                      {row.expected_swing !== null
                        ? `${row.expected_swing >= 0 ? "+" : ""}${row.expected_swing.toFixed(1)}`
                        : "—"}
                    </td>
                  )}
                  <td className={row.surplus_vs_bracket >= 0 ? "positive" : "negative"}>
                    {row.surplus_vs_bracket >= 0 ? "+" : ""}
                    {row.surplus_vs_bracket.toFixed(1)}
                  </td>
                  {isLeagueLens && (
                    <td
                      title={
                        row.replaces
                          ? "Swing comparison only -- does not check budget or squad legality"
                          : undefined
                      }
                    >
                      {row.replaces ? (
                        <span style={{ color: exposureTextColour(row.replaces.net_swing_delta) }}>
                          vs {directory[row.replaces.outgoing_player_id]?.name ??
                            `#${row.replaces.outgoing_player_id}`}{" "}
                          ({row.replaces.net_swing_delta >= 0 ? "+" : ""}
                          {row.replaces.net_swing_delta.toFixed(1)}, £
                          {(row.replaces.price_delta / 10).toFixed(1)}m)
                        </span>
                      ) : (
                        "—"
                      )}
                    </td>
                  )}
                  <td className="confidence-dots" title={row.confidence}>
                    {confidenceDots(row.confidence)}
                  </td>
                  {showUnderlying && (
                    <>
                      <td>£{(row.price / 10).toFixed(1)}m</td>
                      <td className="differentials-muted-cell">
                        {row.ownership_trend_pct_per_gw !== null
                          ? `${row.ownership_trend_pct_per_gw >= 0 ? "▲" : "▼"}${Math.abs(
                              row.ownership_trend_pct_per_gw,
                            ).toFixed(2)}`
                          : "—"}
                      </td>
                      <td>{row.shrunk_points_per_90.toFixed(1)}</td>
                      <td>{row.xgi_per_90.toFixed(2)}</td>
                      <td>{row.defensive_contribution_per_90.toFixed(1)}</td>
                    </>
                  )}
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
                {isExpanded && hasThesis && !showInlineThesis && (
                  <tr className="differentials-thesis-row">
                    <td colSpan={baseColumnCount + 3}>{thesis(row)}</td>
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
