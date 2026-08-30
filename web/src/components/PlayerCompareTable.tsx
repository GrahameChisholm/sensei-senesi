import { OwnershipStatus, PlayerStatsRowOut, TeamOut } from "../api";
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
  averageMinutesPerMatch,
  formatRatio,
  formatStat,
  ratioTooltip,
  ratioVerdict,
  statValue,
} from "../lib/playerStats";

interface PlayerCompareTableProps {
  rows: PlayerStatsRowOut[];
  teams: Record<number, TeamOut>;
  perNinety: boolean;
  ownershipStatus: OwnershipStatus;
  onRemove: (playerId: number) => void;
}

/** The selected players' stats transposed -- one column per player, one row per stat -- rather
 * than filtering the main (player-per-row) table down to a handful of rows. Reading across two
 * dozen columns for each of a few players you're actively deciding between is far harder than
 * reading down a column once per stat, which is the entire point of a dedicated comparison view.
 * Reuses the same stat definitions and per-90 toggle as PlayerStatsTable (features/playerStats.ts)
 * so a number here always means the same thing it does in the main table. */
export function PlayerCompareTable({
  rows,
  teams,
  perNinety,
  ownershipStatus,
  onRemove,
}: PlayerCompareTableProps) {
  if (rows.length === 0) return null;

  const gameweeks = Array.from(
    new Set(rows.flatMap((row) => row.fixtures.map((fixture) => fixture.gameweek))),
  ).sort((a, b) => a - b);

  return (
    <div className="player-panel compare-table-panel">
      <h3>Comparing {rows.length} players</h3>
      <div className="stats-table-scroll">
        <table className="stats-table compare-table">
          <thead>
            <tr>
              <th title="Each column is one of the players you've added to compare.">Player</th>
              {rows.map((row) => (
                <th key={row.player_id}>
                  {row.name}
                  <span className="panel-team">
                    {row.team_id !== null ? teams[row.team_id]?.short_name : ""}
                  </span>
                  <button
                    type="button"
                    className="compare-remove"
                    aria-label={`Remove ${row.name} from comparison`}
                    onClick={() => onRemove(row.player_id)}
                  >
                    ×
                  </button>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr>
              <td title="Current FPL price in £m.">Price</td>
              {rows.map((row) => (
                <td key={row.player_id}>{row.price !== null ? `£${(row.price / 10).toFixed(1)}m` : "—"}</td>
              ))}
            </tr>
            <tr>
              <td title="Appearances: gameweeks in the selected range with any minutes played.">
                Apps
              </td>
              {rows.map((row) => (
                <td key={row.player_id}>{row.actuals.apps}</td>
              ))}
            </tr>
            <tr>
              <td title="Average minutes played per match started or appeared in, not total minutes across the selected range">
                Mins/Match
              </td>
              {rows.map((row) => (
                <td key={row.player_id}>{averageMinutesPerMatch(row).toFixed(1)}</td>
              ))}
            </tr>
            {STAT_COLUMNS.map((column) => (
              <tr key={column.key}>
                <td title={column.tooltip}>{column.label}</td>
                {rows.map((row) => (
                  <td key={row.player_id}>{formatStat(statValue(row, column, perNinety), column, perNinety)}</td>
                ))}
              </tr>
            ))}
            {(["attacking", "defensive"] as const).map((kind) => (
              <tr key={kind}>
                <td title={RATIO_COLUMN_TOOLTIP[kind]}>{kind === "attacking" ? "vs xGI" : "vs xCS"}</td>
                {rows.map((row) => {
                  const ratio =
                    kind === "attacking"
                      ? row.actuals.attacking_ratio
                      : row.actuals.defensive_ratio;
                  const conclusive = ratioVerdict(ratio) !== "inconclusive";
                  return (
                    <td
                      key={row.player_id}
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
              </tr>
            ))}
            <tr>
              <td title={OWNERSHIP_STATUS_TOOLTIP[ownershipStatus]}>Own%</td>
              {rows.map((row) => (
                <td key={row.player_id}>
                  {row.actuals.ownership_percent !== null
                    ? `${row.actuals.ownership_percent.toFixed(1)}%`
                    : "—"}
                </td>
              ))}
            </tr>
            {gameweeks.map((gameweek) => (
              <tr key={gameweek}>
                <td title="Predicted expected points for this gameweek, from the engine's projections, with the opponent and venue below it.">
                  GW{gameweek}
                </td>
                {rows.map((row) => {
                  const fixture = row.fixtures.find((f) => f.gameweek === gameweek);
                  return (
                    <td
                      key={row.player_id}
                      style={{
                        background: expectedPointsColour(fixture?.expected_points ?? null),
                        color: expectedPointsTextColour(fixture?.expected_points ?? null),
                      }}
                    >
                      {fixture?.expected_points !== undefined && fixture.expected_points !== null
                        ? fixture.expected_points.toFixed(1)
                        : "—"}
                      <div className="fixture-label">
                        {fixture?.opponent_id !== undefined &&
                        fixture.opponent_id !== null &&
                        fixture.is_home !== null
                          ? `${teams[fixture.opponent_id]?.short_name ?? fixture.opponent_id} (${
                              fixture.is_home ? "H" : "A"
                            })`
                          : "—"}
                      </div>
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
