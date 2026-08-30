import { PlayerStatsRowOut, TeamOut } from "../api";
import { expectedPointsColour, expectedPointsTextColour } from "../lib/colours";
import { STAT_COLUMNS, averageMinutesPerMatch, formatStat, statValue } from "../lib/playerStats";

interface PlayerCompareTableProps {
  rows: PlayerStatsRowOut[];
  teams: Record<number, TeamOut>;
  perNinety: boolean;
  onRemove: (playerId: number) => void;
}

/** The selected players' stats transposed -- one column per player, one row per stat -- rather
 * than filtering the main (player-per-row) table down to a handful of rows. Reading across two
 * dozen columns for each of a few players you're actively deciding between is far harder than
 * reading down a column once per stat, which is the entire point of a dedicated comparison view.
 * Reuses the same stat definitions and per-90 toggle as PlayerStatsTable (features/playerStats.ts)
 * so a number here always means the same thing it does in the main table. */
export function PlayerCompareTable({ rows, teams, perNinety, onRemove }: PlayerCompareTableProps) {
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
              <th>Player</th>
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
              <td>Price</td>
              {rows.map((row) => (
                <td key={row.player_id}>{row.price !== null ? `£${(row.price / 10).toFixed(1)}m` : "—"}</td>
              ))}
            </tr>
            <tr>
              <td>Apps</td>
              {rows.map((row) => (
                <td key={row.player_id}>{row.actuals.apps}</td>
              ))}
            </tr>
            <tr>
              <td>Mins/Match</td>
              {rows.map((row) => (
                <td key={row.player_id}>{averageMinutesPerMatch(row).toFixed(1)}</td>
              ))}
            </tr>
            {STAT_COLUMNS.map((column) => (
              <tr key={column.key}>
                <td>{column.label}</td>
                {rows.map((row) => (
                  <td key={row.player_id}>{formatStat(statValue(row, column, perNinety), column, perNinety)}</td>
                ))}
              </tr>
            ))}
            <tr>
              <td>Own%</td>
              {rows.map((row) => (
                <td key={row.player_id}>
                  {row.actuals.selected_by_percent !== null
                    ? `${row.actuals.selected_by_percent.toFixed(1)}%`
                    : "—"}
                </td>
              ))}
            </tr>
            {gameweeks.map((gameweek) => (
              <tr key={gameweek}>
                <td>GW{gameweek}</td>
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
