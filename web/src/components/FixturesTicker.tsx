import { useMemo, useState } from "react";
import { FixtureTickerRowOut, TeamOut } from "../api";
import { useFixtureTicker } from "../hooks/useProjections";
import { fixtureDifficultyColour } from "../lib/colours";

const HORIZON_OPTIONS = [3, 5, 8, 10];

type SortKey = "difficulty" | "alphabetical";

const SORT_OPTIONS: { key: SortKey; label: string }[] = [
  { key: "difficulty", label: "Best fixtures first" },
  { key: "alphabetical", label: "Alphabetical" },
];

function sortRows(
  rows: FixtureTickerRowOut[],
  teams: Record<number, TeamOut>,
  sortKey: SortKey,
): FixtureTickerRowOut[] {
  const sorted = [...rows];
  if (sortKey === "alphabetical") {
    sorted.sort((a, b) =>
      (teams[a.team_id]?.name ?? "").localeCompare(teams[b.team_id]?.name ?? ""),
    );
    return sorted;
  }
  sorted.sort((a, b) => {
    if (a.average_difficulty === null) return 1;
    if (b.average_difficulty === null) return -1;
    return a.average_difficulty - b.average_difficulty;
  });
  return sorted;
}

interface FixturesTickerProps {
  teams: Record<number, TeamOut>;
}

export function FixturesTicker({ teams }: FixturesTickerProps) {
  const [horizon, setHorizon] = useState(5);
  const [sortKey, setSortKey] = useState<SortKey>("difficulty");

  const { rows: unsortedRows, loading } = useFixtureTicker(horizon);
  const rows = useMemo(() => sortRows(unsortedRows, teams, sortKey), [unsortedRows, teams, sortKey]);

  return (
    <div className="player-panel fixture-ticker">
      <h3>Fixtures</h3>

      <div className="panel-filters">
        <label>
          Gameweeks
          <select value={horizon} onChange={(e) => setHorizon(Number(e.target.value))}>
            {HORIZON_OPTIONS.map((option) => (
              <option key={option} value={option}>
                Next {option}
              </option>
            ))}
          </select>
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
      </div>

      {loading ? (
        <p>Loading…</p>
      ) : (
        <table className="panel-table">
          <thead>
            <tr>
              <th>Team</th>
              <th>Avg FDR</th>
              {rows[0]?.gameweeks.map((cell) => <th key={cell.gameweek}>GW{cell.gameweek}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr key={row.team_id}>
                <td>{teams[row.team_id]?.short_name ?? row.team_id}</td>
                <td style={{ background: fixtureDifficultyColour(row.average_difficulty) }}>
                  {row.average_difficulty !== null ? row.average_difficulty.toFixed(1) : "Blank"}
                </td>
                {row.gameweeks.map((cell) =>
                  cell.fixtures.length === 0 ? (
                    <td key={cell.gameweek} style={{ background: fixtureDifficultyColour(null) }}>
                      Blank
                    </td>
                  ) : (
                    <td key={cell.gameweek}>
                      {cell.fixtures.map((entry, index) => (
                        <div
                          key={index}
                          className="ticker-fixture-entry"
                          style={{ background: fixtureDifficultyColour(entry.difficulty) }}
                        >
                          {teams[entry.opponent_id]?.short_name ?? entry.opponent_id}
                          <span className="fixture-label">{entry.is_home ? "(H)" : "(A)"}</span>
                        </div>
                      ))}
                    </td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}
