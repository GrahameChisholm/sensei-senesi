import { useMemo, useState } from "react";
import { FixtureTickerRowOut, TeamOut } from "../api";
import { useFixtureTicker } from "../hooks/useProjections";
import {
  fixtureDifficultyColour,
  fixtureDifficultyTextColour,
  relativeDifficultyRating,
} from "../lib/colours";

type SortKey = "team" | "difficulty" | "attacking" | "defending";

// Which direction a column's first click lands on -- "best fixtures first" for each metric, not
// always the same direction, since a high value is good for Attacking but bad for Avg FDR/
// Defending. Clicking the same header again just flips it.
const DEFAULT_ASCENDING: Record<SortKey, boolean> = {
  team: true,
  difficulty: true,
  attacking: false,
  defending: true,
};

function sortValue(
  row: FixtureTickerRowOut,
  teams: Record<number, TeamOut>,
  sortKey: SortKey,
): string | number | null {
  switch (sortKey) {
    case "team":
      return (teams[row.team_id]?.name ?? "").toLowerCase();
    case "difficulty":
      return row.average_difficulty;
    case "attacking":
      return row.total_expected_goals_for;
    case "defending":
      return row.total_expected_goals_against;
  }
}

interface FixturesTickerProps {
  teams: Record<number, TeamOut>;
  // Empty (or omitted) means no filter -- every team shows, matching PlayerStats' own team-picker
  // convention (features/player_stats filters use the same "empty selection = show all" rule).
  selectedTeamIds?: Set<number>;
  // Undefined leaves the bound to resolve server-side against the app's own current gameweek,
  // matching useFixtureSwing's own near/far bound convention.
  gameweekFrom?: number;
  gameweekTo?: number;
}

export function FixturesTicker({
  teams,
  selectedTeamIds,
  gameweekFrom,
  gameweekTo,
}: FixturesTickerProps) {
  const [sortKey, setSortKey] = useState<SortKey>("difficulty");
  const [ascending, setAscending] = useState(true);

  const { rows: unsortedRows, loading } = useFixtureTicker(gameweekFrom, gameweekTo);

  const filteredRows = useMemo(() => {
    if (!selectedTeamIds || selectedTeamIds.size === 0) return unsortedRows;
    return unsortedRows.filter((row) => selectedTeamIds.has(row.team_id));
  }, [unsortedRows, selectedTeamIds]);

  const rows = useMemo(() => {
    const withValue = filteredRows.map((row) => ({ row, value: sortValue(row, teams, sortKey) }));
    withValue.sort((a, b) => {
      if (a.value === null) return b.value === null ? 0 : 1;
      if (b.value === null) return -1;
      if (typeof a.value === "string" || typeof b.value === "string") {
        return String(a.value).localeCompare(String(b.value)) * (ascending ? 1 : -1);
      }
      return (a.value - b.value) * (ascending ? 1 : -1);
    });
    return withValue.map((entry) => entry.row);
  }, [filteredRows, teams, sortKey, ascending]);

  function toggleSort(key: SortKey) {
    if (key === sortKey) {
      setAscending((prev) => !prev);
    } else {
      setSortKey(key);
      setAscending(DEFAULT_ASCENDING[key]);
    }
  }

  function sortIndicator(key: SortKey): string {
    if (key !== sortKey) return "";
    return ascending ? " ▲" : " ▼";
  }

  // Attacking/Defending have no fixed 1-5 scale of their own the way FDR does -- their totals
  // scale with however many gameweeks are in view, so their colour is relative to whichever teams
  // are currently shown, reusing the FDR palette via relativeDifficultyRating.
  const attackingValues = filteredRows
    .map((row) => row.total_expected_goals_for)
    .filter((value): value is number => value !== null);
  const defendingValues = filteredRows
    .map((row) => row.total_expected_goals_against)
    .filter((value): value is number => value !== null);
  const attackLow = attackingValues.length ? Math.min(...attackingValues) : 0;
  const attackHigh = attackingValues.length ? Math.max(...attackingValues) : 0;
  const defendLow = defendingValues.length ? Math.min(...defendingValues) : 0;
  const defendHigh = defendingValues.length ? Math.max(...defendingValues) : 0;

  return (
    <div className="player-panel fixture-ticker">
      <h3>Fixtures</h3>

      {loading ? (
        <p>Loading…</p>
      ) : (
        <table className="panel-table">
          <thead>
            <tr>
              <th className="sortable" onClick={() => toggleSort("team")}>
                Team{sortIndicator("team")}
              </th>
              <th className="sortable ticker-metric-col" onClick={() => toggleSort("difficulty")}>
                FDR{sortIndicator("difficulty")}
              </th>
              <th className="sortable ticker-metric-col" onClick={() => toggleSort("attacking")}>
                ATK{sortIndicator("attacking")}
              </th>
              <th className="sortable ticker-metric-col" onClick={() => toggleSort("defending")}>
                DEF{sortIndicator("defending")}
              </th>
              {rows[0]?.gameweeks.map((cell) => <th key={cell.gameweek}>GW{cell.gameweek}</th>)}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const attackRating =
                row.total_expected_goals_for !== null
                  ? relativeDifficultyRating(
                      row.total_expected_goals_for,
                      attackLow,
                      attackHigh,
                      true,
                    )
                  : null;
              const defendRating =
                row.total_expected_goals_against !== null
                  ? relativeDifficultyRating(
                      row.total_expected_goals_against,
                      defendLow,
                      defendHigh,
                      false,
                    )
                  : null;
              return (
                <tr key={row.team_id}>
                  <td>{teams[row.team_id]?.short_name ?? row.team_id}</td>
                  <td
                    className="ticker-metric-col"
                    style={{
                      background: fixtureDifficultyColour(row.average_difficulty),
                      color: fixtureDifficultyTextColour(row.average_difficulty),
                    }}
                  >
                    {row.average_difficulty !== null ? row.average_difficulty.toFixed(1) : "Blank"}
                  </td>
                  <td
                    className="ticker-metric-col"
                    style={{
                      background: fixtureDifficultyColour(attackRating),
                      color: fixtureDifficultyTextColour(attackRating),
                    }}
                  >
                    {row.total_expected_goals_for !== null
                      ? row.total_expected_goals_for.toFixed(1)
                      : "Blank"}
                  </td>
                  <td
                    className="ticker-metric-col"
                    style={{
                      background: fixtureDifficultyColour(defendRating),
                      color: fixtureDifficultyTextColour(defendRating),
                    }}
                  >
                    {row.total_expected_goals_against !== null
                      ? row.total_expected_goals_against.toFixed(1)
                      : "Blank"}
                  </td>
                  {row.gameweeks.map((cell) =>
                    cell.fixtures.length === 0 ? (
                      <td
                        key={cell.gameweek}
                        style={{ background: fixtureDifficultyColour(null), color: fixtureDifficultyTextColour(null) }}
                      >
                        Blank
                      </td>
                    ) : (
                      <td key={cell.gameweek}>
                        {cell.fixtures.map((entry, index) => (
                          <div
                            key={index}
                            className="ticker-fixture-entry"
                            style={{
                              background: fixtureDifficultyColour(entry.difficulty),
                              color: fixtureDifficultyTextColour(entry.difficulty),
                            }}
                          >
                            {teams[entry.opponent_id]?.short_name ?? entry.opponent_id}
                            <span className="fixture-label">{entry.is_home ? "(H)" : "(A)"}</span>
                            {entry.expected_goals_for !== null && entry.expected_goals_against !== null && (
                              <div className="fixture-expected-goals">
                                {entry.expected_goals_for.toFixed(1)} : {entry.expected_goals_against.toFixed(1)}
                              </div>
                            )}
                          </div>
                        ))}
                      </td>
                    ),
                  )}
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}
