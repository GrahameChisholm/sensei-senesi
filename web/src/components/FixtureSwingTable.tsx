import { useMemo, useState } from "react";
import { TeamOut, TeamSwingRowOut } from "../api";
import { fixtureDifficultyColour, fixtureDifficultyTextColour } from "../lib/colours";

type SortKey =
  | "team"
  | "near_attack"
  | "near_defense"
  | "far_attack"
  | "far_defense"
  | "attack_swing"
  | "defense_swing";

function sortValue(row: TeamSwingRowOut, teams: Record<number, TeamOut>, sortKey: SortKey) {
  switch (sortKey) {
    case "team":
      return (teams[row.team_id]?.name ?? "").toLowerCase();
    case "near_attack":
      return row.near?.attack_rating ?? Number.NEGATIVE_INFINITY;
    case "near_defense":
      return row.near?.defense_rating ?? Number.NEGATIVE_INFINITY;
    case "far_attack":
      return row.far?.attack_rating ?? Number.NEGATIVE_INFINITY;
    case "far_defense":
      return row.far?.defense_rating ?? Number.NEGATIVE_INFINITY;
    case "attack_swing":
      return row.attack_swing ?? Number.NEGATIVE_INFINITY;
    case "defense_swing":
      return row.defense_swing ?? Number.NEGATIVE_INFINITY;
  }
}

/** Green for an improving swing, red for a worsening one, muted for flat/unknown -- reuses the
 * same high/low text colour tokens the Differentials table's surplus_vs_bracket column already
 * uses for a signed value, rather than inventing a second green/red convention. */
function swingTextColour(swing: number | null): string {
  if (swing === null || swing === 0) return "var(--text-muted)";
  return swing > 0 ? "var(--high-text)" : "var(--low-text)";
}

function formatSwing(swing: number | null): string {
  if (swing === null) return "—";
  if (swing === 0) return "0";
  return swing > 0 ? `+${swing}` : `${swing}`;
}

function RatingCell({ rating }: { rating: number | null }) {
  return (
    <td
      style={{
        background: fixtureDifficultyColour(rating),
        color: fixtureDifficultyTextColour(rating),
      }}
    >
      {rating ?? "Blank"}
    </td>
  );
}

interface FixtureSwingTableProps {
  rows: TeamSwingRowOut[];
  teams: Record<number, TeamOut>;
  nearGameweeks: number[];
  farGameweeks: number[];
  // Empty (or omitted) means no filter -- every team shows, matching FixturesTicker's own
  // selectedTeamIds convention so both tables share one selection with identical semantics.
  selectedTeamIds?: Set<number>;
}

export function FixtureSwingTable({
  rows,
  teams,
  nearGameweeks,
  farGameweeks,
  selectedTeamIds,
}: FixtureSwingTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("attack_swing");
  const [sortDescending, setSortDescending] = useState(true);

  const sortedRows = useMemo(() => {
    const filtered =
      !selectedTeamIds || selectedTeamIds.size === 0
        ? rows
        : rows.filter((row) => selectedTeamIds.has(row.team_id));
    const withValue = filtered.map((row) => ({ row, value: sortValue(row, teams, sortKey) }));
    withValue.sort((a, b) => {
      if (typeof a.value === "string" || typeof b.value === "string") {
        return String(a.value).localeCompare(String(b.value)) * (sortDescending ? -1 : 1);
      }
      return (a.value - b.value) * (sortDescending ? -1 : 1);
    });
    return withValue.map((entry) => entry.row);
  }, [rows, teams, sortKey, sortDescending, selectedTeamIds]);

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

  const nearLabel = nearGameweeks.length
    ? `GW${nearGameweeks[0]}-${nearGameweeks[nearGameweeks.length - 1]}`
    : "Near";
  const farLabel = farGameweeks.length
    ? `GW${farGameweeks[0]}-${farGameweeks[farGameweeks.length - 1]}`
    : "Far";

  return (
    <div className="player-panel fixture-swing">
      <h3>Fixture swing</h3>
      <p className="differentials-muted-header">
        Is this team's run of fixtures getting easier ({nearLabel} vs {farLabel})? A positive swing
        means the near-term fixtures are harder than what's coming -- buy in before it turns.
      </p>

      {rows.length === 0 ? (
        <div className="differentials-empty">No team-rate data available yet.</div>
      ) : (
        <div className="stats-table-scroll">
          <table className="stats-table">
            <thead>
              <tr>
                <th className="sortable" onClick={() => toggleSort("team")}>
                  Team{sortIndicator("team")}
                </th>
                <th className="sortable" onClick={() => toggleSort("near_attack")}>
                  Near Atk{sortIndicator("near_attack")}
                </th>
                <th className="sortable" onClick={() => toggleSort("near_defense")}>
                  Near Def{sortIndicator("near_defense")}
                </th>
                <th className="sortable" onClick={() => toggleSort("far_attack")}>
                  Far Atk{sortIndicator("far_attack")}
                </th>
                <th className="sortable" onClick={() => toggleSort("far_defense")}>
                  Far Def{sortIndicator("far_defense")}
                </th>
                <th className="sortable" onClick={() => toggleSort("attack_swing")}>
                  Atk Swing{sortIndicator("attack_swing")}
                </th>
                <th className="sortable" onClick={() => toggleSort("defense_swing")}>
                  Def Swing{sortIndicator("defense_swing")}
                </th>
              </tr>
            </thead>
            <tbody>
              {sortedRows.map((row) => (
                <tr key={row.team_id}>
                  <td>
                    {teams[row.team_id]?.short_name ?? row.team_id}
                    {row.has_owned_player && (
                      <span className="archetype-badge owned-badge">Owned</span>
                    )}
                  </td>
                  <RatingCell rating={row.near?.attack_rating ?? null} />
                  <RatingCell rating={row.near?.defense_rating ?? null} />
                  <RatingCell rating={row.far?.attack_rating ?? null} />
                  <RatingCell rating={row.far?.defense_rating ?? null} />
                  <td style={{ color: swingTextColour(row.attack_swing), fontWeight: 700 }}>
                    {formatSwing(row.attack_swing)}
                  </td>
                  <td style={{ color: swingTextColour(row.defense_swing), fontWeight: 700 }}>
                    {formatSwing(row.defense_swing)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
