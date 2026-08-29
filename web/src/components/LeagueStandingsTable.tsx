import { MiniLeaguePanelOut } from "../api";

const CHIP_LABEL: Record<string, string> = {
  wildcard: "WC",
  freehit: "FH",
  bboost: "BB",
  "3xc": "TC",
};

// Mathematically settled either way -- dimmed so the two or three rivals still genuinely live
// for the run-in don't have to compete for attention with a done deal (MINI_LEAGUE_PLAN M22).
const SETTLED_THRESHOLD = 0.95;

interface LeagueStandingsTableProps {
  panel: MiniLeaguePanelOut;
  selectedRivalId: number | null;
  onSelectRival: (entryId: number) => void;
}

/** Standings plus chip state plus this gameweek's posture, one row per rival (MINI_LEAGUE_PLAN
 * M11/M12/M22 zone 4, right half). Clicking a row both selects it for the head-to-head panel and
 * retargets the header's posture verdict -- one click reframes the whole page around a different
 * opponent. */
export function LeagueStandingsTable({
  panel,
  selectedRivalId,
  onSelectRival,
}: LeagueStandingsTableProps) {
  const rows = [
    { isMe: true, rank: panel.my_rank, name: "You", points: panel.my_total_points, rival: null },
    ...panel.rivals.map((rival) => ({
      isMe: false,
      rank: rival.rank,
      name: rival.manager_name,
      points: rival.total_points,
      rival,
    })),
  ].sort((a, b) => a.rank - b.rank);

  return (
    <div className="player-panel">
      <h3>Standings</h3>
      <table className="panel-table">
        <thead>
          <tr>
            <th title="League rank">#</th>
            <th title="Manager and team name">Manager</th>
            <th title="Total league points this season">Pts</th>
            <th title="Points behind or ahead of you">Gap</th>
            <th title="Chips this rival still has available to play">Chips</th>
            <th title="Modelled probability this rival finishes the season ahead of you, given the current gap and the expected points gap projected forward over the remaining gameweeks">
              P(finish ahead)
            </th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => {
            const settled =
              row.rival !== null &&
              (row.rival.posture.p_finish_ahead >= SETTLED_THRESHOLD ||
                row.rival.posture.p_finish_ahead <= 1 - SETTLED_THRESHOLD);
            const gap = row.points - panel.my_total_points;
            return (
              <tr
                key={row.isMe ? "me" : row.rival!.entry_id}
                className={
                  (row.rival ? "clickable-row " : "") +
                  (settled ? "unfillable-row " : "") +
                  (row.rival?.entry_id === selectedRivalId ? "player-card-swap-source" : "")
                }
                onClick={row.rival ? () => onSelectRival(row.rival!.entry_id) : undefined}
              >
                <td>{row.rank}</td>
                <td>{row.name}</td>
                <td>{row.points}</td>
                <td>{row.isMe ? "—" : `${gap > 0 ? "+" : ""}${gap}`}</td>
                <td>
                  {row.rival
                    ? row.rival.chip_state.remaining_chip_names
                        .map((name) => CHIP_LABEL[name] ?? name)
                        .join(" ") || "—"
                    : "—"}
                </td>
                <td>
                  {row.rival ? `${(row.rival.posture.p_finish_ahead * 100).toFixed(0)}%` : "—"}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
