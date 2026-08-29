import { useMemo, useState } from "react";
import { CaptainOptionOut, PlayerPanelRowOut } from "../api";

type SortKey = "expected_points" | "captain_share_percent" | "eo_multiplier" | "net_captain_ev";

interface CaptainEoGridProps {
  options: CaptainOptionOut[];
  directory: Record<number, PlayerPanelRowOut>;
  currentCaptainId: number | null;
}

/** Captains ranked by gain on the field, not raw xP (MINI_LEAGUE_PLAN M10) -- a heavily-captained
 * high-xP player can net less than a lightly-owned lower-xP one, which is invisible on the Team
 * page's own captain picker. */
export function CaptainEoGrid({ options, directory, currentCaptainId }: CaptainEoGridProps) {
  const [sortKey, setSortKey] = useState<SortKey>("net_captain_ev");
  const [sortDescending, setSortDescending] = useState(true);

  const sortedOptions = useMemo(() => {
    const withValue = options.map((option) => ({
      option,
      value: option[sortKey] ?? Number.NEGATIVE_INFINITY,
    }));
    withValue.sort((a, b) => (a.value - b.value) * (sortDescending ? -1 : 1));
    return withValue.map((entry) => entry.option);
  }, [options, sortKey, sortDescending]);

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

  return (
    <div className="fixture-swing">
      <p className="stats-filter-label" style={{ marginBottom: "0.5rem" }}>
        Captain, ranked by gain on the field
      </p>
      <div className="stats-table-scroll">
        <table className="stats-table">
          <thead>
            <tr>
              <th title="Player name">Player</th>
              <th
                className="sortable"
                onClick={() => toggleSort("expected_points")}
                title="Engine projected points for this gameweek"
              >
                xP{sortIndicator("expected_points")}
              </th>
              <th
                className="sortable"
                onClick={() => toggleSort("captain_share_percent")}
                title="Percentage of your league rivals who are captaining this player"
              >
                Capt share{sortIndicator("captain_share_percent")}
              </th>
              <th
                className="sortable"
                onClick={() => toggleSort("eo_multiplier")}
                title="League effective ownership multiplier (0 to 3), the average points multiplier for this player across every rival in your league"
              >
                EO mult{sortIndicator("eo_multiplier")}
              </th>
              <th
                className="sortable"
                onClick={() => toggleSort("net_captain_ev")}
                title="Expected gain from captaining this player relative to the field: (2 minus league effective ownership multiplier) times expected points"
              >
                Net EV{sortIndicator("net_captain_ev")}
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedOptions.map((option) => (
              <tr key={option.player_id}>
                <td>
                  {directory[option.player_id]?.name ?? `#${option.player_id}`}
                  {option.player_id === currentCaptainId && (
                    <span className="badge captain-badge" style={{ marginLeft: "0.4rem" }}>
                      C
                    </span>
                  )}
                </td>
                <td>{option.expected_points !== null ? option.expected_points.toFixed(1) : "—"}</td>
                <td>{option.captain_share_percent.toFixed(0)}%</td>
                <td>{option.eo_multiplier.toFixed(2)}</td>
                <td className={(option.net_captain_ev ?? 0) >= 0 ? "positive" : "negative"}>
                  {option.net_captain_ev !== null
                    ? `${option.net_captain_ev >= 0 ? "+" : ""}${option.net_captain_ev.toFixed(1)}`
                    : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
