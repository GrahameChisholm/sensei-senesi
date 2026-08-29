import { MiniLeagueRivalOut, PlayerPanelRowOut } from "../api";
import { exposureColour, exposureTextColour } from "../lib/colours";

/** Bands p_outscore to one of a few plain-language buckets rather than a decimal place the
 * underlying normal-CDF approximation hasn't earned (MINI_LEAGUE_PLAN M9's own stated caveat:
 * players are treated as independent and the gap as normal, both of which understate the tails). */
function bandedOutscore(p: number): string {
  if (p >= 0.9) return "almost certain";
  if (p >= 0.7) return "likely";
  if (p >= 0.55) return "slightly favoured";
  if (p >= 0.45) return "roughly even";
  if (p >= 0.3) return "slightly unfavoured";
  if (p >= 0.1) return "unlikely";
  return "almost never";
}

interface RivalHeadToHeadProps {
  rival: MiniLeagueRivalOut | null;
  directory: Record<number, PlayerPanelRowOut>;
}

/** One rival's decomposition (MINI_LEAGUE_PLAN M8/M22 zone 4, left half): shared picks are
 * collapsed to a count since they contribute nothing to the gap by construction, and only the
 * handful of players that actually differ are shown row by row. */
export function RivalHeadToHead({ rival, directory }: RivalHeadToHeadProps) {
  if (rival === null) {
    return <div className="differentials-empty">Pick a rival from the standings to compare.</div>;
  }

  const h2h = rival.head_to_head;
  const sortedDifferentials = [...h2h.differentials].sort(
    (a, b) => Math.abs(b.expected_gap_contribution) - Math.abs(a.expected_gap_contribution),
  );

  return (
    <div className="player-panel">
      <h3>Head to head vs {rival.manager_name}</h3>
      <div className="header-tiles" style={{ marginBottom: "0.85rem" }}>
        <div className="tile">
          <div
            className="tile-value"
            style={{ color: exposureTextColour(h2h.expected_gap) }}
          >
            {h2h.expected_gap >= 0 ? "+" : ""}
            {h2h.expected_gap.toFixed(1)}
          </div>
          <div className="tile-label">Expected gap</div>
        </div>
        <div className="tile">
          <div className="tile-value">{h2h.gap_std.toFixed(1)}</div>
          <div className="tile-label">Std dev</div>
        </div>
        <div className="tile">
          <div className="tile-value">{(h2h.p_outscore * 100).toFixed(0)}%</div>
          <div className="tile-label">You win it ({bandedOutscore(h2h.p_outscore)})</div>
        </div>
      </div>

      {sortedDifferentials.length === 0 ? (
        <p className="stats-row-count">Every pick matches, nothing separates you this gameweek.</p>
      ) : (
        <table className="panel-table">
          <thead>
            <tr>
              <th title="Player name">Player</th>
              <th title="Your points multiplier for this player: 0 benched, 1 started, 2 captained, 3 triple captained">
                You
              </th>
              <th title="This rival's points multiplier for this player">Them</th>
              <th title="Engine projected points for this gameweek">xP</th>
              <th title="Contribution to the expected points gap between you and this rival: (your multiplier minus theirs) times expected points">
                Impact
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedDifferentials.map((pick) => (
              <tr key={pick.player_id}>
                <td>{directory[pick.player_id]?.name ?? `#${pick.player_id}`}</td>
                <td>{pick.your_multiplier.toFixed(0)}</td>
                <td>{pick.rival_multiplier.toFixed(0)}</td>
                <td>{pick.expected_points !== null ? pick.expected_points.toFixed(1) : "—"}</td>
                <td
                  style={{
                    background: exposureColour(pick.expected_gap_contribution),
                    color: exposureTextColour(pick.expected_gap_contribution),
                  }}
                >
                  {pick.expected_gap_contribution >= 0 ? "+" : ""}
                  {pick.expected_gap_contribution.toFixed(1)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      <p className="stats-row-count" style={{ marginTop: "0.6rem" }}>
        {h2h.shared_count} shared picks cancel out.
      </p>
    </div>
  );
}
