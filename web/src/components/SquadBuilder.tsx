import { useState } from "react";
import { SquadPlayerOut, TeamOut } from "../api";
import { usePlayerPanel } from "../hooks/useProjections";

const QUOTA: Record<string, number> = { GK: 2, DEF: 5, MID: 5, FWD: 3 };
const BUDGET = 1000;

interface SquadBuilderProps {
  picks: SquadPlayerOut[];
  teams: Record<number, TeamOut>;
  onAdd: (playerId: number, position: string, price: number) => void;
  onRemove: (playerId: number) => void;
  onConfirm: (body: {
    player_ids: number[];
    starting_xi: number[];
    bench_order: number[];
    captain_id: number;
    vice_captain_id: number;
  }) => void;
}

/** A sensible default XI/captain from the confirmed 15 -- the manager can immediately refine it
 * afterward via "Optimise lineup" or the normal edit-draft flow; D23 only requires an explicit
 * confirm action once the picks themselves are legal, not that this exact click also finalise
 * every tactical choice. */
function buildDefaultLineup(picks: SquadPlayerOut[]) {
  const byPos: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const p of picks) byPos[p.position]?.push(p.player_id);
  const formations: [number, number, number][] = [
    [4, 4, 2],
    [3, 5, 2],
    [3, 4, 3],
    [4, 3, 3],
    [5, 4, 1],
    [5, 3, 2],
    [4, 5, 1],
  ];
  for (const [d, m, f] of formations) {
    if (byPos.DEF.length >= d && byPos.MID.length >= m && byPos.FWD.length >= f) {
      const xi = [byPos.GK[0], ...byPos.DEF.slice(0, d), ...byPos.MID.slice(0, m), ...byPos.FWD.slice(0, f)];
      const bench = picks.map((p) => p.player_id).filter((id) => !xi.includes(id));
      const reserveGk = byPos.GK[1];
      const benchOrdered = [...bench.filter((id) => id !== reserveGk), reserveGk];
      return {
        starting_xi: xi,
        bench_order: benchOrdered,
        captain_id: byPos.MID[0] ?? xi[1],
        vice_captain_id: byPos.MID[1] ?? xi[2],
      };
    }
  }
  return null;
}

const POSITIONS = ["All", "GK", "DEF", "MID", "FWD"];

export function SquadBuilder({ picks, teams, onAdd, onRemove, onConfirm }: SquadBuilderProps) {
  const [search, setSearch] = useState("");
  const [position, setPosition] = useState("All");
  const { rows } = usePlayerPanel({ search: search || undefined, position: position === "All" ? undefined : position });

  const spent = picks.reduce((sum, p) => sum + p.purchase_price, 0);
  const remaining = BUDGET - spent;
  const counts: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  for (const p of picks) counts[p.position] = (counts[p.position] ?? 0) + 1;
  const isComplete =
    picks.length === 15 &&
    Object.entries(QUOTA).every(([pos, n]) => counts[pos] === n) &&
    remaining >= 0;
  const pickedIds = new Set(picks.map((p) => p.player_id));

  function handleConfirm() {
    const lineup = buildDefaultLineup(picks);
    if (!lineup) return;
    onConfirm({ player_ids: picks.map((p) => p.player_id), ...lineup });
  }

  return (
    <div className="squad-builder">
      <h2>Build your squad</h2>
      <div className="builder-summary">
        <span>Budget remaining: £{(remaining / 10).toFixed(1)}m</span>
        {Object.entries(QUOTA).map(([pos, n]) => (
          <span key={pos}>
            {pos}: {counts[pos] ?? 0}/{n}
          </span>
        ))}
        <span>{picks.length}/15 picked</span>
      </div>

      <div className="builder-picks">
        {picks.map((p) => (
          <div key={p.player_id} className="builder-pick">
            #{p.player_id} ({p.position}) £{(p.purchase_price / 10).toFixed(1)}m
            <button onClick={() => onRemove(p.player_id)}>Remove</button>
          </div>
        ))}
      </div>

      <button disabled={!isComplete} onClick={handleConfirm}>
        Confirm squad
      </button>

      <div className="builder-search">
        <div className="position-tabs">
          {POSITIONS.map((p) => (
            <button key={p} className={position === p ? "active" : ""} onClick={() => setPosition(p)}>
              {p}
            </button>
          ))}
        </div>
        <input
          type="search"
          placeholder="Search by player or team name"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
        <table>
          <tbody>
            {rows
              .filter((row) => !pickedIds.has(row.player_id))
              .map((row) => (
                <tr key={row.player_id}>
                  <td>{row.name}</td>
                  <td>{row.position}</td>
                  <td>{row.team_id !== null ? teams[row.team_id]?.short_name : ""}</td>
                  <td>{row.price !== null ? `£${(row.price / 10).toFixed(1)}m` : "—"}</td>
                  <td>
                    <button
                      disabled={picks.length >= 15}
                      onClick={() => row.price !== null && onAdd(row.player_id, row.position, row.price)}
                    >
                      Add
                    </button>
                  </td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
