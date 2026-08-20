import { SquadPlayerOut } from "../api";

export const QUOTA: Record<string, number> = { GK: 2, DEF: 5, MID: 5, FWD: 3 };
export const BUDGET = 1000;

export interface DefaultLineup {
  starting_xi: number[];
  bench_order: number[];
  captain_id: number;
  vice_captain_id: number;
}

/** A sensible default XI/captain from a complete 15. The manager can immediately refine it
 * afterward via "Optimise lineup" or the normal live-pitch edit flow, so this only needs to
 * produce *a* legal formation, not the best one. */
export function buildDefaultLineup(picks: SquadPlayerOut[]): DefaultLineup | null {
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
