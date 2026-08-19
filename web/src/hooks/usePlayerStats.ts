import { useEffect, useState } from "react";
import { api, PlayerStatsRowOut } from "../api";

/** The Player Stats page's one server round trip (PLAYER_STATS_PLAN D14) -- refetches only when
 * the gameweek range changes, since every other filter (search/team/position/price) is applied
 * client-side over whatever this returns. */
export function usePlayerStats(gameweekFrom: number, gameweekTo: number) {
  const [rows, setRows] = useState<PlayerStatsRowOut[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getPlayerStats(gameweekFrom, gameweekTo)
      .then((result) => {
        if (!cancelled) setRows(result);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [gameweekFrom, gameweekTo]);

  return { rows, loading };
}
