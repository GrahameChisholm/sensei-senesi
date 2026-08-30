import { useEffect, useState } from "react";
import { api, OwnershipStatus, PlayerStatsRowOut } from "../api";

/** The Player Stats page's one server round trip (PLAYER_STATS_PLAN D14) -- refetches only when
 * the gameweek range changes, since every other filter (search/team/position/price) is applied
 * client-side over whatever this returns.
 *
 * ``ownershipStatus`` comes back on the response itself rather than being inferred here: the Own%
 * column shows mini-league ownership or nothing at all, so when it is empty the UI states the
 * reason instead of guessing at one. */
export function usePlayerStats(gameweekFrom: number, gameweekTo: number) {
  const [rows, setRows] = useState<PlayerStatsRowOut[]>([]);
  const [ownershipStatus, setOwnershipStatus] = useState<OwnershipStatus>("not_configured");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getPlayerStats(gameweekFrom, gameweekTo)
      .then((result) => {
        if (!cancelled) {
          setRows(result.rows);
          setOwnershipStatus(result.ownership_status);
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [gameweekFrom, gameweekTo]);

  return { rows, ownershipStatus, loading };
}
