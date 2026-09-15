import { useCallback, useEffect, useState } from "react";
import { api, ApiError, RoundupOut } from "../api";

/** The Roundup page's one bulk round trip (ROUNDUP_PLAN) for a single league/gameweek --
 * refetches whenever the league or gameweek changes. `refresh` re-issues the same request
 * bypassing the server's indefinite cache, for the rare case FPL retroactively corrects a
 * result. `gameweek` of `null` asks the server for its own default (the latest data_checked
 * gameweek), whose actual value comes back on the response as `gameweek`. */
export function useRoundup(leagueId: number | null, gameweek: number | null) {
  const [roundup, setRoundup] = useState<RoundupOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (refresh: boolean = false) => {
      if (leagueId === null) {
        setRoundup(null);
        setLoading(false);
        return;
      }
      setLoading(true);
      try {
        const result = await api.getRoundup(leagueId, {
          gameweek: gameweek ?? undefined,
          refresh,
        });
        setRoundup(result);
        setError(null);
      } catch (e) {
        setError(e instanceof ApiError ? e.violation.message : "Failed to load roundup data");
      } finally {
        setLoading(false);
      }
    },
    [leagueId, gameweek],
  );

  useEffect(() => {
    void load();
  }, [load]);

  return { roundup, loading, error, refresh: () => load(true) };
}
