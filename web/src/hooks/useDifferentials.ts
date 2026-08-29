import { useEffect, useState } from "react";
import { api, DifferentialsResponseOut } from "../api";

const EMPTY: DifferentialsResponseOut = {
  window: { gameweek_from: 1, gameweek_to: 0, requested_gameweeks: 0 },
  ownership_lens: "global",
  picks_gameweek: null,
  n_rivals: null,
  rows: [],
};

/** The Differentials page's one server round trip -- refetches whenever a server-side filter
 * changes (window length, ownership ceiling under whichever lens is active, hide-owned, or which
 * league is selected), since D1's ownership ceiling and D9's squad exclusion are both domain logic
 * the server applies, not a client-side display filter the way Player Stats' search/team/position
 * filters are. Which lens actually won (MINI_LEAGUE_PLAN M24's fallback chain) comes back on the
 * response itself -- this hook never guesses. */
export function useDifferentials(
  windowGameweeks: number,
  maxOwnership: number | undefined,
  maxLeagueOwners: number | undefined,
  leagueId: number | undefined,
  hideOwned: boolean,
) {
  const [data, setData] = useState<DifferentialsResponseOut>(EMPTY);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getDifferentials({
        window: windowGameweeks,
        maxOwnership,
        maxLeagueOwners,
        leagueId,
        hideOwned,
      })
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [windowGameweeks, maxOwnership, maxLeagueOwners, leagueId, hideOwned]);

  return {
    window: data.window,
    ownershipLens: data.ownership_lens,
    picksGameweek: data.picks_gameweek,
    nRivals: data.n_rivals,
    rows: data.rows,
    loading,
  };
}
