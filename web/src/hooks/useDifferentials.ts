import { useEffect, useState } from "react";
import { api, DifferentialsResponseOut } from "../api";

const EMPTY: DifferentialsResponseOut = {
  window: { gameweek_from: 1, gameweek_to: 0, requested_gameweeks: 0 },
  rows: [],
};

/** The Differentials page's one server round trip -- refetches whenever a server-side filter
 * changes (window length, max ownership, hide-owned), since D1's ownership ceiling and D9's squad
 * exclusion are both domain logic the server applies, not a client-side display filter the way
 * Player Stats' search/team/position filters are. */
export function useDifferentials(
  windowGameweeks: number,
  maxOwnership: number | undefined,
  hideOwned: boolean,
) {
  const [data, setData] = useState<DifferentialsResponseOut>(EMPTY);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getDifferentials(windowGameweeks, maxOwnership, hideOwned)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [windowGameweeks, maxOwnership, hideOwned]);

  return { window: data.window, rows: data.rows, loading };
}
