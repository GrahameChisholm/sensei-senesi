import { useEffect, useState } from "react";
import { api, FixtureSwingResponseOut } from "../api";

const EMPTY: FixtureSwingResponseOut = { near_gameweeks: [], far_gameweeks: [], rows: [] };

/** The Fixtures page's swing table -- refetches whenever either window's bounds change. Any bound
 * left undefined resolves server-side against the app's own current gameweek
 * (features.fixture_swing's locked-in defaults), matching useFixtureTicker's own
 * horizon-refetch shape. */
export function useFixtureSwing(
  nearFrom?: number,
  nearTo?: number,
  farFrom?: number,
  farTo?: number,
) {
  const [data, setData] = useState<FixtureSwingResponseOut>(EMPTY);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getFixtureSwing(nearFrom, nearTo, farFrom, farTo)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [nearFrom, nearTo, farFrom, farTo]);

  return {
    nearGameweeks: data.near_gameweeks,
    farGameweeks: data.far_gameweeks,
    rows: data.rows,
    loading,
  };
}
