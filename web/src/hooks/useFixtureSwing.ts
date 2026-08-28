import { useEffect, useState } from "react";
import { api, FixtureSwingResponseOut } from "../api";

const EMPTY: FixtureSwingResponseOut = { near_gameweeks: [], far_gameweeks: [], rows: [] };

/** The Fixtures page's swing table -- refetches whenever the near/far window sizes change, since
 * those resolve server-side against the app's own current gameweek (features.fixture_swing),
 * matching useFixtureTicker's own horizon-refetch shape. */
export function useFixtureSwing(near: number, far: number) {
  const [data, setData] = useState<FixtureSwingResponseOut>(EMPTY);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getFixtureSwing(near, far)
      .then((result) => {
        if (!cancelled) setData(result);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [near, far]);

  return {
    nearGameweeks: data.near_gameweeks,
    farGameweeks: data.far_gameweeks,
    rows: data.rows,
    loading,
  };
}
