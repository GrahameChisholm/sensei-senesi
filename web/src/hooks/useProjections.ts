import { useCallback, useEffect, useState } from "react";
import {
  api,
  FixtureTickerRowOut,
  GameweekOut,
  PlayerPanelRowOut,
  SquadPointsOut,
  TeamOut,
} from "../api";

/** Returns [gameweek, refresh]. */
export function useGameweek(): [GameweekOut | null, () => Promise<void>] {
  const [gameweek, setGameweek] = useState<GameweekOut | null>(null);

  const refresh = useCallback(async () => {
    const result = await api.getGameweek();
    setGameweek(result);
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return [gameweek, refresh];
}

export function useTeams(): Record<number, TeamOut> {
  const [teams, setTeams] = useState<Record<number, TeamOut>>({});

  useEffect(() => {
    void api.getTeams().then((rows) => {
      setTeams(Object.fromEntries(rows.map((row) => [row.team_id, row])));
    });
  }, []);

  return teams;
}

/** Every player's panel row, unfiltered -- doubles as the pitch's own player directory (name,
 * team, position, price, 3 fixture cells) so the pitch never needs a separate bulk lookup.
 * Returns [directory, refresh]. */
export function usePlayerDirectory(): [Record<number, PlayerPanelRowOut>, () => Promise<void>] {
  const [directory, setDirectory] = useState<Record<number, PlayerPanelRowOut>>({});

  const refresh = useCallback(async () => {
    const rows = await api.listPlayers({});
    setDirectory(Object.fromEntries(rows.map((row) => [row.player_id, row])));
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return [directory, refresh];
}

export function useSquadPoints(
  chip: string | null,
  horizon: number,
  refreshKey: unknown,
  enabled: boolean = true,
  gameweek?: number,
) {
  const [points, setPoints] = useState<SquadPointsOut | null>(null);

  useEffect(() => {
    if (!enabled) {
      setPoints(null);
      return;
    }
    let cancelled = false;
    api
      .getSquadPoints(chip, horizon, gameweek)
      .then((result) => {
        if (!cancelled) setPoints(result);
      })
      .catch(() => {
        if (!cancelled) setPoints(null);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [chip, horizon, refreshKey, enabled, gameweek]);

  return points;
}

/** How many transfers each horizon gameweek's own plan has made, keyed by gameweek -- what the
 * gameweek pill selector badges itself with. Fetches `GET /squad?gameweek=X` for every gameweek in
 * parallel (the list is short, today at most 3) rather than needing a dedicated bulk endpoint, and
 * refetches whenever `refreshKey` (the currently-viewed squad's own identity) changes, since an
 * edit to one gameweek can change another's count (a swap at GW2 changes GW3's live-followed
 * squad, for instance, even though GW3 itself wasn't touched). */
export function useSquadTransferCounts(
  gameweeks: number[],
  refreshKey: unknown,
): Record<number, number> {
  const [counts, setCounts] = useState<Record<number, number>>({});
  const gameweeksKey = gameweeks.join(",");

  useEffect(() => {
    if (gameweeks.length === 0) return;
    let cancelled = false;
    Promise.all(gameweeks.map((gw) => api.getSquad(gw)))
      .then((results) => {
        if (cancelled) return;
        setCounts(Object.fromEntries(results.map((r) => [r.gameweek, r.transfers_made])));
      })
      .catch(() => {
        if (!cancelled) setCounts({});
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [gameweeksKey, refreshKey]);

  return counts;
}

export function usePlayerPanel(filters: {
  position?: string;
  min_price?: number;
  max_price?: number;
  search?: string;
}) {
  const [rows, setRows] = useState<PlayerPanelRowOut[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .listPlayers(filters)
      .then((result) => {
        if (!cancelled) setRows(result);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.position, filters.min_price, filters.max_price, filters.search]);

  return { rows, loading };
}

export function useFixtureTicker(gameweekFrom?: number, gameweekTo?: number) {
  const [rows, setRows] = useState<FixtureTickerRowOut[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    setLoading(true);
    let cancelled = false;
    api
      .getFixtureTicker(gameweekFrom, gameweekTo)
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
