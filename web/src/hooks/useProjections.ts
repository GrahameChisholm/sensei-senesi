import { useCallback, useEffect, useState } from "react";
import { api, GameweekOut, PlayerPanelRowOut, SquadPointsOut, TeamOut } from "../api";

/** Returns [gameweek, refresh] -- Season Replay's "Advance" moves the process-wide app state on to
 * the next gameweek's cache server-side, so the header needs an explicit way to refetch rather
 * than relying on the once-on-mount fetch alone. */
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
 * Returns [directory, refresh] -- Season Replay's "Advance" needs to force a refetch once the
 * process-wide app state has moved on to a new gameweek's prices/fixtures/projections. */
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
  source: "draft" | "committed" = "draft",
  enabled: boolean = true,
) {
  const [points, setPoints] = useState<SquadPointsOut | null>(null);

  useEffect(() => {
    if (!enabled) {
      setPoints(null);
      return;
    }
    let cancelled = false;
    api
      .getSquadPoints(chip, horizon, source)
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
  }, [chip, horizon, refreshKey, source, enabled]);

  return points;
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
