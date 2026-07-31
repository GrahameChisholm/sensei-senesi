import { useEffect, useState } from "react";
import { api, GameweekOut, PlayerPanelRowOut, SquadPointsOut, TeamOut } from "../api";

export function useGameweek() {
  const [gameweek, setGameweek] = useState<GameweekOut | null>(null);

  useEffect(() => {
    void api.getGameweek().then(setGameweek);
  }, []);

  return gameweek;
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
 * team, position, price, 3 fixture cells) so the pitch never needs a separate bulk lookup. */
export function usePlayerDirectory(): Record<number, PlayerPanelRowOut> {
  const [directory, setDirectory] = useState<Record<number, PlayerPanelRowOut>>({});

  useEffect(() => {
    void api.listPlayers({}).then((rows) => {
      setDirectory(Object.fromEntries(rows.map((row) => [row.player_id, row])));
    });
  }, []);

  return directory;
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
