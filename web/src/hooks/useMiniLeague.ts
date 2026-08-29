import { useCallback, useEffect, useState } from "react";
import { api, ApiError, MiniLeaguePanelOut, MiniLeagueSettingsOut } from "../api";

// The app-wide FPL team ID / tracked league IDs (MINI_LEAGUE_PLAN M14) -- separate from the
// squad's own state since these persist independently of any one squad.
export function useMiniLeagueSettings() {
  const [settings, setSettings] = useState<MiniLeagueSettingsOut | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setSettings(await api.getMiniLeagueSettings());
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const save = useCallback(async (fplTeamId: number | null, miniLeagueIds: number[]) => {
    const result = await api.setMiniLeagueSettings(fplTeamId, miniLeagueIds);
    setSettings(result);
    return result;
  }, []);

  return { settings, loading, save, refresh };
}

/** The Mini League page's one bulk round trip (M17) for a single league -- refetches whenever
 * the league or chip preview changes; `refresh` re-issues the same request bypassing the
 * server's TTL cache (M15), for the moments right after a deadline. */
export function useMiniLeaguePanel(leagueId: number | null, chip: string | null) {
  const [panel, setPanel] = useState<MiniLeaguePanelOut | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(
    async (refresh: boolean = false) => {
      if (leagueId === null) {
        setPanel(null);
        setLoading(false);
        return;
      }
      setLoading(true);
      try {
        const result = await api.getMiniLeague(leagueId, { refresh, chip });
        setPanel(result);
        setError(null);
      } catch (e) {
        setError(e instanceof ApiError ? e.violation.message : "Failed to load mini-league data");
      } finally {
        setLoading(false);
      }
    },
    [leagueId, chip],
  );

  useEffect(() => {
    void load();
  }, [load]);

  return { panel, loading, error, refresh: () => load(true) };
}
