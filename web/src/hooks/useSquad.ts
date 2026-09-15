import { useCallback, useEffect, useState } from "react";
import { api, ApiError, SquadOut } from "../api";

// The server is always the source of truth: every mutation just calls the API and replaces
// local state with whatever it returns -- never reconstructed client-side. Every action here
// applies instantly, there is no confirm step anywhere.
//
// `gameweek` selects which horizon gameweek's plan this hook reads/edits (the week-on-week
// simulator) -- omit it for the decision gameweek. Every mutation targets that same gameweek, and
// the squad refetches whenever `gameweek` changes so switching the selector shows that gameweek's
// own plan.
export function useSquad(gameweek?: number) {
  const [squad, setSquad] = useState<SquadOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const result = await api.getSquad(gameweek);
      setSquad(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load squad");
    } finally {
      setLoading(false);
    }
  }, [gameweek]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const run = useCallback(async (action: () => Promise<SquadOut>) => {
    try {
      const result = await action();
      setSquad(result);
      setError(null);
      return result;
    } catch (e) {
      setError(e instanceof ApiError ? e.violation.message : "Something went wrong");
      return null;
    }
  }, []);

  return {
    squad,
    error,
    loading,
    clearError: () => setError(null),
    refresh,
    addPlayer: (playerId: number, position: string, price: number) =>
      run(() => api.addPlayer(playerId, position, price, gameweek)),
    removePlayer: (playerId: number) => run(() => api.removePlayer(playerId, gameweek)),
    clearSquad: () => run(() => api.clearSquad()),
    setCaptain: (playerId: number, role: "captain" | "vice") =>
      run(() => api.setCaptain(playerId, role, gameweek)),
    setBenchOrder: (startingXi: number[], benchOrder: number[]) =>
      run(() => api.setBenchOrder(startingXi, benchOrder, gameweek)),
    substitute: (outId: number, inId: number) =>
      run(() => api.substitute(outId, inId, gameweek)),
    optimiseXi: () => run(() => api.optimiseXi(gameweek)),
    optimise: (objective: "starting_xi" | "full_squad", captainMultiplier?: number) =>
      run(() => api.optimise(objective, captainMultiplier, gameweek)),
    importSquad: (teamId: number) => run(() => api.importSquad(teamId)),
    applyTransfers: (outPlayerIds: number[], inPlayerIds: number[]) =>
      run(() => api.applyTransfers(outPlayerIds, inPlayerIds, gameweek)),
  };
}
