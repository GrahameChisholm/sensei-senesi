import { useCallback, useEffect, useState } from "react";
import { api, ApiError, SquadOut } from "../api";

// The server is always the source of truth: every mutation just calls the API and replaces
// local state with whatever it returns -- never reconstructed client-side. Every action here
// applies instantly, there is no confirm step anywhere.
export function useSquad() {
  const [squad, setSquad] = useState<SquadOut | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const result = await api.getSquad();
      setSquad(result);
      setError(null);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Failed to load squad");
    } finally {
      setLoading(false);
    }
  }, []);

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
      run(() => api.addPlayer(playerId, position, price)),
    removePlayer: (playerId: number) => run(() => api.removePlayer(playerId)),
    clearSquad: () => run(() => api.clearSquad()),
    setCaptain: (playerId: number, role: "captain" | "vice") =>
      run(() => api.setCaptain(playerId, role)),
    setBenchOrder: (startingXi: number[], benchOrder: number[]) =>
      run(() => api.setBenchOrder(startingXi, benchOrder)),
    substitute: (outId: number, inId: number) => run(() => api.substitute(outId, inId)),
    optimiseXi: () => run(() => api.optimiseXi()),
    optimise: (objective: "starting_xi" | "full_squad", captainMultiplier?: number) =>
      run(() => api.optimise(objective, captainMultiplier)),
    importSquad: (teamId: number) => run(() => api.importSquad(teamId)),
    applyTransfers: (outPlayerIds: number[], inPlayerIds: number[]) =>
      run(() => api.applyTransfers(outPlayerIds, inPlayerIds)),
  };
}
