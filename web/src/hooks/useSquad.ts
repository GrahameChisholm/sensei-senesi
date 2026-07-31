import { useCallback, useEffect, useState } from "react";
import { api, ApiError, SquadOut } from "../api";

// The server is always the source of truth (D16): every mutation just calls the API and replaces
// local state with whatever it returns -- never reconstructed client-side.
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
    addBuildPlayer: (playerId: number, position: string, price: number) =>
      run(() => api.addBuildPlayer(playerId, position, price)),
    removeBuildPlayer: (playerId: number) => run(() => api.removeBuildPlayer(playerId)),
    confirmBuild: (body: {
      player_ids: number[];
      starting_xi: number[];
      bench_order: number[];
      captain_id: number;
      vice_captain_id: number;
    }) => run(() => api.confirmBuild(body)),
    openDraft: () => run(() => api.openDraft()),
    discardDraft: () => run(() => api.discardDraft()),
    substitute: (outId: number, inId: number) => run(() => api.substitute(outId, inId)),
    transfer: (outId: number, inId: number, inPrice: number, inPosition: string) =>
      run(() => api.transfer(outId, inId, inPrice, inPosition)),
    setCaptain: (playerId: number, role: "captain" | "vice") =>
      run(() => api.setCaptain(playerId, role)),
    setBenchOrder: (benchOrder: number[]) => run(() => api.setBenchOrder(benchOrder)),
    setDraftChip: (chip: string | null) => run(() => api.setDraftChip(chip)),
    confirmDraft: () => run(() => api.confirmDraft()),
    optimiseXi: () => run(() => api.optimiseXi()),
  };
}
