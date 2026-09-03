import { useCallback, useEffect, useState } from "react";
import { api, ApiError, TransferSuggestionOut } from "../api";

/** The Team page banner's suggestion. Refetches whenever the transfer count, horizon, chip, or
 * the squad itself changes, which is what `squadKey` is for: the server caches a solve against
 * exactly those inputs, so re-requesting after an unrelated render is cheap and re-requesting
 * after a real squad edit correctly misses that cache and re-solves.
 *
 * `enabled` is false until the squad is a complete 15, since the endpoint requires one. Asking
 * before then would be a guaranteed 400 rendered as an error the manager cannot act on. */
export function useTransferSuggestion(options: {
  transfers: number;
  horizon: number;
  chip: string | null;
  squadKey: string;
  enabled: boolean;
}) {
  const { transfers, horizon, chip, squadKey, enabled } = options;
  const [suggestion, setSuggestion] = useState<TransferSuggestionOut | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    if (!enabled) {
      setSuggestion(null);
      return;
    }
    setLoading(true);
    try {
      setSuggestion(await api.getTransferSuggestions({ transfers, horizon, chip }));
      setError(null);
    } catch (e) {
      setError(e instanceof ApiError ? e.violation.message : "Failed to load transfer suggestions");
      setSuggestion(null);
    } finally {
      setLoading(false);
    }
    // squadKey is a real dependency even though it is not passed to the request: it is what makes
    // a squad edit refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [transfers, horizon, chip, squadKey, enabled]);

  useEffect(() => {
    void load();
  }, [load]);

  return { suggestion, loading, error, refresh: load };
}
