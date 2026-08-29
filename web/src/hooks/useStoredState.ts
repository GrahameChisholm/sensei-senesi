import { useEffect, useRef, useState } from "react";

const STORAGE_PREFIX = "fpl-app:v1:";

function readStored<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(STORAGE_PREFIX + key);
    return raw === null ? fallback : (JSON.parse(raw) as T);
  } catch {
    return fallback;
  }
}

function writeStored<T>(key: string, value: T): void {
  try {
    window.localStorage.setItem(STORAGE_PREFIX + key, JSON.stringify(value));
  } catch {
    // Private browsing, blocked site data, or a full quota can make the accessor itself throw --
    // a preference that fails to persist is never worse than a crashed page over it.
  }
}

/** A `useState` that mirrors its value to `localStorage` under a versioned, per-view key --
 * read once per `key` and written on every change, wrapped in `try`/`catch` throughout since a
 * private window or blocked site data can make the accessor itself throw. Changing `key` (e.g.
 * switching which mini-league is selected) re-reads from storage for the new key rather than
 * carrying the previous key's value forward. */
export function useStoredState<T>(
  key: string,
  initial: T,
): [T, (value: T | ((prev: T) => T)) => void] {
  const [value, setValue] = useState<T>(() => readStored(key, initial));
  const previousKey = useRef(key);

  useEffect(() => {
    if (previousKey.current !== key) {
      previousKey.current = key;
      setValue(readStored(key, initial));
    }
    // Only ever reacts to `key` changing -- `initial` is a per-render literal in every call site
    // this hook has, so depending on it would re-read on every render instead of only on a real
    // key change.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  useEffect(() => {
    writeStored(key, value);
  }, [key, value]);

  return [value, setValue];
}
