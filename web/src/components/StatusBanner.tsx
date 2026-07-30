interface StatusBannerProps {
  loading: boolean
  error: string | null
}

/** Shared loading/error banner every page renders while its `useApiData` call is in flight or
 * has failed (e.g. the API isn't running) -- keeps that boilerplate out of every page body. */
export function StatusBanner({ loading, error }: StatusBannerProps) {
  if (loading) {
    return <p className="text-sm text-neutral-500 dark:text-neutral-400">Loading…</p>
  }
  if (error) {
    return (
      <p className="rounded-lg border border-red-300 bg-red-50 p-3 text-sm text-red-700 dark:border-red-900 dark:bg-red-950/40 dark:text-red-300">
        Couldn't reach the API: {error}
      </p>
    )
  }
  return null
}
