import { api } from '../api'
import { useApiData } from '../hooks/useApiData'

/** Shows whether the numbers on screen are real (and how stale) or the bundled demo dataset --
 * queries GET /data-status (BUILD_PLAN Phase 6/A6), fails silently since freshness is a nice-to-
 * know, not something that should block the rest of the app from rendering. */
export function DataFreshnessBanner() {
  const { data, error } = useApiData(() => api.dataStatus(), [])

  if (error || !data) return null

  if (data.is_demo_data) {
    return (
      <p className="bg-amber-50 px-4 py-1.5 text-center text-xs text-amber-800 dark:bg-amber-950/40 dark:text-amber-300">
        Showing demo data -- run the weekly refresh to see your real team.
      </p>
    )
  }

  const generatedAt = new Date(data.generated_at as string)
  const formatted = generatedAt.toLocaleString(undefined, {
    dateStyle: 'medium',
    timeStyle: 'short',
  })

  return (
    <p className="bg-neutral-100 px-4 py-1.5 text-center text-xs text-neutral-600 dark:bg-neutral-900 dark:text-neutral-400">
      Data as of {formatted}
    </p>
  )
}
