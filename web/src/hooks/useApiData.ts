import { useEffect, useState } from 'react'

interface ApiDataState<T> {
  data: T | null
  error: string | null
  loading: boolean
}

/** Runs `fetcher` whenever `deps` changes, tracking loading/error/data -- every page uses this
 * instead of hand-rolling its own effect, so the loading/error UI stays consistent everywhere. */
export function useApiData<T>(fetcher: () => Promise<T>, deps: unknown[]): ApiDataState<T> {
  const [state, setState] = useState<ApiDataState<T>>({ data: null, error: null, loading: true })

  useEffect(() => {
    let cancelled = false
    setState({ data: null, error: null, loading: true })
    fetcher()
      .then((data) => {
        if (!cancelled) setState({ data, error: null, loading: false })
      })
      .catch((err: unknown) => {
        if (!cancelled) {
          const message = err instanceof Error ? err.message : String(err)
          setState({ data: null, error: message, loading: false })
        }
      })
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  return state
}
