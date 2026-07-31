import { useEffect, useState } from 'react'
import { api, type Settings as SettingsData } from '../api'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

/** FPL team ID, mini-league IDs, and planning-horizon default (BUILD_PLAN 5.2) -- persisted via
 * GET/PUT /settings, independently of AppState so they survive a weekly refresh/restart. */
export function Settings() {
  const { data, error, loading } = useApiData(() => api.settings(), [])
  const [form, setForm] = useState<SettingsData | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => {
    if (data) setForm(data)
  }, [data])

  if (!form) {
    return (
      <div className="mx-auto max-w-2xl px-4 py-8">
        <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
          Settings
        </h1>
        <StatusBanner loading={loading} error={error} />
      </div>
    )
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!form) return
    setSaveError(null)
    setSaved(false)
    try {
      const updated = await api.updateSettings(form)
      setForm(updated)
      setSaved(true)
    } catch (err) {
      setSaveError(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        Settings
      </h1>
      <form onSubmit={handleSubmit} className="space-y-5">
        <label className="block">
          <span className="mb-1 block text-sm font-medium text-neutral-700 dark:text-neutral-300">
            FPL team ID
          </span>
          <input
            type="number"
            value={form.fpl_team_id ?? ''}
            onChange={(e) =>
              setForm({
                ...form,
                fpl_team_id: e.target.value ? Number(e.target.value) : null,
              })
            }
            placeholder="e.g. 1234567"
            className="w-full rounded border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-sm font-medium text-neutral-700 dark:text-neutral-300">
            Mini-league IDs (comma-separated)
          </span>
          <input
            type="text"
            value={form.mini_league_ids.join(', ')}
            onChange={(e) =>
              setForm({
                ...form,
                mini_league_ids: e.target.value
                  .split(',')
                  .map((part) => part.trim())
                  .filter(Boolean)
                  .map(Number)
                  .filter((n) => !Number.isNaN(n)),
              })
            }
            placeholder="e.g. 111, 222"
            className="w-full rounded border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-sm font-medium text-neutral-700 dark:text-neutral-300">
            Planning horizon (gameweeks)
          </span>
          <input
            type="number"
            min={1}
            max={10}
            value={form.planning_horizon_gameweeks}
            onChange={(e) =>
              setForm({ ...form, planning_horizon_gameweeks: Number(e.target.value) })
            }
            className="w-full rounded border border-neutral-300 px-3 py-2 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
        <button
          type="submit"
          className="rounded bg-neutral-900 px-4 py-2 text-sm font-medium text-white dark:bg-neutral-100 dark:text-neutral-900"
        >
          Save
        </button>
        {saved && <p className="text-sm text-green-600 dark:text-green-400">Saved.</p>}
        {saveError && <p className="text-sm text-red-600 dark:text-red-400">{saveError}</p>}
      </form>
    </div>
  )
}
