import { useState } from 'react'
import type { ChipEvaluation, WildcardEvaluation } from '../api'
import { api } from '../api'
import { DEFAULT_GAMEWEEK } from '../config'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

function ChipCard({
  title,
  evaluation,
}: {
  title: string
  evaluation: ChipEvaluation | WildcardEvaluation | null
}) {
  const recommendation = evaluation?.recommendation
  const playNow = recommendation === 'play_now'
  return (
    <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
      <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
        {title}
      </h3>
      <p
        className={`text-lg font-semibold ${
          playNow
            ? 'text-emerald-600 dark:text-emerald-400'
            : 'text-neutral-900 dark:text-neutral-50'
        }`}
      >
        {evaluation ? (playNow ? 'Play now' : recommendation === 'wait' ? 'Wait' : 'Hold') : '—'}
      </p>
      {evaluation && (
        <p className="text-sm text-neutral-600 dark:text-neutral-300">{evaluation.reasoning}</p>
      )}
    </div>
  )
}

/** Per-chip value-now-vs-waiting verdicts (BUILD_PLAN 5.2/4). Bench Boost, Triple Captain, and
 * Free Hit share the gameweek selector (they're all "which week in the horizon" questions);
 * Wildcard is a standalone snapshot since its effect isn't tied to one target gameweek. */
export function Chips() {
  const [gameweek, setGameweek] = useState(DEFAULT_GAMEWEEK)
  const benchBoost = useApiData(() => api.benchBoost(gameweek), [gameweek])
  const tripleCaptain = useApiData(() => api.tripleCaptain(gameweek), [gameweek])
  const freeHit = useApiData(() => api.freeHit(gameweek), [gameweek])
  const wildcard = useApiData(() => api.wildcard(), [])

  const loading = benchBoost.loading || tripleCaptain.loading || wildcard.loading
  const error = benchBoost.error ?? tripleCaptain.error ?? wildcard.error

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">Chips</h1>
        <label className="text-sm text-neutral-600 dark:text-neutral-300">
          Gameweek{' '}
          <input
            type="number"
            value={gameweek}
            min={1}
            onChange={(e) => setGameweek(Number(e.target.value) || 1)}
            className="w-16 rounded border border-neutral-300 px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
      </div>
      <StatusBanner loading={loading} error={error} />
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <ChipCard title="Bench Boost" evaluation={benchBoost.data} />
        <ChipCard title="Triple Captain" evaluation={tripleCaptain.data} />
        <ChipCard
          title="Free Hit"
          evaluation={freeHit.error ? null : freeHit.data}
        />
        <ChipCard title="Wildcard" evaluation={wildcard.data} />
      </div>
      {freeHit.error && (
        <p className="mt-4 text-sm text-neutral-500 dark:text-neutral-400">
          Free Hit: {freeHit.error}
        </p>
      )}
    </div>
  )
}
