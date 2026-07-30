import { useState } from 'react'
import type { CaptaincyOption } from '../api'
import { api } from '../api'
import { DEFAULT_GAMEWEEK } from '../config'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

function PickCallout({ label, option }: { label: string; option: CaptaincyOption | null }) {
  return (
    <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
      <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
        {label}
      </h3>
      {option ? (
        <>
          <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
            #{option.player_id} ({option.position})
          </p>
          <p className="text-sm text-neutral-600 dark:text-neutral-300">{option.reasoning}</p>
        </>
      ) : (
        <p className="text-sm text-neutral-500 dark:text-neutral-400">No eligible player</p>
      )}
    </div>
  )
}

/** Ranks the full player pool by EV, with the three side-by-side picks (BUILD_PLAN 5.2: no
 * single headline recommendation) -- owned/eligible players are badged within the full
 * ranking, not filtered down to just the user's 15. */
export function Captaincy() {
  const [gameweek, setGameweek] = useState(DEFAULT_GAMEWEEK)
  const { data, error, loading } = useApiData(() => api.captaincy(gameweek), [gameweek])

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
          Captaincy
        </h1>
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
      {data && (
        <>
          <div className="mb-6 grid grid-cols-1 gap-4 sm:grid-cols-3">
            <PickCallout label="Highest EV" option={data.top_ev_pick} />
            <PickCallout label="Safe (floor)" option={data.safe_pick} />
            <PickCallout label="Punt (ceiling)" option={data.punt_pick} />
          </div>
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-neutral-200 text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
                <th className="py-2 pr-4">Player</th>
                <th className="py-2 pr-4">Pos</th>
                <th className="py-2 pr-4">EV</th>
                <th className="py-2 pr-4">Floor</th>
                <th className="py-2 pr-4">Ceiling</th>
                <th className="py-2 pr-4">Owned</th>
                <th className="py-2 pr-4">Eligible</th>
              </tr>
            </thead>
            <tbody>
              {data.ranked_pool.slice(0, 25).map((option) => (
                <tr
                  key={option.player_id}
                  className="border-b border-neutral-100 dark:border-neutral-900"
                  title={option.reasoning}
                >
                  <td className="py-2 pr-4">#{option.player_id}</td>
                  <td className="py-2 pr-4">{option.position}</td>
                  <td className="py-2 pr-4">{option.expected_points.toFixed(1)}</td>
                  <td className="py-2 pr-4">{option.floor?.toFixed(1) ?? '—'}</td>
                  <td className="py-2 pr-4">{option.ceiling?.toFixed(1) ?? '—'}</td>
                  <td className="py-2 pr-4">{option.is_owned ? 'Yes' : ''}</td>
                  <td className="py-2 pr-4">{option.is_eligible ? 'Yes' : ''}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="mt-2 text-xs text-neutral-500 dark:text-neutral-400">
            Showing top 25 of {data.ranked_pool.length} by expected points. Hover a row for its
            full reasoning.
          </p>
        </>
      )}
    </div>
  )
}
