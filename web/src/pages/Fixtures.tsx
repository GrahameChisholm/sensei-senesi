import { useState } from 'react'
import { api } from '../api'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

/** Sortable difficulty ticker (BUILD_PLAN 5.2), built from the engine's own opponent xG/xGA
 * rather than FPL's colour scale (features/fixtures.py). Attack and defense ratings are shown
 * separately since a fixture can be easy for one and hard for the other. */
export function Fixtures() {
  const [gameweek, setGameweek] = useState<number | undefined>(undefined)
  const { data, error, loading } = useApiData(() => api.fixtures(gameweek), [gameweek])
  const [sortKey, setSortKey] = useState<'attack_rating' | 'defense_rating' | 'overall_rating'>(
    'overall_rating',
  )

  const rows = data ? [...data].sort((a, b) => a[sortKey] - b[sortKey]) : []

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <div className="mb-6 flex items-center justify-between">
        <h1 className="text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
          Fixtures
        </h1>
        <label className="text-sm text-neutral-600 dark:text-neutral-300">
          Gameweek{' '}
          <input
            type="number"
            min={1}
            placeholder="all"
            onChange={(e) => setGameweek(e.target.value ? Number(e.target.value) : undefined)}
            className="w-20 rounded border border-neutral-300 px-2 py-1 dark:border-neutral-700 dark:bg-neutral-900"
          />
        </label>
      </div>
      <StatusBanner loading={loading} error={error} />
      {data && (
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-neutral-200 text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
              <th className="py-2 pr-4">Team</th>
              <th className="py-2 pr-4">Opponent</th>
              <th className="py-2 pr-4">GW</th>
              <th className="py-2 pr-4">Venue</th>
              <th
                className="cursor-pointer py-2 pr-4"
                onClick={() => setSortKey('attack_rating')}
              >
                Attack (1 easy-5 hard)
              </th>
              <th
                className="cursor-pointer py-2 pr-4"
                onClick={() => setSortKey('defense_rating')}
              >
                Defense (1 easy-5 hard)
              </th>
              <th
                className="cursor-pointer py-2 pr-4"
                onClick={() => setSortKey('overall_rating')}
              >
                Overall
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr
                key={`${row.team_id}-${row.gameweek}-${i}`}
                className="border-b border-neutral-100 dark:border-neutral-900"
              >
                <td className="py-2 pr-4">Team {row.team_id}</td>
                <td className="py-2 pr-4">Team {row.opponent_id}</td>
                <td className="py-2 pr-4">{row.gameweek}</td>
                <td className="py-2 pr-4">{row.is_home ? 'Home' : 'Away'}</td>
                <td className="py-2 pr-4">{row.attack_rating}</td>
                <td className="py-2 pr-4">{row.defense_rating}</td>
                <td className="py-2 pr-4">{row.overall_rating.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}
