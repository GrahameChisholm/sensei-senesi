import { useState } from 'react'
import { api } from '../api'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

const POSITIONS = ['GK', 'DEF', 'MID', 'FWD'] as const

/** Full-pool player search/comparison (BUILD_PLAN 5.2): filter by name/position/price, click a
 * row for the full component breakdown (2.7) and simulation summary -- the same data every other
 * screen already surfaces, just addressable per-player rather than only within one feature's
 * own ranking. */
export function PlayerSearch() {
  const [search, setSearch] = useState('')
  const [position, setPosition] = useState('')
  const [maxPrice, setMaxPrice] = useState<number | undefined>(undefined)
  const [selectedId, setSelectedId] = useState<number | null>(null)

  const { data, error, loading } = useApiData(
    () => api.players({ search: search || undefined, position: position || undefined, maxPrice }),
    [search, position, maxPrice],
  )
  const detail = useApiData(
    () => (selectedId !== null ? api.player(selectedId) : Promise.resolve(null)),
    [selectedId],
  )

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        Player Search
      </h1>
      <div className="mb-6 flex flex-wrap gap-3">
        <input
          type="text"
          placeholder="Search by name"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="rounded border border-neutral-300 px-3 py-1.5 text-sm dark:border-neutral-700 dark:bg-neutral-900"
        />
        <select
          value={position}
          onChange={(e) => setPosition(e.target.value)}
          className="rounded border border-neutral-300 px-3 py-1.5 text-sm dark:border-neutral-700 dark:bg-neutral-900"
        >
          <option value="">All positions</option>
          {POSITIONS.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <input
          type="number"
          placeholder="Max price"
          value={maxPrice ?? ''}
          onChange={(e) => setMaxPrice(e.target.value ? Number(e.target.value) : undefined)}
          className="w-32 rounded border border-neutral-300 px-3 py-1.5 text-sm dark:border-neutral-700 dark:bg-neutral-900"
        />
      </div>
      <StatusBanner loading={loading} error={error} />
      {data && (
        <table className="w-full border-collapse text-left text-sm">
          <thead>
            <tr className="border-b border-neutral-200 text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
              <th className="py-2 pr-4">Name</th>
              <th className="py-2 pr-4">Position</th>
              <th className="py-2 pr-4">Price</th>
              <th className="py-2 pr-4">GW</th>
              <th className="py-2 pr-4">Expected points</th>
            </tr>
          </thead>
          <tbody>
            {data.map((player) => (
              <tr
                key={player.player_id}
                onClick={() => setSelectedId(player.player_id)}
                className="cursor-pointer border-b border-neutral-100 hover:bg-neutral-50 dark:border-neutral-900 dark:hover:bg-neutral-900"
              >
                <td className="py-2 pr-4">{player.name}</td>
                <td className="py-2 pr-4">{player.position}</td>
                <td className="py-2 pr-4">
                  {player.price !== null ? `£${(player.price / 10).toFixed(1)}m` : '—'}
                </td>
                <td className="py-2 pr-4">{player.gameweek}</td>
                <td className="py-2 pr-4">{player.expected_points.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {selectedId !== null && (
        <div className="mt-6 rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
          <StatusBanner loading={detail.loading} error={detail.error} />
          {detail.data && (
            <>
              <h2 className="mb-2 text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                {detail.data.name} — GW{detail.data.gameweek}
              </h2>
              <p className="mb-3 text-sm text-neutral-600 dark:text-neutral-300">
                Expected points: {detail.data.expected_points.toFixed(2)}
                {detail.data.floor !== null && detail.data.ceiling !== null && (
                  <>
                    {' '}
                    &middot; floor {detail.data.floor.toFixed(1)}, ceiling{' '}
                    {detail.data.ceiling.toFixed(1)}
                    {detail.data.prob_big_haul !== null &&
                      ` · P(10+) ${(detail.data.prob_big_haul * 100).toFixed(0)}%`}
                  </>
                )}
              </p>
              <ul className="grid grid-cols-2 gap-x-6 gap-y-1 text-sm sm:grid-cols-3">
                {Object.entries(detail.data.breakdown).map(([name, value]) => (
                  <li key={name} className="flex justify-between">
                    <span className="text-neutral-500 capitalize dark:text-neutral-400">
                      {name.replace(/_/g, ' ')}
                    </span>
                    <span className="text-neutral-900 dark:text-neutral-50">
                      {value.toFixed(2)}
                    </span>
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
      )}
    </div>
  )
}
