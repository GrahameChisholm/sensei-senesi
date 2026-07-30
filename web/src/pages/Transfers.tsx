import { api } from '../api'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

/** The multi-gameweek planner (BUILD_PLAN 5.2): every affordable one-swap candidate, ranked,
 * with the single recommended move (if any) called out first. */
export function Transfers() {
  const { data, error, loading } = useApiData(() => api.transfers(), [])

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        Transfers
      </h1>
      <StatusBanner loading={loading} error={error} />
      {data && (
        <>
          <div className="mb-6 rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
            <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
              Recommended
            </h3>
            {data.recommended ? (
              <>
                <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                  #{data.recommended.sell_player_id} &rarr; #{data.recommended.buy_player_id} (
                  {data.recommended.position})
                </p>
                <p className="text-sm text-neutral-600 dark:text-neutral-300">
                  {data.recommended.reasoning}
                </p>
              </>
            ) : (
              <p className="text-sm text-neutral-500 dark:text-neutral-400">
                Nothing beats holding your squad this week.
              </p>
            )}
          </div>
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-neutral-200 text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
                <th className="py-2 pr-4">Sell</th>
                <th className="py-2 pr-4">Buy</th>
                <th className="py-2 pr-4">Pos</th>
                <th className="py-2 pr-4">Net spend</th>
                <th className="py-2 pr-4">Pts gain</th>
                <th className="py-2 pr-4">Net (after hit)</th>
                <th className="py-2 pr-4">Forced</th>
              </tr>
            </thead>
            <tbody>
              {data.affordable_candidates.map((candidate) => (
                <tr
                  key={`${candidate.sell_player_id}-${candidate.buy_player_id}`}
                  className="border-b border-neutral-100 dark:border-neutral-900"
                  title={candidate.reasoning}
                >
                  <td className="py-2 pr-4">#{candidate.sell_player_id}</td>
                  <td className="py-2 pr-4">#{candidate.buy_player_id}</td>
                  <td className="py-2 pr-4">{candidate.position}</td>
                  <td className="py-2 pr-4">£{(candidate.net_spend / 10).toFixed(1)}m</td>
                  <td className="py-2 pr-4">{candidate.points_gain.toFixed(1)}</td>
                  <td className="py-2 pr-4">{candidate.net_points_gain.toFixed(1)}</td>
                  <td className="py-2 pr-4">{candidate.is_forced ? 'Yes' : ''}</td>
                </tr>
              ))}
              {data.affordable_candidates.length === 0 && (
                <tr>
                  <td colSpan={7} className="py-4 text-neutral-500 dark:text-neutral-400">
                    No affordable candidates found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}
