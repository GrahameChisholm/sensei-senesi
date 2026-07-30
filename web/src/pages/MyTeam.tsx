import { api } from '../api'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

function formatMoney(tenths: number): string {
  return `£${(tenths / 10).toFixed(1)}m`
}

/** The shared-state view (BUILD_PLAN 5.2): current 15, bank, sell prices, free transfers, and
 * chips remaining -- the same `MyTeamState` every other decision page reads from. */
export function MyTeam() {
  const { data, error, loading } = useApiData(() => api.team(), [])

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        My Team
      </h1>
      <StatusBanner loading={loading} error={error} />
      {data && (
        <>
          <div className="mb-6 grid grid-cols-2 gap-4 sm:grid-cols-4">
            <Stat label="Bank" value={formatMoney(data.bank)} />
            <Stat label="Squad sell value" value={formatMoney(data.total_sell_value)} />
            <Stat label="Free transfers" value={String(data.free_transfers)} />
            <Stat label="Chips remaining" value={data.chips_remaining.join(', ') || 'None'} />
          </div>
          <table className="w-full border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-neutral-200 text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
                <th className="py-2 pr-4">Player</th>
                <th className="py-2 pr-4">Position</th>
                <th className="py-2 pr-4">Role</th>
                <th className="py-2 pr-4">Purchase</th>
                <th className="py-2 pr-4">Current</th>
                <th className="py-2 pr-4">Sell price</th>
              </tr>
            </thead>
            <tbody>
              {data.squad.map((player) => (
                <tr
                  key={player.player_id}
                  className="border-b border-neutral-100 dark:border-neutral-900"
                >
                  <td className="py-2 pr-4">#{player.player_id}</td>
                  <td className="py-2 pr-4">{player.position}</td>
                  <td className="py-2 pr-4 text-neutral-500 dark:text-neutral-400">
                    {player.player_id === data.captain_id
                      ? 'Captain'
                      : player.player_id === data.vice_captain_id
                        ? 'Vice-captain'
                        : data.starting_xi.includes(player.player_id)
                          ? 'Starting XI'
                          : 'Bench'}
                  </td>
                  <td className="py-2 pr-4">{formatMoney(player.purchase_price)}</td>
                  <td className="py-2 pr-4">{formatMoney(player.current_price)}</td>
                  <td className="py-2 pr-4">{formatMoney(player.sell_price)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </div>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-lg border border-neutral-200 p-3 dark:border-neutral-800">
      <p className="text-xs text-neutral-500 dark:text-neutral-400">{label}</p>
      <p className="text-base font-semibold text-neutral-900 dark:text-neutral-50">{value}</p>
    </div>
  )
}
