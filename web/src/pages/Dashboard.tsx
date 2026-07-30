import { api } from '../api'
import { DEFAULT_GAMEWEEK } from '../config'
import { VerdictCard } from '../components/VerdictCard'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

/** Home screen: a row of compact, equal-weight cards -- Captain, Transfers, Chips, plus a
 * lighter "urgent" card for injury/forced-sell flags (BUILD_PLAN 5.2). Each card is a preview:
 * verdict only, reasoning on click, full breakdown one link away. */
export function Dashboard() {
  const captaincy = useApiData(() => api.captaincy(DEFAULT_GAMEWEEK), [])
  const transfers = useApiData(() => api.transfers(), [])
  const benchBoost = useApiData(() => api.benchBoost(DEFAULT_GAMEWEEK), [])

  const loading = captaincy.loading || transfers.loading || benchBoost.loading
  const error = captaincy.error ?? transfers.error ?? benchBoost.error

  if (loading || error) {
    return (
      <div className="mx-auto max-w-5xl px-4 py-8">
        <StatusBanner loading={loading} error={error} />
      </div>
    )
  }

  const topPick = captaincy.data?.top_ev_pick
  const recommendedTransfer = transfers.data?.recommended
  const forcedTransfer = transfers.data?.affordable_candidates.find((c) => c.is_forced)

  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        Gameweek {DEFAULT_GAMEWEEK}
      </h1>
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <VerdictCard
          title="Captain"
          verdict={topPick ? `Player #${topPick.player_id}` : 'No eligible pick'}
          detail={topPick?.reasoning}
          linkTo="/captaincy"
        />
        <VerdictCard
          title="Transfers"
          verdict={
            recommendedTransfer
              ? `#${recommendedTransfer.sell_player_id} -> #${recommendedTransfer.buy_player_id}`
              : 'Hold your squad'
          }
          detail={recommendedTransfer?.reasoning ?? 'Nothing on the board beats staying put.'}
          linkTo="/transfers"
        />
        <VerdictCard
          title="Chips"
          verdict={
            benchBoost.data
              ? `Bench Boost: ${benchBoost.data.recommendation === 'play_now' ? 'play now' : 'wait'}`
              : 'Unavailable'
          }
          detail={benchBoost.data?.reasoning}
          linkTo="/chips"
        />
        <VerdictCard
          title="Urgent"
          tone={forcedTransfer ? 'urgent' : 'default'}
          verdict={
            forcedTransfer
              ? `Player #${forcedTransfer.sell_player_id} looks unlikely to feature`
              : 'No urgent issues'
          }
          detail={forcedTransfer?.reasoning}
          linkTo="/transfers"
        />
      </div>
    </div>
  )
}
