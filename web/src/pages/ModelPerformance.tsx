import { api } from '../api'
import { StatusBanner } from '../components/StatusBanner'
import { useApiData } from '../hooks/useApiData'

function GateRow({ label, ok }: { label: string; ok: boolean }) {
  return (
    <li className="flex items-center gap-2 text-sm">
      <span className={ok ? 'text-emerald-600 dark:text-emerald-400' : 'text-neutral-400'}>
        {ok ? '✓' : '—'}
      </span>
      <span className="text-neutral-700 dark:text-neutral-300">{label}</span>
    </li>
  )
}

/** Surfaces Phase 3's backtest metrics (BUILD_PLAN 5.2): prediction vs actual, captaincy hit-rate,
 * component calibration, and the Definition-of-Done gate checklist -- from the stored
 * season-backtest report (`backtest.run_season`'s CLI output), not recomputed in the browser.
 * Live accuracy (stats-only vs stats+market, once enough live gameweeks exist) isn't available
 * yet -- `has_live_accuracy` is honestly `false` until a real weekly refresh has run. */
export function ModelPerformance() {
  const { data, error, loading } = useApiData(() => api.modelPerformance(), [])

  return (
    <div className="mx-auto max-w-4xl px-4 py-8">
      <h1 className="mb-6 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        Model Performance
      </h1>
      <StatusBanner loading={loading} error={error} />
      {data && !data.headline && (
        <p className="text-sm text-neutral-500 dark:text-neutral-400">
          No backtest report has been generated yet. Run{' '}
          <code className="rounded bg-neutral-100 px-1 py-0.5 dark:bg-neutral-800">
            python -m backtest.run_season --report-path backtest/reports/2025-26.txt
          </code>{' '}
          to produce one.
        </p>
      )}
      {data?.headline && (
        <div className="space-y-8">
          <section className="grid grid-cols-2 gap-4 sm:grid-cols-4">
            <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
              <p className="text-xs tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
                Overall MAE
              </p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                {data.headline.overall_mae.toFixed(3)}
              </p>
            </div>
            <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
              <p className="text-xs tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
                Overall RMSE
              </p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                {data.headline.overall_rmse.toFixed(3)}
              </p>
            </div>
            <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
              <p className="text-xs tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
                Pooled Spearman
              </p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                {data.headline.pooled_spearman.toFixed(3)}
              </p>
            </div>
            <div className="rounded-xl border border-neutral-200 p-4 dark:border-neutral-800">
              <p className="text-xs tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
                Captaincy hit-rate
              </p>
              <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">
                {data.headline.captaincy_hit_rate !== null
                  ? `${(data.headline.captaincy_hit_rate * 100).toFixed(1)}%`
                  : '—'}
              </p>
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
              Top-N mean actual points
            </h2>
            <div className="flex gap-6">
              {Object.entries(data.headline.top_n_mean_actual).map(([n, value]) => (
                <div key={n} className="text-sm">
                  <span className="text-neutral-500 dark:text-neutral-400">Top {n}: </span>
                  <span className="font-medium text-neutral-900 dark:text-neutral-50">
                    {value.toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
              Component calibration (played rows only)
            </h2>
            <table className="w-full border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-neutral-200 text-neutral-500 dark:border-neutral-800 dark:text-neutral-400">
                  <th className="py-2 pr-4">Component</th>
                  <th className="py-2 pr-4">Predicted</th>
                  <th className="py-2 pr-4">Actual</th>
                  <th className="py-2 pr-4">Relative gap</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(data.headline.mean_calibrations_played).map(([name, cal]) => (
                  <tr key={name} className="border-b border-neutral-100 dark:border-neutral-900">
                    <td className="py-2 pr-4 capitalize">{name}</td>
                    <td className="py-2 pr-4">{cal.predicted.toFixed(4)}</td>
                    <td className="py-2 pr-4">{cal.actual.toFixed(4)}</td>
                    <td className="py-2 pr-4">{(cal.relative_gap * 100).toFixed(1)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
              Definition-of-Done gate
            </h2>
            <ul className="space-y-1">
              <GateRow label="Beats all baselines, statistically" ok={data.headline.gate.beats_baselines} />
              <GateRow
                label="No severe systematic bias in any position/price tier"
                ok={data.headline.gate.no_severe_bias}
              />
              <GateRow
                label="Each component reasonably calibrated"
                ok={data.headline.gate.calibration_acceptable}
              />
              <GateRow
                label="Predictions logged immutably, tagged by model version"
                ok={data.headline.gate.predictions_logged}
              />
              <GateRow label="Personally trusted enough to act on" ok={data.headline.gate.trusted_by_user} />
            </ul>
            <p className="mt-3 text-sm font-medium text-neutral-900 dark:text-neutral-50">
              {data.headline.gate.passed ? 'PASSED' : 'NOT YET'} — gate to Phase 5
            </p>
          </section>

          <section>
            <h2 className="mb-2 text-sm font-semibold text-neutral-700 dark:text-neutral-300">
              Live vs stats+market comparison
            </h2>
            <p className="text-sm text-neutral-500 dark:text-neutral-400">
              {data.has_live_accuracy
                ? 'Live accuracy available.'
                : 'Not enough live gameweeks have run yet -- this comparison appears once the weekly refresh has been operating for a while.'}
            </p>
          </section>
        </div>
      )}
    </div>
  )
}
