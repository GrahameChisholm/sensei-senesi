import { useState } from 'react'
import { Link } from 'react-router-dom'

interface VerdictCardProps {
  title: string
  verdict: string
  detail?: string | null
  tone?: 'default' | 'urgent'
  linkTo?: string
}

/** Dashboard's "verdict first, reasoning on click" pattern (BUILD_PLAN 5.2): the card always
 * shows its headline number/decision; the reasoning behind it only appears once the user asks
 * for it, keeping the home screen a compact summary rather than four expanded panels at once. */
export function VerdictCard({ title, verdict, detail, tone = 'default', linkTo }: VerdictCardProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <div
      className={`flex flex-col gap-2 rounded-xl border p-4 ${
        tone === 'urgent'
          ? 'border-amber-400/60 bg-amber-50 dark:bg-amber-950/30'
          : 'border-neutral-200 bg-white dark:border-neutral-800 dark:bg-neutral-900'
      }`}
    >
      <h3 className="text-xs font-semibold tracking-wide text-neutral-500 uppercase dark:text-neutral-400">
        {title}
      </h3>
      <p className="text-lg font-semibold text-neutral-900 dark:text-neutral-50">{verdict}</p>
      {detail && (
        <button
          type="button"
          className="cursor-pointer self-start text-sm text-neutral-500 underline decoration-dotted hover:text-neutral-700 dark:text-neutral-400 dark:hover:text-neutral-200"
          onClick={() => setExpanded((value) => !value)}
        >
          {expanded ? 'Hide reasoning' : 'Why?'}
        </button>
      )}
      {expanded && detail && (
        <p className="text-sm text-neutral-600 dark:text-neutral-300">{detail}</p>
      )}
      {linkTo && (
        <Link
          to={linkTo}
          className="mt-auto text-sm text-blue-600 hover:underline dark:text-blue-400"
        >
          View full breakdown &rarr;
        </Link>
      )}
    </div>
  )
}
