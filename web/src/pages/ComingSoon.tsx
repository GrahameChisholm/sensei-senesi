interface ComingSoonProps {
  title: string
  reason: string
}

/** Honest placeholder for a sitemap screen (BUILD_PLAN 5.2) that has no backend endpoint yet --
 * used instead of fabricating data the API doesn't serve. */
export function ComingSoon({ title, reason }: ComingSoonProps) {
  return (
    <div className="mx-auto max-w-5xl px-4 py-8">
      <h1 className="mb-2 text-2xl font-semibold text-neutral-900 dark:text-neutral-50">
        {title}
      </h1>
      <p className="text-sm text-neutral-500 dark:text-neutral-400">{reason}</p>
    </div>
  )
}
