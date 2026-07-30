import { NavLink } from 'react-router-dom'

const LINKS = [
  { to: '/', label: 'Dashboard' },
  { to: '/team', label: 'My Team' },
  { to: '/captaincy', label: 'Captaincy' },
  { to: '/transfers', label: 'Transfers' },
  { to: '/chips', label: 'Chips' },
  { to: '/fixtures', label: 'Fixtures' },
  { to: '/players', label: 'Player Search' },
  { to: '/performance', label: 'Model Performance' },
  { to: '/settings', label: 'Settings' },
]

export function Nav() {
  return (
    <nav className="border-b border-neutral-200 dark:border-neutral-800">
      <div className="mx-auto flex max-w-5xl flex-wrap items-center gap-1 px-4 py-3">
        <span className="mr-4 text-sm font-semibold text-neutral-900 dark:text-neutral-50">
          FPL Assistant
        </span>
        {LINKS.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            end={link.to === '/'}
            className={({ isActive }) =>
              `rounded-md px-2.5 py-1.5 text-sm ${
                isActive
                  ? 'bg-neutral-900 text-white dark:bg-neutral-100 dark:text-neutral-900'
                  : 'text-neutral-600 hover:bg-neutral-100 dark:text-neutral-300 dark:hover:bg-neutral-800'
              }`
            }
          >
            {link.label}
          </NavLink>
        ))}
      </div>
    </nav>
  )
}
