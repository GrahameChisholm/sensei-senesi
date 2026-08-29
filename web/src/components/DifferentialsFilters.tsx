import { OwnershipLensSource } from "../api";

const WINDOW_OPTIONS = [4, 6, 8, 10];
const MAX_LEAGUE_OWNERS_CEILING = 20;

/** A plain integer +/- stepper, reusing GameweekStepper's CSS classes without its hardcoded
 * "GW{value}" label -- this counts rivals, not gameweeks. */
function CountStepper({
  value,
  min,
  max,
  onChange,
}: {
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  return (
    <span className="gameweek-stepper">
      <button
        type="button"
        className="gameweek-stepper-button"
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
      >
        −
      </button>
      <span className="gameweek-stepper-value">{value}</span>
      <button
        type="button"
        className="gameweek-stepper-button"
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
      >
        +
      </button>
    </span>
  );
}

interface DifferentialsFiltersProps {
  windowGameweeks: number;
  onWindowChange: (value: number) => void;
  ownershipLens: OwnershipLensSource;
  maxOwnership: number;
  onMaxOwnershipChange: (value: number) => void;
  maxLeagueOwners: number;
  onMaxLeagueOwnersChange: (value: number) => void;
  hideOwned: boolean;
  onHideOwnedChange: (value: boolean) => void;
}

/** The ownership filter changes shape by lens (MINI_LEAGUE_PLAN M25): a percentage slider makes
 * sense over the whole FPL player base, but over a mini-league's handful of rivals it's an
 * illusion of precision -- the only reachable values are multiples of 1/n_rivals. Under the
 * league lens the control becomes a plain "owned by at most N rivals" integer stepper instead,
 * which is also the truer question a mini-league manager is actually asking. */
export function DifferentialsFilters({
  windowGameweeks,
  onWindowChange,
  ownershipLens,
  maxOwnership,
  onMaxOwnershipChange,
  maxLeagueOwners,
  onMaxLeagueOwnersChange,
  hideOwned,
  onHideOwnedChange,
}: DifferentialsFiltersProps) {
  return (
    <div className="stats-filters">
      <div className="stats-filters-row">
        <label className="stats-range-label">
          Window
          <select value={windowGameweeks} onChange={(e) => onWindowChange(Number(e.target.value))}>
            {WINDOW_OPTIONS.map((value) => (
              <option key={value} value={value}>
                Last {value} GWs
              </option>
            ))}
          </select>
        </label>

        {ownershipLens === "league" ? (
          <label className="stats-range-label">
            Owned by at most {maxLeagueOwners} rivals
            <CountStepper
              value={maxLeagueOwners}
              min={0}
              max={MAX_LEAGUE_OWNERS_CEILING}
              onChange={onMaxLeagueOwnersChange}
            />
          </label>
        ) : (
          <label className="stats-range-label">
            Ownership under {maxOwnership.toFixed(0)}%
            <span className="stats-range-inputs">
              <input
                type="range"
                min={1}
                max={50}
                step={1}
                value={maxOwnership}
                onChange={(e) => onMaxOwnershipChange(Number(e.target.value))}
              />
            </span>
          </label>
        )}

        <label className="stats-filter-group">
          <input
            type="checkbox"
            checked={hideOwned}
            onChange={(e) => onHideOwnedChange(e.target.checked)}
          />
          Hide players already in my squad
        </label>
      </div>
    </div>
  );
}
