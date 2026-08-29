import { useState } from "react";
import { OwnershipLensSource } from "../api";
import { DIFFERENTIAL_PRESETS, DifferentialIntent } from "../lib/differentialPresets";

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
  intent: DifferentialIntent;
  onIntentChange: (intent: DifferentialIntent) => void;
  windowGameweeks: number;
  onWindowChange: (value: number) => void;
  ownershipLens: OwnershipLensSource;
  maxOwnership: number;
  onMaxOwnershipChange: (value: number) => void;
  maxLeagueOwners: number | undefined;
  onMaxLeagueOwnersChange: (value: number) => void;
  hideOwned: boolean;
  onHideOwnedChange: (value: boolean) => void;
}

/** Three intent presets (item 2) replace tuning the window/ownership-ceiling/hide-owned knobs by
 * hand on every visit -- each bundles values tuned for one question ("what can I attack the
 * leader with," "what survives a wider check," "what's simply best"). The raw knobs still exist
 * behind "Advanced" for whenever a preset isn't quite right; touching any of them marks the
 * current state "Custom" rather than silently contradicting whichever preset pill is lit.
 *
 * The ownership filter itself still changes shape by lens (MINI_LEAGUE_PLAN M25): a percentage
 * slider makes sense over the whole FPL player base, but over a mini-league's handful of rivals
 * it's an illusion of precision -- the only reachable values are multiples of 1/n_rivals. Under
 * the league lens the control becomes a plain "owned by at most N rivals" integer stepper
 * instead, which is also the truer question a mini-league manager is actually asking. */
export function DifferentialsFilters({
  intent,
  onIntentChange,
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
  const [advancedOpen, setAdvancedOpen] = useState(intent === "custom");

  return (
    <div className="stats-filters">
      <div className="stats-filters-row">
        <div className="stats-team-picker">
          {DIFFERENTIAL_PRESETS.map((preset) => (
            <button
              key={preset.key}
              className={intent === preset.key ? "active" : ""}
              onClick={() => onIntentChange(preset.key)}
              title={preset.description}
            >
              {preset.label}
            </button>
          ))}
        </div>
        {intent === "custom" && <span className="stats-row-count">Custom</span>}
        <button type="button" className="link-button" onClick={() => setAdvancedOpen((prev) => !prev)}>
          {advancedOpen ? "Hide advanced" : "Advanced"}
        </button>
      </div>

      {advancedOpen && (
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
              Owned by at most {maxLeagueOwners ?? MAX_LEAGUE_OWNERS_CEILING} rivals
              <CountStepper
                value={maxLeagueOwners ?? MAX_LEAGUE_OWNERS_CEILING}
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
      )}
    </div>
  );
}
