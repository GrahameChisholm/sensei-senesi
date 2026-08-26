const WINDOW_OPTIONS = [4, 6, 8, 10];

interface DifferentialsFiltersProps {
  windowGameweeks: number;
  onWindowChange: (value: number) => void;
  maxOwnership: number;
  onMaxOwnershipChange: (value: number) => void;
  hideOwned: boolean;
  onHideOwnedChange: (value: boolean) => void;
}

export function DifferentialsFilters({
  windowGameweeks,
  onWindowChange,
  maxOwnership,
  onMaxOwnershipChange,
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
