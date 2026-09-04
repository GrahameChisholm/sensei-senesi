import { TeamOut } from "../api";
import { DualRangeSlider } from "./DualRangeSlider";

const POSITIONS = ["GK", "DEF", "MID", "FWD"];

interface PlayerStatsFiltersProps {
  teams: Record<number, TeamOut>;
  search: string;
  onSearchChange: (value: string) => void;
  selectedTeamIds: Set<number>;
  onToggleTeam: (teamId: number) => void;
  selectedPositions: Set<string>;
  onTogglePosition: (position: string) => void;
  minPrice: number;
  maxPrice: number;
  onPriceChange: (minPrice: number, maxPrice: number) => void;
  gameweekFrom: number;
  gameweekTo: number;
  maxGameweek: number;
  onGameweekRangeChange: (from: number, to: number) => void;
  perNinety: boolean;
  onPerNinetyChange: (value: boolean) => void;
}

const PRICE_FLOOR = 40; // £4.0m
const PRICE_CEILING = 155; // £15.5m

export function PlayerStatsFilters({
  teams,
  search,
  onSearchChange,
  selectedTeamIds,
  onToggleTeam,
  selectedPositions,
  onTogglePosition,
  minPrice,
  maxPrice,
  onPriceChange,
  gameweekFrom,
  gameweekTo,
  maxGameweek,
  onGameweekRangeChange,
  perNinety,
  onPerNinetyChange,
}: PlayerStatsFiltersProps) {
  const teamList = Object.values(teams).sort((a, b) => a.short_name.localeCompare(b.short_name));

  return (
    <div className="stats-filters">
      <div className="stats-filters-row">
        <input
          type="search"
          placeholder="Search by player or team name"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
        />

        <label>
          Per 90
          <input
            type="checkbox"
            checked={perNinety}
            onChange={(e) => onPerNinetyChange(e.target.checked)}
          />
        </label>
      </div>

      <div className="stats-filters-row">
        <div className="stats-filter-group">
          <span className="stats-filter-label">Position</span>
          <div className="position-tabs">
            {POSITIONS.map((position) => (
              <button
                key={position}
                className={selectedPositions.has(position) ? "active" : ""}
                onClick={() => onTogglePosition(position)}
              >
                {position}
              </button>
            ))}
          </div>
        </div>

        <div className="stats-filter-group">
          <span className="stats-filter-label">Team</span>
          <div className="stats-team-picker">
            {teamList.map((team) => (
              <button
                key={team.team_id}
                className={selectedTeamIds.has(team.team_id) ? "active" : ""}
                onClick={() => onToggleTeam(team.team_id)}
              >
                {team.short_name}
              </button>
            ))}
          </div>
        </div>
      </div>

      <div className="stats-filters-row">
        <label className="stats-range-label">
          Price £{(minPrice / 10).toFixed(1)}m – £{(maxPrice / 10).toFixed(1)}m
          <DualRangeSlider
            min={PRICE_FLOOR}
            max={PRICE_CEILING}
            step={5}
            valueMin={minPrice}
            valueMax={maxPrice}
            onChange={onPriceChange}
            ariaLabelMin="Minimum price"
            ariaLabelMax="Maximum price"
          />
        </label>

        <label className="stats-range-label">
          Gameweeks {gameweekFrom} – {gameweekTo}
          <DualRangeSlider
            min={1}
            max={maxGameweek}
            step={1}
            valueMin={gameweekFrom}
            valueMax={gameweekTo}
            onChange={onGameweekRangeChange}
            ariaLabelMin="From gameweek"
            ariaLabelMax="To gameweek"
          />
        </label>
      </div>
    </div>
  );
}
