import { useEffect, useState } from "react";
import { useGameweek, useTeams } from "../hooks/useProjections";
import { useFixtureSwing } from "../hooks/useFixtureSwing";
import { FixturesTicker } from "../components/FixturesTicker";
import { FixtureSwingTable } from "../components/FixtureSwingTable";

// Standard Premier League season length -- the upper bound on how far out either window's
// sliders can reach.
const MAX_GAMEWEEK = 38;

// Matches the locked-in defaults on the backend (features.fixture_swing.DEFAULT_NEAR_GAMEWEEKS/
// DEFAULT_FAR_GAMEWEEKS) -- used once, to seed the initial near/far ranges from the app's current
// gameweek, before the user adjusts either slider.
const DEFAULT_NEAR_SPAN = 3;
const DEFAULT_FAR_SPAN = 5;

export function FixturesPage() {
  const teams = useTeams();
  const [gameweek] = useGameweek();

  // One shared team selection, same "empty = show every team" convention PlayerStatsFilters' own
  // team picker uses -- both tables below filter down to just these teams, so picking a handful
  // makes comparing them side by side easier without the other ~15 teams in the way.
  const [selectedTeamIds, setSelectedTeamIds] = useState<Set<number>>(new Set());
  const teamList = Object.values(teams).sort((a, b) => a.short_name.localeCompare(b.short_name));

  function toggleTeam(teamId: number) {
    setSelectedTeamIds((prev) => {
      const next = new Set(prev);
      if (next.has(teamId)) next.delete(teamId);
      else next.add(teamId);
      return next;
    });
  }

  // Near/far gameweek ranges for the swing table -- each is independently adjustable (they don't
  // have to be adjacent, or even in order), so e.g. "GW5-7 vs GW2-8" is as valid as the default
  // "GW2-4 vs GW5-9". Seeded once from the app's current gameweek, mirroring
  // PlayerStats.tsx's own rangeInitialized pattern for its gameweek slider.
  const [nearFrom, setNearFrom] = useState<number | null>(null);
  const [nearTo, setNearTo] = useState<number | null>(null);
  const [farFrom, setFarFrom] = useState<number | null>(null);
  const [farTo, setFarTo] = useState<number | null>(null);
  const [rangeInitialized, setRangeInitialized] = useState(false);

  useEffect(() => {
    if (gameweek !== null && !rangeInitialized) {
      const seededNearTo = gameweek.gameweek + DEFAULT_NEAR_SPAN - 1;
      setNearFrom(gameweek.gameweek);
      setNearTo(seededNearTo);
      setFarFrom(seededNearTo + 1);
      setFarTo(seededNearTo + DEFAULT_FAR_SPAN);
      setRangeInitialized(true);
    }
  }, [gameweek, rangeInitialized]);

  const { rows, nearGameweeks, farGameweeks, loading } = useFixtureSwing(
    nearFrom ?? undefined,
    nearTo ?? undefined,
    farFrom ?? undefined,
    farTo ?? undefined,
  );

  const minGameweek = gameweek?.gameweek ?? 1;

  return (
    <div className="team-selection">
      <div className="stats-filters">
        <div className="stats-filters-row">
          <div className="stats-filter-group">
            <span className="stats-filter-label">Team</span>
            <div className="stats-team-picker">
              {teamList.map((team) => (
                <button
                  key={team.team_id}
                  className={selectedTeamIds.has(team.team_id) ? "active" : ""}
                  onClick={() => toggleTeam(team.team_id)}
                >
                  {team.short_name}
                </button>
              ))}
            </div>
          </div>
        </div>

        {rangeInitialized && nearFrom !== null && nearTo !== null && farFrom !== null && farTo !== null && (
          <div className="stats-filters-row">
            <label className="stats-range-label">
              Near gameweeks {nearFrom} – {nearTo}
              <span className="stats-range-inputs">
                <input
                  type="range"
                  min={minGameweek}
                  max={MAX_GAMEWEEK}
                  value={nearFrom}
                  onChange={(e) => setNearFrom(Math.min(Number(e.target.value), nearTo))}
                />
                <input
                  type="range"
                  min={minGameweek}
                  max={MAX_GAMEWEEK}
                  value={nearTo}
                  onChange={(e) => setNearTo(Math.max(Number(e.target.value), nearFrom))}
                />
              </span>
            </label>

            <label className="stats-range-label">
              Far gameweeks {farFrom} – {farTo}
              <span className="stats-range-inputs">
                <input
                  type="range"
                  min={minGameweek}
                  max={MAX_GAMEWEEK}
                  value={farFrom}
                  onChange={(e) => setFarFrom(Math.min(Number(e.target.value), farTo))}
                />
                <input
                  type="range"
                  min={minGameweek}
                  max={MAX_GAMEWEEK}
                  value={farTo}
                  onChange={(e) => setFarTo(Math.max(Number(e.target.value), farFrom))}
                />
              </span>
            </label>
          </div>
        )}
      </div>

      <FixturesTicker teams={teams} selectedTeamIds={selectedTeamIds} />
      {!loading && (
        <FixtureSwingTable
          rows={rows}
          teams={teams}
          nearGameweeks={nearGameweeks}
          farGameweeks={farGameweeks}
          selectedTeamIds={selectedTeamIds}
        />
      )}
    </div>
  );
}
