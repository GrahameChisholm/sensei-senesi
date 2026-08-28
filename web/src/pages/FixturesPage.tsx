import { useEffect, useState } from "react";
import { useGameweek, useTeams } from "../hooks/useProjections";
import { useFixtureSwing } from "../hooks/useFixtureSwing";
import { FixturesTicker } from "../components/FixturesTicker";
import { FixtureSwingTable } from "../components/FixtureSwingTable";
import { GameweekStepper } from "../components/GameweekStepper";

// Standard Premier League season length -- the upper bound on how far out any window's sliders
// can reach.
const MAX_GAMEWEEK = 38;

// Matches the locked-in defaults on the backend (features.fixture_swing.DEFAULT_NEAR_GAMEWEEKS/
// DEFAULT_FAR_GAMEWEEKS) -- used once, to seed the initial near/far ranges from the app's current
// gameweek, before the user adjusts either slider.
const DEFAULT_NEAR_SPAN = 3;
const DEFAULT_FAR_SPAN = 5;

// Matches api.fixtures_view.DEFAULT_FIXTURE_TICKER_HORIZON -- used once, to seed the ticker's own
// gameweek window from the app's current gameweek, before the user adjusts either bound.
const DEFAULT_TICKER_SPAN = 5;

export function FixturesPage() {
  const teams = useTeams();
  const [gameweek] = useGameweek();

  // Team selection and gameweek window both scope only the Fixtures ticker below (the "first
  // chart") -- same "empty = show every team" convention PlayerStatsFilters' own team picker
  // uses. The Fixture swing table has its own separate near/far windows and isn't affected by
  // either of these.
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

  // The ticker's own gameweek window -- an arbitrary GW-to-GW range (e.g. GW4-6), not tied to the
  // Fixture swing table's near/far windows below. Seeded once from the app's current gameweek.
  const [tickerFrom, setTickerFrom] = useState<number | null>(null);
  const [tickerTo, setTickerTo] = useState<number | null>(null);
  const [tickerRangeInitialized, setTickerRangeInitialized] = useState(false);

  useEffect(() => {
    if (gameweek !== null && !tickerRangeInitialized) {
      setTickerFrom(gameweek.gameweek);
      setTickerTo(gameweek.gameweek + DEFAULT_TICKER_SPAN - 1);
      setTickerRangeInitialized(true);
    }
  }, [gameweek, tickerRangeInitialized]);

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
        <p className="differentials-muted-header">Fixtures ticker filters</p>
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

          {tickerRangeInitialized && tickerFrom !== null && tickerTo !== null && (
            <label className="stats-range-label">
              Gameweeks
              <span className="stats-range-inputs">
                <GameweekStepper
                  value={tickerFrom}
                  min={minGameweek}
                  max={tickerTo}
                  onChange={setTickerFrom}
                />
                <GameweekStepper
                  value={tickerTo}
                  min={tickerFrom}
                  max={MAX_GAMEWEEK}
                  onChange={setTickerTo}
                />
              </span>
            </label>
          )}
        </div>
      </div>

      <FixturesTicker
        teams={teams}
        selectedTeamIds={selectedTeamIds}
        gameweekFrom={tickerFrom ?? undefined}
        gameweekTo={tickerTo ?? undefined}
      />
      {!loading && (
        <FixtureSwingTable
          rows={rows}
          teams={teams}
          nearGameweeks={nearGameweeks}
          farGameweeks={farGameweeks}
          nearFrom={nearFrom}
          nearTo={nearTo}
          farFrom={farFrom}
          farTo={farTo}
          onNearFromChange={setNearFrom}
          onNearToChange={setNearTo}
          onFarFromChange={setFarFrom}
          onFarToChange={setFarTo}
          minGameweek={minGameweek}
          maxGameweek={MAX_GAMEWEEK}
        />
      )}
    </div>
  );
}
