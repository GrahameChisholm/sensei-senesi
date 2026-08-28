import { useTeams } from "../hooks/useProjections";
import { useFixtureSwing } from "../hooks/useFixtureSwing";
import { FixturesTicker } from "../components/FixturesTicker";
import { FixtureSwingTable } from "../components/FixtureSwingTable";

// Matches the locked-in default on the backend (features.fixture_swing.DEFAULT_NEAR_GAMEWEEKS/
// DEFAULT_FAR_GAMEWEEKS) -- passed explicitly here rather than omitted so the hook's dependency
// array has a stable value to key off, same as useFixtureTicker's own default horizon.
const NEAR_GAMEWEEKS = 3;
const FAR_GAMEWEEKS = 5;

export function FixturesPage() {
  const teams = useTeams();
  const { rows, nearGameweeks, farGameweeks, loading } = useFixtureSwing(
    NEAR_GAMEWEEKS,
    FAR_GAMEWEEKS,
  );
  return (
    <div className="team-selection">
      <FixturesTicker teams={teams} />
      {!loading && (
        <FixtureSwingTable
          rows={rows}
          teams={teams}
          nearGameweeks={nearGameweeks}
          farGameweeks={farGameweeks}
        />
      )}
    </div>
  );
}
