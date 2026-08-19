import { useTeams } from "../hooks/useProjections";
import { FixturesTicker } from "../components/FixturesTicker";

export function FixturesPage() {
  const teams = useTeams();
  return (
    <div className="team-selection">
      <FixturesTicker teams={teams} />
    </div>
  );
}
