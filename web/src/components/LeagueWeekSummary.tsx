import { LeagueInsightOut, PlayerPanelRowOut } from "../api";
import { exposureTextColour } from "../lib/colours";

interface LeagueWeekSummaryProps {
  insights: LeagueInsightOut[];
  directory: Record<number, PlayerPanelRowOut>;
}

function playerName(directory: Record<number, PlayerPanelRowOut>, playerId: number): string {
  return directory[playerId]?.name ?? `#${playerId}`;
}

function sentenceFor(insight: LeagueInsightOut, directory: Record<number, PlayerPanelRowOut>): string {
  const name = playerName(directory, insight.player_id);
  const signedValue = `${insight.value >= 0 ? "+" : ""}${insight.value.toFixed(1)}`;
  switch (insight.kind) {
    case "edge":
      return `${name} is your edge. ${signedValue} expected swing, owned by ${insight.owner_count} of ${insight.n_rivals} rivals.`;
    case "drag":
      return `${name} is your biggest drag. ${signedValue} expected swing, owned by ${insight.owner_count} of ${insight.n_rivals} rivals you are missing.`;
    case "captain": {
      const referenceName =
        insight.reference_player_id !== null
          ? playerName(directory, insight.reference_player_id)
          : "your current captain";
      return `Captaining ${name} nets ${signedValue} over ${referenceName}, captained by ${insight.owner_count} of ${insight.n_rivals} rivals.`;
    }
    default:
      return "";
  }
}

/** The page's headline (item 1 of the Differentials/Mini League insight review): up to three
 * plain sentences derived from exactly the same exposures and captain options the tables below
 * already compute, so a manager gets the week's answer before scanning any table at all. At most
 * one sentence per kind (edge, drag, captain swap); any that has nothing to point at is simply
 * absent, per features.mini_league.summarise_week. */
export function LeagueWeekSummary({ insights, directory }: LeagueWeekSummaryProps) {
  return (
    <div className="player-panel">
      <h3>This week</h3>
      {insights.length === 0 ? (
        <p className="stats-row-count">No standout edge, drag, or captain swap this gameweek.</p>
      ) : (
        insights.map((insight) => (
          <p
            key={insight.kind}
            className="stats-row-count"
            style={{ color: exposureTextColour(insight.value), fontWeight: 600 }}
          >
            {sentenceFor(insight, directory)}
          </p>
        ))
      )}
    </div>
  );
}
