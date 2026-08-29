import { PlayerPanelRowOut, TeamOut } from "../api";
import { POSITION_ORDER } from "./Pitch";

interface LeagueTemplateXiProps {
  templateIds: number[];
  ownedPlayerIds: Set<number>;
  directory: Record<number, PlayerPanelRowOut>;
  teams: Record<number, TeamOut>;
}

/** The league's own highest-EO XI, laid out in formation (MINI_LEAGUE_PLAN M13/M22 zone 6) --
 * players you own in the normal accent colour, players you're missing in red. A glance at how
 * much red is on this pitch is the fastest read on the whole page: it says how correlated you are
 * with the field before a single number has been read.
 *
 * A lighter-weight sibling of the Team page's own <Pitch>, not a reuse of it -- <Pitch> renders a
 * caller's actual SquadOut (captain/vice/bench, remove/swap actions), which doesn't fit a set of
 * player IDs most of which the manager doesn't own at all. */
export function LeagueTemplateXi({
  templateIds,
  ownedPlayerIds,
  directory,
  teams,
}: LeagueTemplateXiProps) {
  const owned = templateIds.filter((id) => ownedPlayerIds.has(id));
  const byPosition: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const playerId of templateIds) {
    const position = directory[playerId]?.position;
    if (position && byPosition[position]) byPosition[position].push(playerId);
  }

  return (
    <div className="fixture-swing">
      <div className="stats-filters-row" style={{ marginBottom: "0.5rem" }}>
        <p className="stats-filter-label">The league's template XI</p>
        <p className="stats-row-count">
          You own {owned.length} of {templateIds.length}
        </p>
      </div>
      <div className="pitch">
        {POSITION_ORDER.map((position) => (
          <div className="pitch-row" key={position}>
            {byPosition[position].map((playerId) => {
              const player = directory[playerId];
              const isOwned = ownedPlayerIds.has(playerId);
              return (
                <div
                  key={playerId}
                  className="player-card"
                  style={
                    isOwned
                      ? undefined
                      : { borderColor: "var(--danger)", boxShadow: "0 0 0 2px var(--danger-soft)" }
                  }
                >
                  <div className="player-card-name">
                    <span className="player-card-name-text">
                      {player?.name ?? `#${playerId}`}
                    </span>
                  </div>
                  <div className="player-card-price">
                    {player?.team_id !== null && player?.team_id !== undefined
                      ? teams[player.team_id]?.short_name
                      : ""}
                  </div>
                  <div style={{ fontSize: "0.65rem", color: isOwned ? "var(--accent)" : "var(--danger)" }}>
                    {isOwned ? "Owned" : "Missing"}
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}
