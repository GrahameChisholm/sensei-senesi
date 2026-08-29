import { useEffect, useState } from "react";
import { MiniLeaguePanelOut } from "../api";
import { CaptainEoGrid } from "../components/CaptainEoGrid";
import { ExposureTable } from "../components/ExposureTable";
import { LeagueStandingsTable } from "../components/LeagueStandingsTable";
import { LeagueTemplateXi } from "../components/LeagueTemplateXi";
import { LeagueWeekSummary } from "../components/LeagueWeekSummary";
import { MiniLeagueHeader } from "../components/MiniLeagueHeader";
import { MiniLeagueSetup } from "../components/MiniLeagueSetup";
import { RivalHeadToHead } from "../components/RivalHeadToHead";
import { useStoredState } from "../hooks/useStoredState";
import { useSquad } from "../hooks/useSquad";
import { useMiniLeaguePanel, useMiniLeagueSettings } from "../hooks/useMiniLeague";
import { usePlayerDirectory, useTeams } from "../hooks/useProjections";

/** The rival directly above you in the standings by default -- a chasing manager's most relevant
 * comparison. Falls back to whoever is directly below (the manager chasing *you*) if you're
 * already top, and to null if there are no rivals at all. */
function defaultTargetRivalId(panel: MiniLeaguePanelOut): number | null {
  const above = panel.rivals
    .filter((rival) => rival.rank < panel.my_rank)
    .sort((a, b) => b.rank - a.rank)[0];
  if (above) return above.entry_id;
  const closest = [...panel.rivals].sort((a, b) => a.rank - b.rank)[0];
  return closest ? closest.entry_id : null;
}

export function MiniLeague() {
  const { settings, loading: settingsLoading, save } = useMiniLeagueSettings();
  const [selectedLeagueId, setSelectedLeagueId] = useState<number | null>(null);

  useEffect(() => {
    if (settings && settings.mini_league_ids.length > 0 && selectedLeagueId === null) {
      setSelectedLeagueId(settings.mini_league_ids[0]);
    }
  }, [settings, selectedLeagueId]);

  const { panel, loading: panelLoading, error, refresh } = useMiniLeaguePanel(
    selectedLeagueId,
    null,
  );
  // Persisted per league (item 7): a target rival's entry ID from one league is meaningless in
  // another, so switching leagues must never restore a stale ID from the previous one.
  const [targetRivalId, setTargetRivalId] = useStoredState<number | null>(
    `mini-league.targetRival.${selectedLeagueId ?? "none"}`,
    null,
  );

  useEffect(() => {
    if (!panel) return;
    const stillExists = panel.rivals.some((rival) => rival.entry_id === targetRivalId);
    if (targetRivalId === null || !stillExists) {
      setTargetRivalId(defaultTargetRivalId(panel));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [panel]);

  const { squad } = useSquad();
  const teams = useTeams();
  const [directory] = usePlayerDirectory();

  if (settingsLoading) {
    return <p className="loading">Loading…</p>;
  }

  if (!settings || !settings.fpl_team_id || settings.mini_league_ids.length === 0) {
    return (
      <div className="team-selection">
        <h2>Mini League</h2>
        <MiniLeagueSetup
          fplTeamId={settings?.fpl_team_id ?? null}
          miniLeagueIds={settings?.mini_league_ids ?? []}
          onSave={save}
        />
      </div>
    );
  }

  return (
    <div className="team-selection">
      <h2>Mini League</h2>

      {settings.mini_league_ids.length > 1 && (
        <div className="stats-team-picker" style={{ marginBottom: "0.85rem" }}>
          {settings.mini_league_ids.map((leagueId) => (
            <button
              key={leagueId}
              className={leagueId === selectedLeagueId ? "active" : ""}
              onClick={() => setSelectedLeagueId(leagueId)}
            >
              League {leagueId}
            </button>
          ))}
        </div>
      )}

      {panelLoading ? (
        <p className="loading">Loading…</p>
      ) : error ? (
        <div className="differentials-empty">{error}</div>
      ) : panel === null ? null : panel.rivals.length === 0 ? (
        <div className="differentials-empty">
          Nobody else in this league has a fetchable squad yet -- most likely nothing has been
          played this season, so there is no field to measure yourself against.
        </div>
      ) : (
        <>
          {panel.picks_gameweek < panel.gameweek && (
            <p className="transfer-hint">
              Rival squads are as of GW{panel.picks_gameweek}. GW{panel.gameweek} picks go public
              once this gameweek's deadline passes.
            </p>
          )}

          <MiniLeagueHeader
            panel={panel}
            targetRivalId={targetRivalId}
            onTargetRivalChange={setTargetRivalId}
            onRefresh={refresh}
          />

          <LeagueWeekSummary insights={panel.insights} directory={directory} />

          {squad && (
            <ExposureTable
              rows={panel.exposures}
              directory={directory}
              teams={teams}
              ownedPlayerIds={new Set(squad.squad.map((p) => p.player_id))}
            />
          )}

          <div className="main-content" style={{ marginTop: "1rem" }}>
            <RivalHeadToHead
              rival={panel.rivals.find((rival) => rival.entry_id === targetRivalId) ?? null}
              directory={directory}
            />
            <LeagueStandingsTable
              panel={panel}
              selectedRivalId={targetRivalId}
              onSelectRival={setTargetRivalId}
            />
          </div>

          <CaptainEoGrid
            options={panel.captain_options}
            directory={directory}
            currentCaptainId={squad?.captain_id ?? null}
          />

          {squad && (
            <LeagueTemplateXi
              templateIds={panel.template_xi}
              ownedPlayerIds={new Set(squad.squad.map((p) => p.player_id))}
              directory={directory}
              teams={teams}
            />
          )}
        </>
      )}
    </div>
  );
}
