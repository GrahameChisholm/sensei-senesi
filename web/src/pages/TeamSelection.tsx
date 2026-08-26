import { useState } from "react";
import { useSquad } from "../hooks/useSquad";
import { useGameweek, usePlayerDirectory, useSquadPoints, useTeams } from "../hooks/useProjections";
import { GameweekHeader } from "../components/GameweekHeader";
import { ActiveChip, ChipBar } from "../components/ChipBar";
import { Pitch, QUOTA } from "../components/Pitch";
import { PlayerPanel } from "../components/PlayerPanel";
import { RuleViolationToast } from "../components/RuleViolationToast";

export function TeamSelection() {
  const squadState = useSquad();
  const [gameweek] = useGameweek();
  const teams = useTeams();
  const [directory] = usePlayerDirectory();

  const [horizon, setHorizon] = useState<"next" | "three">("next");
  const [activeChip, setActiveChip] = useState<ActiveChip>(null);

  const { squad, error, loading, clearError } = squadState;

  const points = useSquadPoints(
    activeChip,
    horizon === "three" ? 3 : 1,
    squad,
    squad?.is_complete ?? false,
  );

  if (loading || squad === null) {
    return <p className="loading">Loading…</p>;
  }

  const counts: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  for (const pick of squad.squad) counts[pick.position] = (counts[pick.position] ?? 0) + 1;
  const fillablePositions = Object.entries(QUOTA)
    .filter(([position, quota]) => (counts[position] ?? 0) < quota)
    .map(([position]) => position);

  async function handleAutoBuild() {
    await squadState.optimise(
      activeChip === "bench_boost" ? "full_squad" : "starting_xi",
      activeChip === "triple_captain" ? 3.0 : 2.0,
    );
  }

  async function handleClearSquad() {
    if (
      !window.confirm(
        "Clear your entire squad and start over with a fresh £100m budget? This can't be undone.",
      )
    ) {
      return;
    }
    await squadState.clearSquad();
  }

  return (
    <div className="team-selection">
      {error && <RuleViolationToast message={error} onDismiss={clearError} />}

      <GameweekHeader
        gameweek={gameweek}
        squad={squad}
        points={points}
        horizon={horizon}
        onHorizonChange={setHorizon}
        onAutoBuild={() => void handleAutoBuild()}
        onClearSquad={() => void handleClearSquad()}
        onImportSquad={(teamId) => void squadState.importSquad(teamId)}
      />

      <ChipBar activeChip={activeChip} onChange={setActiveChip} />

      <div className="main-content">
        <Pitch
          squad={squad}
          directory={directory}
          teams={teams}
          horizon={horizon}
          onRemove={(playerId) => void squadState.removePlayer(playerId)}
          onSetCaptain={(playerId, role) => void squadState.setCaptain(playerId, role)}
        />

        <PlayerPanel
          teams={teams}
          fillablePositions={fillablePositions}
          squadPlayerIds={squad.squad.map((p) => p.player_id)}
          affordableBudget={squad.budget_remaining}
          onAdd={(playerId, position, price) =>
            void squadState.addPlayer(playerId, position, price)
          }
        />
      </div>
    </div>
  );
}
