import { useState } from "react";
import { useSquad } from "../hooks/useSquad";
import { useGameweek, usePlayerDirectory, useSquadPoints, useTeams } from "../hooks/useProjections";
import { GameweekHeader } from "../components/GameweekHeader";
import { ActiveChip, ChipBar } from "../components/ChipBar";
import { Pitch, QUOTA } from "../components/Pitch";
import { PlayerPanel } from "../components/PlayerPanel";
import { RuleViolationToast } from "../components/RuleViolationToast";
import { TransferBanner } from "../components/TransferBanner";
import { useTransferSuggestion } from "../hooks/useTransferSuggestion";
import { useStoredState } from "../hooks/useStoredState";
import { TransferPlanOut } from "../api";

export function TeamSelection() {
  const squadState = useSquad();
  const [gameweek] = useGameweek();
  const teams = useTeams();
  const [directory] = usePlayerDirectory();

  const [horizon, setHorizon] = useState<"next" | "three">("next");
  const [activeChip, setActiveChip] = useState<ActiveChip>(null);
  const [viewGameweek, setViewGameweek] = useState<number | null>(null);
  const [swapSourceId, setSwapSourceId] = useState<number | null>(null);
  // Persisted: how many transfers a manager plans in a week is a standing preference, not a
  // per-visit one, and re-picking it on every page load would be busywork.
  const [transfers, setTransfers] = useStoredState<number>("team.transfers", 1);
  const [applying, setApplying] = useState(false);

  const { squad, error, loading, clearError } = squadState;

  const points = useSquadPoints(
    activeChip,
    horizon === "three" ? 3 : 1,
    squad,
    squad?.is_complete ?? false,
    viewGameweek ?? undefined,
  );

  const pinnedGameweek = viewGameweek ?? undefined;

  // The squad's own identity, which is what makes an edit refetch the suggestion. Captain is part
  // of it because the plan is scored with the captain applied, so moving the armband changes the
  // answer even though the 15 has not.
  const squadKey = squad
    ? `${squad.squad
        .map((p) => p.player_id)
        .sort((a, b) => a - b)
        .join(",")}|${squad.captain_id}`
    : "";

  const transferSuggestion = useTransferSuggestion({
    transfers,
    horizon: horizon === "three" ? 3 : 1,
    chip: activeChip,
    squadKey,
    enabled: squad?.is_complete ?? false,
  });

  if (loading || squad === null) {
    return <p className="loading">Loading…</p>;
  }

  const counts: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
  for (const pick of squad.squad) counts[pick.position] = (counts[pick.position] ?? 0) + 1;
  const fillablePositions = Object.entries(QUOTA)
    .filter(([position, quota]) => (counts[position] ?? 0) < quota)
    .map(([position]) => position);

  async function handleAutoBuild() {
    setSwapSourceId(null);
    // "full_squad" values every one of the 15 picks, not just the XI, so the bench holds real
    // squad depth (autosubs, price rises, future gameweeks) instead of minimum-cost fodder --
    // independent of the starting XI itself, which features.formation.select_starting_xi always
    // derives afterward from just the 11 highest-EV picks, chip or no chip.
    await squadState.optimise("full_squad", activeChip === "triple_captain" ? 3.0 : 2.0);
  }

  function handleSwapSelect(playerId: number) {
    if (!squad) return;
    if (swapSourceId === null) {
      setSwapSourceId(playerId);
      return;
    }
    if (swapSourceId === playerId) {
      setSwapSourceId(null);
      return;
    }
    const sourceIsBench = squad.bench_order.includes(swapSourceId);
    const targetIsBench = squad.bench_order.includes(playerId);
    if (sourceIsBench === targetIsBench) {
      // Same side as the armed player -- treat this click as re-arming the selection instead.
      setSwapSourceId(playerId);
      return;
    }
    const outId = sourceIsBench ? playerId : swapSourceId;
    const inId = sourceIsBench ? swapSourceId : playerId;
    setSwapSourceId(null);
    void squadState.substitute(outId, inId);
  }

  async function handleApplyTransfers(plan: TransferPlanOut) {
    setSwapSourceId(null);
    setApplying(true);
    try {
      await squadState.applyTransfers(plan.out_player_ids, plan.in_player_ids);
    } finally {
      setApplying(false);
    }
  }

  async function handleClearSquad() {
    if (
      !window.confirm(
        "Clear your entire squad and start over with a fresh £100m budget? This can't be undone.",
      )
    ) {
      return;
    }
    setSwapSourceId(null);
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
        viewGameweek={viewGameweek}
        onViewGameweekChange={setViewGameweek}
        onAutoBuild={() => void handleAutoBuild()}
        onClearSquad={() => void handleClearSquad()}
        onImportSquad={(teamId) => {
          setSwapSourceId(null);
          void squadState.importSquad(teamId);
        }}
      />

      <ChipBar activeChip={activeChip} onChange={setActiveChip} />

      <div className="main-content">
        <div className="pitch-column">
          <Pitch
            squad={squad}
            directory={directory}
            teams={teams}
            horizon={horizon}
            pinnedGameweek={pinnedGameweek}
            swapSourceId={swapSourceId}
            onRemove={(playerId) => {
              if (swapSourceId === playerId) setSwapSourceId(null);
              void squadState.removePlayer(playerId);
            }}
            onSetCaptain={(playerId, role) => void squadState.setCaptain(playerId, role)}
            onSwapSelect={handleSwapSelect}
          />

          {squad.is_complete && (
            <TransferBanner
              suggestion={transferSuggestion.suggestion}
              loading={transferSuggestion.loading}
              error={transferSuggestion.error}
              transfers={transfers}
              onTransfersChange={setTransfers}
              onApply={(plan) => void handleApplyTransfers(plan)}
              applying={applying}
            />
          )}
        </div>

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
