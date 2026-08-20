import { useState } from "react";
import { useSquad } from "../hooks/useSquad";
import { useGameweek, usePlayerDirectory, useSquadPoints, useTeams } from "../hooks/useProjections";
import { GameweekHeader } from "../components/GameweekHeader";
import { ChipBar } from "../components/ChipBar";
import { DraftCompareBar } from "../components/DraftCompareBar";
import { BuildPitch } from "../components/BuildPitch";
import { Pitch } from "../components/Pitch";
import { PlayerPanel } from "../components/PlayerPanel";
import { RuleViolationToast } from "../components/RuleViolationToast";
import { BUDGET, QUOTA, buildDefaultLineup } from "../lib/squadBuild";

export function TeamSelection() {
  const squadState = useSquad();
  const [gameweek, refreshGameweek] = useGameweek();
  const teams = useTeams();
  const [directory, refreshDirectory] = usePlayerDirectory();

  const [horizon, setHorizon] = useState<"next" | "three">("next");
  const [previewChip, setPreviewChip] = useState<string | null>(null);
  // Every squad member currently marked for removal (hover "x" on their card), if any, purely
  // local until a same-position replacement is picked in the Player Panel. Any number of players
  // can be marked at once, each independently filled or cancelled.
  const [removingIds, setRemovingIds] = useState<number[]>([]);

  const { squad, error, loading, clearError } = squadState;

  const activeChipForPoints = previewChip ?? squad?.draft?.chip ?? squad?.active_chip ?? null;
  const points = useSquadPoints(
    activeChipForPoints,
    horizon === "three" ? 3 : 1,
    squad,
    "draft",
    squad?.is_complete ?? false,
  );

  if (loading || squad === null) {
    return <p className="loading">Loading…</p>;
  }

  if (!squad.is_complete) {
    const picks = squad.build_picks ?? [];
    const counts: Record<string, number> = { GK: 0, DEF: 0, MID: 0, FWD: 0 };
    for (const pick of picks) counts[pick.position] = (counts[pick.position] ?? 0) + 1;
    const fillablePositions = Object.entries(QUOTA)
      .filter(([position, quota]) => (counts[position] ?? 0) < quota)
      .map(([position]) => position);
    const spent = picks.reduce((sum, pick) => sum + pick.purchase_price, 0);
    const budgetRemaining = BUDGET - spent;
    const isReadyToSave = fillablePositions.length === 0 && budgetRemaining >= 0;

    async function handleSaveTeam() {
      const lineup = buildDefaultLineup(picks);
      if (!lineup) return;
      await squadState.confirmBuild({ player_ids: picks.map((p) => p.player_id), ...lineup });
    }

    return (
      <div className="team-selection">
        {error && <RuleViolationToast message={error} onDismiss={clearError} />}

        <div className="build-summary">
          <span>Budget remaining: £{(budgetRemaining / 10).toFixed(1)}m</span>
          <span>{picks.length}/15 picked</span>
          <button disabled={!isReadyToSave} onClick={() => void handleSaveTeam()}>
            Save team
          </button>
        </div>

        <div className="main-content">
          <BuildPitch
            picks={picks}
            directory={directory}
            teams={teams}
            horizon={horizon}
            onRemove={(playerId) => void squadState.removeBuildPlayer(playerId)}
          />

          <PlayerPanel
            teams={teams}
            fillMode={fillablePositions.length > 0}
            fillablePositions={fillablePositions}
            squadPlayerIds={picks.map((p) => p.player_id)}
            affordableBudget={budgetRemaining}
            onAdd={(playerId, position, price) =>
              void squadState.addBuildPlayer(playerId, position, price)
            }
          />
        </div>
      </div>
    );
  }

  const editing = squad.draft !== null;
  const teamState = squad.draft?.working_state ?? squad.committed!;

  async function handlePlayChip(chip: string) {
    if (!squad!.draft) {
      await squadState.openDraft();
    }
    await squadState.setDraftChip(chip);
    await squadState.confirmDraft();
    setPreviewChip(null);
  }

  async function handleTransferIn(playerId: number, position: string, price: number) {
    const targetId = removingIds.find(
      (id) => teamState.squad.find((p) => p.player_id === id)?.position === position,
    );
    if (targetId === undefined) return;
    const result = await squadState.liveTransfer(targetId, playerId, price, position);
    if (result) setRemovingIds((ids) => ids.filter((id) => id !== targetId));
  }

  async function handleSetCaptain(playerId: number, role: "captain" | "vice") {
    await squadState.liveCaptain(playerId, role);
  }

  const draftChipIsRebuild = squad.draft?.chip === "wildcard" || squad.draft?.chip === "free_hit";
  const removingPlayers = teamState.squad.filter((p) => removingIds.includes(p.player_id));
  const fillablePositions = [...new Set(removingPlayers.map((p) => p.position))];
  const affordableBudget = removingPlayers.length
    ? teamState.bank + removingPlayers.reduce((sum, p) => sum + p.sell_price, 0)
    : null;

  return (
    <div className="team-selection">
      {error && <RuleViolationToast message={error} onDismiss={clearError} />}

      <GameweekHeader
        gameweek={gameweek}
        squad={squad}
        points={points}
        horizon={horizon}
        onHorizonChange={setHorizon}
        editing={editing}
        onOptimise={() => void squadState.optimiseXi()}
        onResetTeam={() => {
          void squadState.discardDraft();
          setRemovingIds([]);
        }}
      />

      <ChipBar squad={squad} previewChip={previewChip} onPreview={setPreviewChip} onPlay={(chip) => void handlePlayChip(chip)} />

      {editing && draftChipIsRebuild && <DraftCompareBar draftChip={squad.draft!.chip} refreshKey={squad} />}

      <div className="main-content">
        <Pitch
          teamState={teamState}
          directory={directory}
          teams={teams}
          horizon={horizon}
          removingIds={removingIds}
          onStartRemove={(playerId) =>
            setRemovingIds((ids) => (ids.includes(playerId) ? ids : [...ids, playerId]))
          }
          onCancelRemove={(playerId) => setRemovingIds((ids) => ids.filter((id) => id !== playerId))}
          onSetCaptain={(playerId, role) => void handleSetCaptain(playerId, role)}
        />

        <PlayerPanel
          teams={teams}
          fillMode={removingIds.length > 0}
          fillablePositions={fillablePositions}
          squadPlayerIds={teamState.squad.map((p) => p.player_id)}
          affordableBudget={affordableBudget}
          onAdd={(playerId, position, price) => void handleTransferIn(playerId, position, price)}
          onCancel={() => setRemovingIds([])}
        />
      </div>

      {editing && (
        <div className="draft-confirm-bar">
          <button onClick={() => void squadState.confirmDraft()}>Confirm</button>
        </div>
      )}
    </div>
  );
}
