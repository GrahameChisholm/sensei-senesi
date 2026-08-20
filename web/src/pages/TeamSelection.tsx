import { useState } from "react";
import { useSquad } from "../hooks/useSquad";
import {
  useGameweek,
  usePlayerDirectory,
  useSquadPoints,
  useTeams,
  useTransferRecommendation,
} from "../hooks/useProjections";
import { GameweekHeader } from "../components/GameweekHeader";
import { ChipBar } from "../components/ChipBar";
import { DraftCompareBar } from "../components/DraftCompareBar";
import { Pitch } from "../components/Pitch";
import { PlayerPanel } from "../components/PlayerPanel";
import { SquadBuilder } from "../components/SquadBuilder";
import { RuleViolationToast } from "../components/RuleViolationToast";
import { TransferRecommendationBanner } from "../components/TransferRecommendationBanner";

export function TeamSelection() {
  const squadState = useSquad();
  const [gameweek, refreshGameweek] = useGameweek();
  const teams = useTeams();
  const [directory, refreshDirectory] = usePlayerDirectory();

  const [horizon, setHorizon] = useState<"next" | "three">("next");
  const [previewChip, setPreviewChip] = useState<string | null>(null);
  // The squad member currently marked for removal (hover "x" on their card), if any -- purely
  // local until a same-position replacement is picked in the Player Panel.
  const [removing, setRemoving] = useState<number | null>(null);
  const [applyingRecommendation, setApplyingRecommendation] = useState(false);

  const { squad, error, loading, clearError } = squadState;
  const recommendation = useTransferRecommendation(squad);

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
    return (
      <div className="team-selection">
        {error && <RuleViolationToast message={error} onDismiss={clearError} />}
        <SquadBuilder
          picks={squad.build_picks ?? []}
          teams={teams}
          onAdd={(playerId, position, price) => void squadState.addBuildPlayer(playerId, position, price)}
          onRemove={(playerId) => void squadState.removeBuildPlayer(playerId)}
          onConfirm={(body) => void squadState.confirmBuild(body)}
        />
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
    if (removing === null) return;
    await squadState.liveTransfer(removing, playerId, price, position);
    setRemoving(null);
  }

  async function handleSetCaptain(playerId: number, role: "captain" | "vice") {
    await squadState.liveCaptain(playerId, role);
  }

  async function handleApplyRecommendation() {
    if (!recommendation) return;
    setApplyingRecommendation(true);
    try {
      const { sell_player_id, buy_player_id, buy_price, position } = recommendation;
      if (squad!.draft) {
        await squadState.transfer(sell_player_id, buy_player_id, buy_price, position);
      } else {
        await squadState.liveTransfer(sell_player_id, buy_player_id, buy_price, position);
      }
    } finally {
      setApplyingRecommendation(false);
    }
  }

  const draftChipIsRebuild = squad.draft?.chip === "wildcard" || squad.draft?.chip === "free_hit";
  const removingPlayer = removing !== null ? teamState.squad.find((p) => p.player_id === removing) : undefined;
  const affordableBudget = removingPlayer ? teamState.bank + removingPlayer.sell_price : null;

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
          setRemoving(null);
        }}
      />

      <ChipBar squad={squad} previewChip={previewChip} onPreview={setPreviewChip} onPlay={(chip) => void handlePlayChip(chip)} />

      {editing && draftChipIsRebuild && <DraftCompareBar draftChip={squad.draft!.chip} refreshKey={squad} />}

      <div className="main-content">
        <div className="squad-column">
          <Pitch
            teamState={teamState}
            directory={directory}
            teams={teams}
            horizon={horizon}
            removing={removing}
            onStartRemove={setRemoving}
            onCancelRemove={() => setRemoving(null)}
            onSetCaptain={(playerId, role) => void handleSetCaptain(playerId, role)}
          />

          <TransferRecommendationBanner
            recommendation={recommendation}
            onApply={() => void handleApplyRecommendation()}
            applying={applyingRecommendation}
          />
        </div>

        <PlayerPanel
          teams={teams}
          transferOutSelected={removing}
          fillPosition={removingPlayer?.position ?? null}
          squadPlayerIds={teamState.squad.map((p) => p.player_id)}
          affordableBudget={affordableBudget}
          onTransferIn={(playerId, position, price) => void handleTransferIn(playerId, position, price)}
          onCancel={() => setRemoving(null)}
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
