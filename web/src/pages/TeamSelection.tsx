import { useState } from "react";
import { useSquad } from "../hooks/useSquad";
import { useGameweek, usePlayerDirectory, useSquadPoints, useTeams } from "../hooks/useProjections";
import { GameweekHeader } from "../components/GameweekHeader";
import { ChipBar } from "../components/ChipBar";
import { DraftCompareBar } from "../components/DraftCompareBar";
import { Pitch } from "../components/Pitch";
import { PlayerPanel } from "../components/PlayerPanel";
import { SquadBuilder } from "../components/SquadBuilder";
import { RuleViolationToast } from "../components/RuleViolationToast";
import { SeasonProgress } from "../components/SeasonProgress";
import { GameweekResultBanner } from "../components/GameweekResultBanner";
import { AdvanceResultOut, SeasonLogEntryOut } from "../api";

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
  const [seasonLog, setSeasonLog] = useState<SeasonLogEntryOut[]>([]);
  const [lastResult, setLastResult] = useState<AdvanceResultOut | null>(null);

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

  // Season Replay still goes through the draft/confirm machinery (hit costs and free transfers
  // matter when simulating a real season); the live season doesn't -- see api.liveTransfer.
  const isReplay = gameweek?.is_replay ?? false;
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
    if (isReplay) {
      if (!squad!.draft) {
        await squadState.openDraft();
      }
      await squadState.transfer(removing, playerId, price, position);
    } else {
      await squadState.liveTransfer(removing, playerId, price, position);
    }
    setRemoving(null);
  }

  async function handleSetCaptain(playerId: number, role: "captain" | "vice") {
    if (isReplay) {
      if (!squad!.draft) {
        await squadState.openDraft();
      }
      await squadState.setCaptain(playerId, role);
    } else {
      await squadState.liveCaptain(playerId, role);
    }
  }

  async function handleAdvance() {
    const result = await squadState.advance();
    if (result === null) return;
    setSeasonLog(result.season_log);
    setLastResult(result);
    // /squad/advance already moved the process-wide app state on to the next gameweek's cache --
    // the header and player directory (prices/fixtures/projections) both need an explicit refetch
    // to pick that up, since neither polls on its own.
    await refreshGameweek();
    await refreshDirectory();
  }

  const draftChipIsRebuild = squad.draft?.chip === "wildcard" || squad.draft?.chip === "free_hit";
  const removingPlayer = removing !== null ? teamState.squad.find((p) => p.player_id === removing) : undefined;
  const affordableBudget = removingPlayer ? teamState.bank + removingPlayer.sell_price : null;

  return (
    <div className="team-selection">
      {error && <RuleViolationToast message={error} onDismiss={clearError} />}
      <GameweekResultBanner result={lastResult} onDismiss={() => setLastResult(null)} />

      <SeasonProgress
        gameweek={gameweek}
        seasonLog={seasonLog}
        seasonComplete={lastResult?.season_complete ?? false}
        editing={editing}
        onAdvance={() => void handleAdvance()}
      />

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
