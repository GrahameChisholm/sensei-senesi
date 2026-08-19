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
  const [selected, setSelected] = useState<number | null>(null);
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
    if (selected === null) return;
    await squadState.transfer(selected, playerId, price, position);
    setSelected(null);
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
  const selectedSquadPlayer = selected !== null ? teamState.squad.find((p) => p.player_id === selected) : undefined;
  const affordableBudget = selectedSquadPlayer ? teamState.bank + selectedSquadPlayer.sell_price : null;

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
        onEditTeam={() => void squadState.openDraft()}
        onOptimise={() => void squadState.optimiseXi()}
        onResetTeam={() => {
          void squadState.discardDraft();
          setSelected(null);
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
          editable={editing}
          selected={selected}
          onSelect={setSelected}
          onSubstitute={(outId, inId) => void squadState.substitute(outId, inId)}
        />

        <PlayerPanel
          teams={teams}
          editable={editing}
          transferOutSelected={selected}
          affordableBudget={affordableBudget}
          onTransferIn={(playerId, position, price) => void handleTransferIn(playerId, position, price)}
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
