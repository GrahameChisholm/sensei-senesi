import { useState } from "react";

interface ImportTeamFormProps {
  /** Return value is ignored (any caller's promise, e.g. useSquad's Promise<SquadOut | null>, is
   * fine) -- only used to know when the request has settled, so the form can show "Importing…"
   * and collapse once it's done. */
  onImport: (teamId: number) => unknown;
  /** Shown via window.confirm before importing, when this would overwrite an existing squad. */
  confirmMessage?: string;
}

export function ImportTeamForm({ onImport, confirmMessage }: ImportTeamFormProps) {
  const [expanded, setExpanded] = useState(false);
  const [teamId, setTeamId] = useState("");
  const [importing, setImporting] = useState(false);

  if (!expanded) {
    return (
      <button type="button" onClick={() => setExpanded(true)}>
        Import team
      </button>
    );
  }

  async function handleSubmit(event: React.FormEvent) {
    event.preventDefault();
    const parsed = Number(teamId);
    if (!Number.isInteger(parsed) || parsed <= 0) return;
    if (confirmMessage && !window.confirm(confirmMessage)) return;

    setImporting(true);
    try {
      await onImport(parsed);
      setExpanded(false);
      setTeamId("");
    } finally {
      setImporting(false);
    }
  }

  return (
    <form className="import-team-form" onSubmit={(event) => void handleSubmit(event)}>
      <input
        type="number"
        min={1}
        placeholder="FPL Team ID"
        value={teamId}
        onChange={(event) => setTeamId(event.target.value)}
        disabled={importing}
        autoFocus
      />
      <button type="submit" className="btn-primary" disabled={importing || teamId === ""}>
        {importing ? "Importing…" : "Import"}
      </button>
      <button
        type="button"
        onClick={() => {
          setExpanded(false);
          setTeamId("");
        }}
        disabled={importing}
      >
        Cancel
      </button>
    </form>
  );
}
