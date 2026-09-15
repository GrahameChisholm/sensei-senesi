import { useState } from "react";

// This tool only ever manages one real FPL squad, so there's nothing to type in.
const MY_TEAM_ID = 308462;

interface ImportTeamFormProps {
  /** Return value is ignored (any caller's promise, e.g. useSquad's Promise<SquadOut | null>, is
   * fine) -- only used to know when the request has settled, so the button can show "Importing…"
   * while it's in flight. */
  onImport: (teamId: number) => unknown;
  /** Shown via window.confirm before importing, when this would overwrite an existing squad. */
  confirmMessage?: string;
}

export function ImportTeamForm({ onImport, confirmMessage }: ImportTeamFormProps) {
  const [importing, setImporting] = useState(false);

  async function handleClick() {
    if (confirmMessage && !window.confirm(confirmMessage)) return;

    setImporting(true);
    try {
      await onImport(MY_TEAM_ID);
    } finally {
      setImporting(false);
    }
  }

  return (
    <button type="button" onClick={() => void handleClick()} disabled={importing}>
      {importing ? "Importing…" : "Import team"}
    </button>
  );
}
