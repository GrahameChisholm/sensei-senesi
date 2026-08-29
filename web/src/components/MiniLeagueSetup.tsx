import { useState } from "react";

interface MiniLeagueSetupProps {
  fplTeamId: number | null;
  miniLeagueIds: number[];
  onSave: (fplTeamId: number | null, miniLeagueIds: number[]) => unknown;
}

/** The Mini League page's empty state: nothing here works without knowing which FPL entry is
 * "you" and which league(s) to fetch (MINI_LEAGUE_PLAN M14/M23). A manager who has already
 * imported their squad via the Team page will usually already have an fpl_team_id saved -- this
 * form lets them add it directly too, for the case where they only ever built a squad by hand. */
export function MiniLeagueSetup({ fplTeamId, miniLeagueIds, onSave }: MiniLeagueSetupProps) {
  const [teamIdInput, setTeamIdInput] = useState(fplTeamId !== null ? String(fplTeamId) : "");
  const [leagueIdInput, setLeagueIdInput] = useState("");
  const [leagueIds, setLeagueIds] = useState<number[]>(miniLeagueIds);
  const [saving, setSaving] = useState(false);

  function addLeague() {
    const parsed = Number(leagueIdInput);
    if (!Number.isInteger(parsed) || parsed <= 0 || leagueIds.includes(parsed)) return;
    setLeagueIds((prev) => [...prev, parsed]);
    setLeagueIdInput("");
  }

  function removeLeague(leagueId: number) {
    setLeagueIds((prev) => prev.filter((id) => id !== leagueId));
  }

  async function handleSave(event: React.FormEvent) {
    event.preventDefault();
    const parsedTeamId = teamIdInput === "" ? null : Number(teamIdInput);
    if (parsedTeamId !== null && (!Number.isInteger(parsedTeamId) || parsedTeamId <= 0)) return;
    setSaving(true);
    try {
      await onSave(parsedTeamId, leagueIds);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="differentials-empty">
      <h3>Set up your mini league</h3>
      <p>
        Your FPL team ID (find it in the URL of your Points page on the official site) and at
        least one classic mini-league ID are needed before this page can compute anything.
      </p>
      <form onSubmit={(event) => void handleSave(event)}>
        <div className="import-team-form">
          <input
            type="number"
            min={1}
            placeholder="Your FPL Team ID"
            value={teamIdInput}
            onChange={(event) => setTeamIdInput(event.target.value)}
          />
        </div>
        <div className="import-team-form" style={{ marginTop: "0.6rem" }}>
          <input
            type="number"
            min={1}
            placeholder="Mini-league ID"
            value={leagueIdInput}
            onChange={(event) => setLeagueIdInput(event.target.value)}
          />
          <button type="button" onClick={addLeague}>
            Add league
          </button>
        </div>
        {leagueIds.length > 0 && (
          <div className="stats-team-picker" style={{ marginTop: "0.5rem" }}>
            {leagueIds.map((leagueId) => (
              <button type="button" key={leagueId} onClick={() => removeLeague(leagueId)}>
                {leagueId} ✕
              </button>
            ))}
          </div>
        )}
        <button
          type="submit"
          className="btn-primary"
          style={{ marginTop: "0.85rem" }}
          disabled={saving || leagueIds.length === 0}
        >
          {saving ? "Saving…" : "Save"}
        </button>
      </form>
    </div>
  );
}
