import { useEffect, useRef, useState } from "react";
import { RoundupPoster } from "../components/RoundupPoster";
import { MiniLeagueSetup } from "../components/MiniLeagueSetup";
import { GameweekStepper } from "../components/GameweekStepper";
import { useMiniLeagueSettings } from "../hooks/useMiniLeague";
import { useRoundup } from "../hooks/useRoundup";

const EXPORT_SCALE = 2;

/** Serialises the poster's own <svg> to a PNG at EXPORT_SCALE and triggers a download -- the
 * poster is a single self-contained SVG (literal colours, embedded font) precisely so this needs
 * no dependency: XMLSerializer -> blob -> <img> -> canvas -> toBlob. */
async function exportPosterAsPng(svg: SVGSVGElement, filename: string): Promise<void> {
  const width = Number(svg.getAttribute("width"));
  const height = Number(svg.getAttribute("height"));
  const serialised = new XMLSerializer().serializeToString(svg);
  const svgBlob = new Blob([serialised], { type: "image/svg+xml;charset=utf-8" });
  const svgUrl = URL.createObjectURL(svgBlob);

  try {
    const image = await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("Failed to rasterise the roundup poster"));
      img.src = svgUrl;
    });

    const canvas = document.createElement("canvas");
    canvas.width = width * EXPORT_SCALE;
    canvas.height = height * EXPORT_SCALE;
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("Canvas 2D context unavailable");
    ctx.scale(EXPORT_SCALE, EXPORT_SCALE);
    ctx.drawImage(image, 0, 0, width, height);

    const pngBlob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/png"));
    if (!pngBlob) throw new Error("Failed to encode PNG");

    const downloadUrl = URL.createObjectURL(pngBlob);
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = filename;
    link.click();
    URL.revokeObjectURL(downloadUrl);
  } finally {
    URL.revokeObjectURL(svgUrl);
  }
}

function slugify(text: string): string {
  return text
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/(^-|-$)/g, "");
}

export function Roundup() {
  const { settings, loading: settingsLoading, save } = useMiniLeagueSettings();
  const [selectedLeagueId, setSelectedLeagueId] = useState<number | null>(null);

  useEffect(() => {
    if (settings && settings.mini_league_ids.length > 0 && selectedLeagueId === null) {
      setSelectedLeagueId(settings.mini_league_ids[0]);
    }
  }, [settings, selectedLeagueId]);

  const [selectedGameweek, setSelectedGameweek] = useState<number | null>(null);
  const [latestCompleteGameweek, setLatestCompleteGameweek] = useState<number | null>(null);

  const { roundup, loading, error, refresh } = useRoundup(selectedLeagueId, selectedGameweek);

  useEffect(() => {
    if (roundup && latestCompleteGameweek === null) {
      setLatestCompleteGameweek(roundup.gameweek);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [roundup]);

  const svgRef = useRef<SVGSVGElement>(null);
  const [exporting, setExporting] = useState(false);

  async function handleDownload() {
    if (!svgRef.current || !roundup) return;
    setExporting(true);
    try {
      await exportPosterAsPng(
        svgRef.current,
        `${slugify(roundup.league_name)}-gw${roundup.gameweek}.png`,
      );
    } finally {
      setExporting(false);
    }
  }

  if (settingsLoading) {
    return <p className="loading">Loading…</p>;
  }

  if (!settings || settings.mini_league_ids.length === 0) {
    return (
      <div className="team-selection">
        <h2>Roundup</h2>
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
      <div
        className="stats-filters"
        style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
          {settings.mini_league_ids.length > 1 && (
            <div className="stats-team-picker">
              {settings.mini_league_ids.map((leagueId) => (
                <button
                  key={leagueId}
                  className={leagueId === selectedLeagueId ? "active" : ""}
                  onClick={() => {
                    setSelectedLeagueId(leagueId);
                    setSelectedGameweek(null);
                    setLatestCompleteGameweek(null);
                  }}
                >
                  League {leagueId}
                </button>
              ))}
            </div>
          )}
          {latestCompleteGameweek !== null && (
            <label className="stats-range-label">
              Gameweek
              <GameweekStepper
                value={selectedGameweek ?? latestCompleteGameweek}
                min={1}
                max={latestCompleteGameweek}
                onChange={setSelectedGameweek}
              />
            </label>
          )}
        </div>
        <div style={{ display: "flex", gap: "0.5rem" }}>
          <button type="button" onClick={() => refresh()} disabled={loading}>
            Refresh
          </button>
          <button
            type="button"
            className="btn-primary"
            onClick={handleDownload}
            disabled={!roundup || exporting}
          >
            {exporting ? "Exporting…" : "Download PNG · 2×"}
          </button>
        </div>
      </div>

      {loading ? (
        <p className="loading">Loading…</p>
      ) : error ? (
        <div className="differentials-empty">{error}</div>
      ) : roundup === null ? null : (
        <div style={{ marginTop: "1rem", maxWidth: 540 }}>
          <RoundupPoster ref={svgRef} data={roundup} />
        </div>
      )}
    </div>
  );
}
