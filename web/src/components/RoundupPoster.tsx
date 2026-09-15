import { forwardRef } from "react";
import { RoundupOut } from "../api";
import { bumpChartLineStyle } from "../lib/roundupColours";
import { FONT_FAMILY, SORA_FONT_FACE_CSS } from "../lib/soraFont";

interface RoundupPosterProps {
  data: RoundupOut;
}

// The whole poster is one self-contained <svg>, exported by serialising it directly to a PNG
// (see Roundup.tsx) -- literal hex colours throughout, never var(--token), since the exported
// image is rasterised from a standalone blob with no access to the page's own stylesheet.
const BG = "#f6f5f2";
const SURFACE = "#ffffff";
const BORDER = "#e6e3dc";
const TEXT = "#1c1d1f";
const TEXT_MUTED = "#767a80";
const ACCENT = "#0f7d6f";
const GOLD = "#b5892b";
const DANGER = "#c0392b";
const LOW_TEXT = "#a8382c";
const HIGH_TEXT = "#1c7a3f";

const POSTER_WIDTH = 1080;
const MARGIN = 48;
const CONTENT_WIDTH = POSTER_WIDTH - MARGIN * 2;
const COLUMN_GAP = 24;
const COLUMN_WIDTH = (CONTENT_WIDTH - COLUMN_GAP) / 2;
const CARD_HEIGHT = 300;
const CARD_GAP_Y = 24;
const CHART_ROW_HEIGHT = 40;
const CHART_TOP_PAD = 24;
const CHART_BOTTOM_PAD = 44;
const CHART_HEADER_HEIGHT = 76;
const HEADER_HEIGHT = 176;

function truncate(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, maxChars - 1)}…` : text;
}

/** Card chrome shared by every stat card: surface, border, rounded corners, and a small-caps
 * label. Content is passed as children, positioned relative to the card's own (0, 0). */
function Card({
  x,
  y,
  width,
  height,
  label,
  children,
}: {
  x: number;
  y: number;
  width: number;
  height: number;
  label: string;
  children: React.ReactNode;
}) {
  return (
    <g transform={`translate(${x}, ${y})`}>
      <rect width={width} height={height} rx={12} fill={SURFACE} stroke={BORDER} />
      <text x={20} y={32} fontSize={13} fontWeight={700} letterSpacing={0.6} fill={ACCENT}>
        {label.toUpperCase()}
      </text>
      {children}
    </g>
  );
}

export const RoundupPoster = forwardRef<SVGSVGElement, RoundupPosterProps>(({ data }, ref) => {
  const managerNameByEntry = new Map<number, string>(
    data.template_overlap.map((row) => [row.entry_id, row.manager_name]),
  );
  const entryOrder = [...managerNameByEntry.keys()].sort((a, b) => a - b);
  const nEntries = entryOrder.length;

  const chartHeight =
    CHART_HEADER_HEIGHT +
    CHART_TOP_PAD +
    Math.max(nEntries - 1, 0) * CHART_ROW_HEIGHT +
    CHART_BOTTOM_PAD;

  const latestStandings = data.standings_by_gameweek[data.gameweek] ?? [];
  const leaderName =
    latestStandings.length > 0 ? managerNameByEntry.get(latestStandings[0]) ?? "" : "";
  const biggestFall = data.movers.length > 0 ? data.movers[data.movers.length - 1] : null;
  const headline =
    leaderName && biggestFall && biggestFall.delta < 0
      ? `${leaderName} leads. ${biggestFall.manager_name} dropped ${Math.abs(biggestFall.delta)} place${Math.abs(biggestFall.delta) === 1 ? "" : "s"}.`
      : leaderName
        ? `${leaderName} leads the league after gameweek ${data.gameweek}.`
        : `Gameweek ${data.gameweek} roundup.`;

  const bestCaptain = data.captain_returns
    .filter((row) => row.captain_player_id !== null)
    .reduce<(typeof data.captain_returns)[number] | null>(
      (best, row) => (best === null || row.points > best.points ? row : best),
      null,
    );
  const worstCaptain = data.captain_returns
    .filter((row) => row.captain_player_id !== null)
    .reduce<(typeof data.captain_returns)[number] | null>(
      (worst, row) => (worst === null || row.points < worst.points ? row : worst),
      null,
    );

  const mostTemplate = data.template_overlap.reduce<
    (typeof data.template_overlap)[number] | null
  >((best, row) => (best === null || row.overlap_count > best.overlap_count ? row : best), null);

  const biggestHaul = data.differential_hauls.reduce<
    (typeof data.differential_hauls)[number] | null
  >((best, row) => (best === null || row.total_points > best.total_points ? row : best), null);

  const worstBenchRegret = data.bench_regret.reduce<(typeof data.bench_regret)[number] | null>(
    (worst, row) =>
      worst === null || row.points_left_on_bench > worst.points_left_on_bench ? row : worst,
    null,
  );

  const biggestClimb = data.movers.length > 0 ? data.movers[0] : null;

  const templateByPosition: Record<string, number[]> = { GK: [], DEF: [], MID: [], FWD: [] };
  for (const playerId of data.template_xi) {
    const position = data.players[playerId]?.position;
    if (position && position in templateByPosition) {
      templateByPosition[position].push(playerId);
    }
  }

  const row1Y = HEADER_HEIGHT + chartHeight + CARD_GAP_Y;
  const row2Y = row1Y + CARD_HEIGHT + CARD_GAP_Y;
  const row3Y = row2Y + CARD_HEIGHT + CARD_GAP_Y;
  const row4Y = row3Y + CARD_HEIGHT + CARD_GAP_Y;
  const chipsY = row4Y + CARD_HEIGHT + CARD_GAP_Y;
  const chipsHeight = 60 + Math.max(data.chips.length - 1, 0) * 32;
  const posterHeight = (data.chips.length > 0 ? chipsY + chipsHeight : row4Y + CARD_HEIGHT) + MARGIN;

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${POSTER_WIDTH} ${posterHeight}`}
      width={POSTER_WIDTH}
      height={posterHeight}
      fontFamily={FONT_FAMILY}
    >
      <defs>
        <style>{SORA_FONT_FACE_CSS}</style>
      </defs>
      <rect width={POSTER_WIDTH} height={posterHeight} fill={BG} />

      {/* Header */}
      <g transform={`translate(${MARGIN}, ${MARGIN})`}>
        <text x={0} y={0} fontSize={13} fontWeight={700} letterSpacing={1.2} fill={TEXT_MUTED}>
          {data.league_name.toUpperCase()}
        </text>
        <text x={0} y={48} fontSize={40} fontWeight={700} fill={TEXT}>
          {`GW${data.gameweek} ROUNDUP`}
        </text>
        <text x={0} y={82} fontSize={16} fontWeight={600} fill={TEXT_MUTED}>
          {truncate(headline, 90)}
        </text>
        <rect x={0} y={100} width={CONTENT_WIDTH} height={2} fill={ACCENT} />
      </g>

      {/* Bump chart */}
      <g transform={`translate(${MARGIN}, ${HEADER_HEIGHT})`}>
        <text x={0} y={16} fontSize={12} fontWeight={700} letterSpacing={0.8} fill={TEXT_MUTED}>
          LEAGUE POSITION · EVERY GAMEWEEK
        </text>
        {(() => {
          const plotX0 = 28;
          const plotWidth = CONTENT_WIDTH - plotX0 - 140;
          const plotY0 = CHART_HEADER_HEIGHT + CHART_TOP_PAD;
          const rowStep = CHART_ROW_HEIGHT;
          const stepX = data.gameweek > 1 ? plotWidth / (data.gameweek - 1) : 0;

          const points = entryOrder.map((entryId, index) => {
            const coords: { gw: number; x: number; y: number }[] = [];
            for (let gw = 1; gw <= data.gameweek; gw++) {
              const standings = data.standings_by_gameweek[gw];
              const rank = standings ? standings.indexOf(entryId) : -1;
              if (rank === -1) continue;
              coords.push({
                gw,
                x: plotX0 + (gw - 1) * stepX,
                y: plotY0 + rank * rowStep,
              });
            }
            return { entryId, index, coords };
          });

          return (
            <>
              {Array.from({ length: nEntries }, (_, rank) => (
                <text
                  key={`rank-${rank}`}
                  x={0}
                  y={plotY0 + rank * rowStep + 5}
                  fontSize={13}
                  fontWeight={600}
                  fill={TEXT_MUTED}
                >
                  {rank + 1}
                </text>
              ))}
              {Array.from({ length: data.gameweek }, (_, i) => (
                <text
                  key={`gw-${i}`}
                  x={plotX0 + i * stepX}
                  y={plotY0 + Math.max(nEntries - 1, 0) * rowStep + 28}
                  fontSize={12}
                  fontWeight={600}
                  fill={TEXT_MUTED}
                  textAnchor="middle"
                >
                  {`GW${i + 1}`}
                </text>
              ))}
              {points.map(({ entryId, index, coords }) => {
                if (coords.length === 0) return null;
                const { colour, dashed } = bumpChartLineStyle(index);
                const path = coords.map((c) => `${c.x},${c.y}`).join(" L ");
                const last = coords[coords.length - 1];
                return (
                  <g key={entryId}>
                    <path
                      d={`M ${path}`}
                      fill="none"
                      stroke={colour}
                      strokeWidth={2}
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeDasharray={dashed ? "6 4" : undefined}
                    />
                    {coords.map((c) => (
                      <circle key={c.gw} cx={c.x} cy={c.y} r={4} fill={colour} />
                    ))}
                    <text
                      x={last.x + 12}
                      y={last.y + 4}
                      fontSize={13}
                      fontWeight={600}
                      fill={TEXT}
                    >
                      {truncate(managerNameByEntry.get(entryId) ?? "", 16)}
                    </text>
                  </g>
                );
              })}
            </>
          );
        })()}
      </g>

      {/* Top of the Shop / The Basement */}
      <Card x={MARGIN} y={row1Y} width={COLUMN_WIDTH} height={CARD_HEIGHT} label="Top of the Shop">
        {data.top_managers.slice(0, 4).map((row, i) => (
          <g key={row.entry_id} transform={`translate(20, ${64 + i * 34})`}>
            <text fontSize={14} fontWeight={700} fill={GOLD}>
              {i + 1}
            </text>
            <text x={28} fontSize={15} fontWeight={600} fill={TEXT}>
              {truncate(row.manager_name, 22)}
            </text>
            <text x={COLUMN_WIDTH - 60} fontSize={16} fontWeight={700} fill={HIGH_TEXT}>
              {row.gameweek_points}
            </text>
          </g>
        ))}
      </Card>
      <Card
        x={MARGIN + COLUMN_WIDTH + COLUMN_GAP}
        y={row1Y}
        width={COLUMN_WIDTH}
        height={CARD_HEIGHT}
        label="The Basement"
      >
        {data.bottom_managers.slice(0, 4).map((row, i) => (
          <g key={row.entry_id} transform={`translate(20, ${64 + i * 34})`}>
            <text fontSize={14} fontWeight={700} fill={TEXT_MUTED}>
              {nEntries > 0 ? nEntries - i : ""}
            </text>
            <text x={28} fontSize={15} fontWeight={600} fill={TEXT}>
              {truncate(row.manager_name, 22)}
            </text>
            <text x={COLUMN_WIDTH - 60} fontSize={16} fontWeight={700} fill={LOW_TEXT}>
              {row.gameweek_points}
            </text>
          </g>
        ))}
      </Card>

      {/* He Did That / The Armband */}
      <Card
        x={MARGIN}
        y={row2Y}
        width={COLUMN_WIDTH}
        height={CARD_HEIGHT}
        label="He Did That"
      >
        {data.top_player && (
          <g transform="translate(20, 70)">
            <text fontSize={26} fontWeight={700} fill={TEXT}>
              {truncate(data.players[data.top_player.player_id]?.web_name ?? "", 18)}
            </text>
            <text x={COLUMN_WIDTH - 90} fontSize={30} fontWeight={700} fill={HIGH_TEXT}>
              {data.top_player.live_points}
            </text>
            <text y={26} fontSize={13} fontWeight={600} fill={TEXT_MUTED}>
              {truncate(
                `Owned by ${data.top_player.owner_entry_ids.length} of ${nEntries}`,
                40,
              )}
            </text>
            <text y={52} fontSize={13} fill={TEXT_MUTED}>
              {truncate(
                data.top_player.owner_entry_ids
                  .map((id) => managerNameByEntry.get(id) ?? "")
                  .join(" · "),
                52,
              )}
            </text>
          </g>
        )}
      </Card>
      <Card
        x={MARGIN + COLUMN_WIDTH + COLUMN_GAP}
        y={row2Y}
        width={COLUMN_WIDTH}
        height={CARD_HEIGHT}
        label="The Armband"
      >
        {bestCaptain && (
          <g transform="translate(20, 60)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              BEST CALL
            </text>
            <text y={26} fontSize={17} fontWeight={700} fill={TEXT}>
              {truncate(
                `${bestCaptain.manager_name} · ${data.players[bestCaptain.captain_player_id ?? -1]?.web_name ?? ""}`,
                34,
              )}
            </text>
            <text x={COLUMN_WIDTH - 60} y={26} fontSize={20} fontWeight={700} fill={HIGH_TEXT}>
              {bestCaptain.points}
            </text>
          </g>
        )}
        {worstCaptain && (
          <g transform="translate(20, 130)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              WORST CALL
            </text>
            <text y={26} fontSize={17} fontWeight={700} fill={TEXT}>
              {truncate(
                `${worstCaptain.manager_name} · ${data.players[worstCaptain.captain_player_id ?? -1]?.web_name ?? ""}`,
                34,
              )}
            </text>
            <text x={COLUMN_WIDTH - 60} y={26} fontSize={20} fontWeight={700} fill={DANGER}>
              {worstCaptain.points}
            </text>
          </g>
        )}
      </Card>

      {/* Sheep of the Week / Hipster of the Week */}
      <Card
        x={MARGIN}
        y={row3Y}
        width={COLUMN_WIDTH}
        height={CARD_HEIGHT}
        label="Sheep of the Week · Template XI"
      >
        {(["GK", "DEF", "MID", "FWD"] as const).map((position, rowIndex) => (
          <g key={position} transform={`translate(20, ${58 + rowIndex * 46})`}>
            {templateByPosition[position].map((playerId, i) => (
              <g key={playerId} transform={`translate(${i * 108}, 0)`}>
                <rect width={100} height={30} rx={6} fill={BG} stroke={BORDER} />
                <text x={8} y={20} fontSize={12} fontWeight={600} fill={TEXT}>
                  {truncate(data.players[playerId]?.web_name ?? "", 12)}
                </text>
              </g>
            ))}
          </g>
        ))}
        {mostTemplate && (
          <text x={20} y={CARD_HEIGHT - 20} fontSize={13} fontWeight={600} fill={TEXT_MUTED}>
            {truncate(
              `Most template: ${mostTemplate.manager_name} (${mostTemplate.overlap_count}/${mostTemplate.template_size})`,
              48,
            )}
          </text>
        )}
      </Card>
      <Card
        x={MARGIN + COLUMN_WIDTH + COLUMN_GAP}
        y={row3Y}
        width={COLUMN_WIDTH}
        height={CARD_HEIGHT}
        label="Hipster of the Week"
      >
        {biggestHaul && (
          <g transform="translate(20, 66)">
            <text fontSize={22} fontWeight={700} fill={TEXT}>
              {truncate(biggestHaul.manager_name, 20)}
            </text>
            <text x={COLUMN_WIDTH - 70} fontSize={26} fontWeight={700} fill={HIGH_TEXT}>
              {biggestHaul.total_points}
            </text>
            <text y={30} fontSize={13} fill={TEXT_MUTED}>
              {truncate(
                biggestHaul.differential_player_ids
                  .map((id) => data.players[id]?.web_name ?? "")
                  .join(" · "),
                44,
              )}
            </text>
            <text y={56} fontSize={12} fontWeight={600} fill={TEXT_MUTED}>
              {`${biggestHaul.differential_player_ids.length} player${biggestHaul.differential_player_ids.length === 1 ? "" : "s"} nobody else owned.`}
            </text>
          </g>
        )}
      </Card>

      {/* Bench Regret / Up and Down */}
      <Card x={MARGIN} y={row4Y} width={COLUMN_WIDTH} height={CARD_HEIGHT} label="Bench Regret">
        {worstBenchRegret && (
          <g transform="translate(20, 66)">
            <text fontSize={22} fontWeight={700} fill={TEXT}>
              {truncate(worstBenchRegret.manager_name, 20)}
            </text>
            <text x={COLUMN_WIDTH - 70} fontSize={26} fontWeight={700} fill={DANGER}>
              {worstBenchRegret.points_left_on_bench}
            </text>
            <text y={30} fontSize={13} fill={TEXT_MUTED}>
              {truncate(
                worstBenchRegret.contributing_player_ids
                  .map((id) => data.players[id]?.web_name ?? "")
                  .join(" · "),
                44,
              )}
            </text>
          </g>
        )}
      </Card>
      <Card
        x={MARGIN + COLUMN_WIDTH + COLUMN_GAP}
        y={row4Y}
        width={COLUMN_WIDTH}
        height={CARD_HEIGHT}
        label="Up and Down"
      >
        {biggestClimb && (
          <g transform="translate(20, 62)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              BIGGEST CLIMB
            </text>
            <text y={26} fontSize={16} fontWeight={700} fill={TEXT}>
              {truncate(biggestClimb.manager_name, 24)}
            </text>
            <text x={COLUMN_WIDTH - 90} y={26} fontSize={18} fontWeight={700} fill={HIGH_TEXT}>
              {`${biggestClimb.rank_before} → ${biggestClimb.rank_after}`}
            </text>
          </g>
        )}
        {biggestFall && (
          <g transform="translate(20, 132)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              BIGGEST FALL
            </text>
            <text y={26} fontSize={16} fontWeight={700} fill={TEXT}>
              {truncate(biggestFall.manager_name, 24)}
            </text>
            <text x={COLUMN_WIDTH - 90} y={26} fontSize={18} fontWeight={700} fill={DANGER}>
              {`${biggestFall.rank_before} → ${biggestFall.rank_after}`}
            </text>
          </g>
        )}
      </Card>

      {/* Chips on the Table -- omitted entirely when nobody played one this gameweek */}
      {data.chips.length > 0 && (
        <Card
          x={MARGIN}
          y={chipsY}
          width={CONTENT_WIDTH}
          height={chipsHeight}
          label="Chips on the Table"
        >
          {data.chips.map((chip, i) => (
            <text key={chip.entry_id} x={20} y={56 + i * 32} fontSize={14} fontWeight={600} fill={TEXT}>
              {truncate(
                `${chip.manager_name} played ${chip.chip_name}` +
                  (chip.points_effect !== null ? `, added ${chip.points_effect} points` : "") +
                  (chip.resulting_rank !== null ? `, finished ${chip.resulting_rank}${chip.resulting_rank === 1 ? "st" : chip.resulting_rank === 2 ? "nd" : chip.resulting_rank === 3 ? "rd" : "th"}` : "") +
                  ".",
                100,
              )}
            </text>
          ))}
        </Card>
      )}

      <text
        x={MARGIN}
        y={posterHeight - 16}
        fontSize={11}
        fontWeight={600}
        fill={TEXT_MUTED}
      >
        {`Live FPL data · ${data.league_name} · GW${data.gameweek}`}
      </text>
    </svg>
  );
});

RoundupPoster.displayName = "RoundupPoster";
