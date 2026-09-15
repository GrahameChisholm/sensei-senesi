import { forwardRef } from "react";
import { CaptainReturnOut, RoundupOut, RoundupPlayerRefOut } from "../api";
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
const NEUTRAL_BADGE = "#9aa0a8";
const PITCH_GREEN = "#2f7a4f";
const PITCH_LINE = "rgba(255, 255, 255, 0.55)";

const POSTER_WIDTH = 1080;
const MARGIN = 48;
const CONTENT_WIDTH = POSTER_WIDTH - MARGIN * 2;
const COLUMN_GAP = 24;
const COLUMN_WIDTH = (CONTENT_WIDTH - COLUMN_GAP) / 2;
const TOP_BOTTOM_CARD_HEIGHT = 170;
// Rows 2 and 4 each pair a single-stat card (Highest Scoring Player, Points Left on the Bench)
// with a two-sub-block card (Captaincy, Rank Movement) -- the two-sub-block layout is always the
// taller of the two (second sub-block starts at local y=130ish, its own content runs to about
// y=160), so this is that height plus bottom breathing room, and both cards in the row share it
// rather than the single-stat card carrying a lot of dead space underneath.
const TWO_STAT_CARD_HEIGHT = 185;
const CARD_GAP_Y = 24;
const CHART_ROW_HEIGHT = 40;
const CHART_TOP_PAD = 24;
const CHART_BOTTOM_PAD = 44;
const CHART_HEADER_HEIGHT = 76;
const HEADER_HEIGHT = 176;

// Row 3 (Template Team / Template & Unique): the pitch needs real room and, per the redesign
// brief, more width than its neighbour -- a 60/40 split rather than the even columns every other
// row uses.
const PITCH_CARD_WIDTH = Math.round(CONTENT_WIDTH * 0.6);
const TEMPLATE_UNIQUE_CARD_WIDTH = CONTENT_WIDTH - COLUMN_GAP - PITCH_CARD_WIDTH;
const ROW3_HEIGHT = 380;

const CHIP_DISPLAY_NAME: Record<string, string> = {
  "3xc": "Triple Captain",
  bboost: "Bench Boost",
  wildcard: "Wildcard",
  freehit: "Free Hit",
};

const CHIP_COLOUR: Record<string, string> = {
  "3xc": ACCENT,
  bboost: GOLD,
};

function chipDisplayName(chipName: string): string {
  return CHIP_DISPLAY_NAME[chipName] ?? chipName.charAt(0).toUpperCase() + chipName.slice(1);
}

function chipColour(chipName: string): string {
  return CHIP_COLOUR[chipName] ?? NEUTRAL_BADGE;
}

function ordinal(n: number): string {
  if (n % 100 >= 11 && n % 100 <= 13) return `${n}th`;
  switch (n % 10) {
    case 1:
      return `${n}st`;
    case 2:
      return `${n}nd`;
    case 3:
      return `${n}rd`;
    default:
      return `${n}th`;
  }
}

function truncate(text: string, maxChars: number): string {
  return text.length > maxChars ? `${text.slice(0, maxChars - 1)}…` : text;
}

/** Breaks `text` onto lines of at most `maxCharsPerLine`, only ever at a space -- never mid-word
 * -- so real content (a manager's name, a player list) is never cut off with an ellipsis. Used
 * wherever the card's own height is computed from the result, so wrapping onto an extra line
 * grows the card rather than clipping the content. */
function wrapText(text: string, maxCharsPerLine: number): string[] {
  if (!text) return [];
  const words = text.split(" ");
  const lines: string[] = [];
  let current = "";
  for (const word of words) {
    const candidate = current ? `${current} ${word}` : word;
    if (candidate.length > maxCharsPerLine && current) {
      lines.push(current);
      current = word;
    } else {
      current = candidate;
    }
  }
  if (current) lines.push(current);
  return lines;
}

/** One end of the captaincy card: the extreme (highest or lowest) points return, and every row
 * tied on that exact value. A tie of one just names that manager; a tie of several is reported
 * as a headcount, naming the shared player only when every tied manager actually captained the
 * same one (almost always true, since a tie is normally the same captain, but not guaranteed). */
function describeCaptainExtreme(
  rows: CaptainReturnOut[],
  pick: (a: CaptainReturnOut, b: CaptainReturnOut) => CaptainReturnOut,
  players: Record<number, RoundupPlayerRefOut>,
): { label: string; points: number } | null {
  const eligible = rows.filter((row) => row.captain_player_id !== null);
  if (eligible.length === 0) return null;
  const extreme = eligible.reduce(pick);
  const tied = eligible.filter((row) => row.points === extreme.points);
  if (tied.length === 1) {
    const playerName = players[tied[0].captain_player_id ?? -1]?.web_name ?? "";
    return { label: `${tied[0].manager_name} · ${playerName}`, points: extreme.points };
  }
  const distinctPlayers = new Set(tied.map((row) => row.captain_player_id));
  const sharedPlayerName =
    distinctPlayers.size === 1 ? players[tied[0].captain_player_id ?? -1]?.web_name ?? "" : null;
  return {
    label: `Captained by ${tied.length} managers${sharedPlayerName ? ` · ${sharedPlayerName}` : ""}`,
    points: extreme.points,
  };
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

  const bestCaptain = describeCaptainExtreme(
    data.captain_returns,
    (a, b) => (b.points > a.points ? b : a),
    data.players,
  );
  const worstCaptain = describeCaptainExtreme(
    data.captain_returns,
    (a, b) => (b.points < a.points ? b : a),
    data.players,
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

  // Captaincy: a tied-manager label ("Captained by 8 managers · Haaland") can run past a single
  // line's width, and it names an actual player -- truncating it with an ellipsis was cutting
  // off exactly the information the card exists to show. Wrapped instead, and the card (and its
  // row) grows to fit however many lines that takes, rather than clipping.
  const CAPTAIN_LABEL_CHARS_PER_LINE = 30;
  const CAPTAIN_LABEL_LINE_HEIGHT = 22;
  const CAPTAIN_BLOCK_TOP = 60;
  const CAPTAIN_BLOCK_GAP = 44;
  const CAPTAIN_BOTTOM_PADDING = 24;
  const bestCaptainLines = bestCaptain ? wrapText(bestCaptain.label, CAPTAIN_LABEL_CHARS_PER_LINE) : [];
  const worstCaptainLines = worstCaptain
    ? wrapText(worstCaptain.label, CAPTAIN_LABEL_CHARS_PER_LINE)
    : [];
  const captainBlockHeight = (lines: string[]) =>
    26 + Math.max(lines.length - 1, 0) * CAPTAIN_LABEL_LINE_HEIGHT;
  const worstCaptainBlockY =
    CAPTAIN_BLOCK_TOP + captainBlockHeight(bestCaptainLines) + CAPTAIN_BLOCK_GAP;
  const captaincyCardHeight =
    worstCaptainBlockY + captainBlockHeight(worstCaptainLines) + CAPTAIN_BOTTOM_PADDING;

  // Most Unique: list every true differential, not just as many as fit on one line -- the same
  // "don't cut off real information" fix, wrapped onto as many lines as the actual list needs.
  const DIFFERENTIAL_LIST_CHARS_PER_LINE = 46;
  const DIFFERENTIAL_LIST_LINE_HEIGHT = 17;
  const differentialNames = biggestHaul
    ? biggestHaul.differential_player_ids.map((id) => data.players[id]?.web_name ?? "").join(" · ")
    : "";
  const differentialLines = wrapText(differentialNames, DIFFERENTIAL_LIST_CHARS_PER_LINE);
  const differentialListY = 50;
  const differentialCountY =
    differentialListY + differentialLines.length * DIFFERENTIAL_LIST_LINE_HEIGHT + 5;
  const templateUniqueCardHeight = biggestHaul ? 160 + differentialCountY + 24 : ROW3_HEIGHT;

  const row1Y = HEADER_HEIGHT + chartHeight + CARD_GAP_Y;
  const row2Height = Math.max(TWO_STAT_CARD_HEIGHT, captaincyCardHeight);
  const row3Height = Math.max(ROW3_HEIGHT, templateUniqueCardHeight);
  const row2Y = row1Y + TOP_BOTTOM_CARD_HEIGHT + CARD_GAP_Y;
  const row3Y = row2Y + row2Height + CARD_GAP_Y;
  const row4Y = row3Y + row3Height + CARD_GAP_Y;
  const chipsY = row4Y + TWO_STAT_CARD_HEIGHT + CARD_GAP_Y;
  // First row sits at local y=52; each badge is 24 tall; CHIPS_BOTTOM_PADDING below the last
  // badge's bottom edge is the card's own close, matching every other card's bottom margin.
  const CHIPS_FIRST_ROW_Y = 52;
  const CHIPS_ROW_HEIGHT = 36;
  const CHIPS_BADGE_HEIGHT = 24;
  const CHIPS_BOTTOM_PADDING = 20;
  const chipsHeight =
    CHIPS_FIRST_ROW_Y +
    Math.max(data.chips.length - 1, 0) * CHIPS_ROW_HEIGHT +
    CHIPS_BADGE_HEIGHT +
    CHIPS_BOTTOM_PADDING;
  const posterHeight =
    (data.chips.length > 0 ? chipsY + chipsHeight : row4Y + TWO_STAT_CARD_HEIGHT) + MARGIN;

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

      {/* Highest Point Scorer / Lowest Point Scorer */}
      <Card
        x={MARGIN}
        y={row1Y}
        width={COLUMN_WIDTH}
        height={TOP_BOTTOM_CARD_HEIGHT}
        label="Highest Point Scorer"
      >
        {data.top_managers.slice(0, 3).map((row, i) => (
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
        height={TOP_BOTTOM_CARD_HEIGHT}
        label="Lowest Point Scorer"
      >
        {data.bottom_managers.slice(0, 3).map((row, i) => (
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

      {/* Highest Scoring Player / Captaincy */}
      <Card
        x={MARGIN}
        y={row2Y}
        width={COLUMN_WIDTH}
        height={row2Height}
        label="Highest Scoring Player"
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
        height={row2Height}
        label="Captaincy"
      >
        {bestCaptain && (
          <g transform={`translate(20, ${CAPTAIN_BLOCK_TOP})`}>
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              BEST CALL
            </text>
            {bestCaptainLines.map((line, i) => (
              <text
                key={i}
                y={26 + i * CAPTAIN_LABEL_LINE_HEIGHT}
                fontSize={17}
                fontWeight={700}
                fill={TEXT}
              >
                {line}
              </text>
            ))}
            <text x={COLUMN_WIDTH - 60} y={26} fontSize={20} fontWeight={700} fill={HIGH_TEXT}>
              {bestCaptain.points}
            </text>
          </g>
        )}
        {worstCaptain && (
          <g transform={`translate(20, ${worstCaptainBlockY})`}>
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              WORST CALL
            </text>
            {worstCaptainLines.map((line, i) => (
              <text
                key={i}
                y={26 + i * CAPTAIN_LABEL_LINE_HEIGHT}
                fontSize={17}
                fontWeight={700}
                fill={TEXT}
              >
                {line}
              </text>
            ))}
            <text x={COLUMN_WIDTH - 60} y={26} fontSize={20} fontWeight={700} fill={DANGER}>
              {worstCaptain.points}
            </text>
          </g>
        )}
      </Card>

      {/* Template Team (pitch) / Most Template + Most Unique */}
      <Card x={MARGIN} y={row3Y} width={PITCH_CARD_WIDTH} height={row3Height} label="Template Team">
        {(() => {
          const pitchX = 20;
          const pitchY = 48;
          const pitchWidth = PITCH_CARD_WIDTH - 40;
          const pitchHeight = ROW3_HEIGHT - 68;
          const rowOrder: ("GK" | "DEF" | "MID" | "FWD")[] = ["GK", "DEF", "MID", "FWD"];
          const rowHeight = pitchHeight / rowOrder.length;
          const tokenWidth = 88;
          const tokenHeight = 42;

          function rowTokenXs(count: number): number[] {
            if (count === 0) return [];
            const gap = (pitchWidth - count * tokenWidth) / (count + 1);
            return Array.from({ length: count }, (_, i) => gap * (i + 1) + tokenWidth * i);
          }

          return (
            <g transform={`translate(${pitchX}, ${pitchY})`}>
              <rect width={pitchWidth} height={pitchHeight} rx={10} fill={PITCH_GREEN} />
              <line
                x1={0}
                y1={pitchHeight / 2}
                x2={pitchWidth}
                y2={pitchHeight / 2}
                stroke={PITCH_LINE}
                strokeWidth={1.5}
              />
              <circle
                cx={pitchWidth / 2}
                cy={pitchHeight / 2}
                r={36}
                fill="none"
                stroke={PITCH_LINE}
                strokeWidth={1.5}
              />
              <rect
                x={pitchWidth / 2 - 90}
                y={0}
                width={180}
                height={28}
                fill="none"
                stroke={PITCH_LINE}
                strokeWidth={1.5}
              />
              <rect
                x={pitchWidth / 2 - 90}
                y={pitchHeight - 28}
                width={180}
                height={28}
                fill="none"
                stroke={PITCH_LINE}
                strokeWidth={1.5}
              />
              <rect
                width={pitchWidth}
                height={pitchHeight}
                rx={10}
                fill="none"
                stroke={PITCH_LINE}
                strokeWidth={1.5}
              />

              {rowOrder.map((position, rowIndex) => {
                const ids = templateByPosition[position];
                const xs = rowTokenXs(ids.length);
                const rowCenterY = rowHeight * rowIndex + rowHeight / 2;
                return (
                  <g key={position}>
                    {ids.map((playerId, i) => {
                      const starterCount = data.template_starter_counts[playerId] ?? 0;
                      return (
                        <g
                          key={playerId}
                          transform={`translate(${xs[i]}, ${rowCenterY - tokenHeight / 2})`}
                        >
                          <rect
                            width={tokenWidth}
                            height={tokenHeight}
                            rx={8}
                            fill={SURFACE}
                            stroke="rgba(0, 0, 0, 0.08)"
                          />
                          <text
                            x={tokenWidth / 2}
                            y={17}
                            fontSize={11}
                            fontWeight={700}
                            fill={TEXT}
                            textAnchor="middle"
                          >
                            {truncate(data.players[playerId]?.web_name ?? "", 11)}
                          </text>
                          <text
                            x={tokenWidth / 2}
                            y={32}
                            fontSize={10}
                            fontWeight={600}
                            fill={TEXT_MUTED}
                            textAnchor="middle"
                          >
                            {`${starterCount}/${nEntries} owned`}
                          </text>
                        </g>
                      );
                    })}
                  </g>
                );
              })}
            </g>
          );
        })()}
      </Card>
      <Card
        x={MARGIN + PITCH_CARD_WIDTH + COLUMN_GAP}
        y={row3Y}
        width={TEMPLATE_UNIQUE_CARD_WIDTH}
        height={row3Height}
        label="Template & Unique"
      >
        {mostTemplate && (
          <g transform="translate(20, 60)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              MOST TEMPLATE
            </text>
            <text y={26} fontSize={18} fontWeight={700} fill={TEXT}>
              {truncate(mostTemplate.manager_name, 20)}
            </text>
            <text y={50} fontSize={14} fontWeight={600} fill={TEXT_MUTED}>
              {`${mostTemplate.overlap_count} of ${mostTemplate.template_size} template players`}
            </text>
          </g>
        )}
        {biggestHaul && (
          <g transform="translate(20, 160)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              MOST UNIQUE
            </text>
            <text y={26} fontSize={18} fontWeight={700} fill={TEXT}>
              {truncate(biggestHaul.manager_name, 20)}
            </text>
            <text x={TEMPLATE_UNIQUE_CARD_WIDTH - 60} y={26} fontSize={18} fontWeight={700} fill={HIGH_TEXT}>
              {biggestHaul.total_points}
            </text>
            {differentialLines.map((line, i) => (
              <text
                key={i}
                y={differentialListY + i * DIFFERENTIAL_LIST_LINE_HEIGHT}
                fontSize={13}
                fill={TEXT_MUTED}
              >
                {line}
              </text>
            ))}
            <text y={differentialCountY} fontSize={12} fontWeight={600} fill={TEXT_MUTED}>
              {`${biggestHaul.differential_player_ids.length} player${biggestHaul.differential_player_ids.length === 1 ? "" : "s"} nobody else owned.`}
            </text>
          </g>
        )}
      </Card>

      {/* Points Left on the Bench / Rank Movement */}
      <Card
        x={MARGIN}
        y={row4Y}
        width={COLUMN_WIDTH}
        height={TWO_STAT_CARD_HEIGHT}
        label="Points Left on the Bench"
      >
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
        height={TWO_STAT_CARD_HEIGHT}
        label="Rank Movement"
      >
        {biggestClimb && (
          <g transform="translate(20, 62)">
            <text fontSize={11} fontWeight={700} letterSpacing={0.5} fill={TEXT_MUTED}>
              HIGHEST CLIMBER
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

      {/* Chips Used -- omitted entirely when nobody played one this gameweek */}
      {data.chips.length > 0 && (
        <Card x={MARGIN} y={chipsY} width={CONTENT_WIDTH} height={chipsHeight} label="Chips Used">
          {data.chips.map((chip, i) => (
            <g key={chip.entry_id} transform={`translate(20, ${52 + i * 36})`}>
              <rect width={128} height={24} rx={12} fill={chipColour(chip.chip_name)} />
              <text
                x={64}
                y={16}
                fontSize={12}
                fontWeight={700}
                fill={SURFACE}
                textAnchor="middle"
              >
                {chipDisplayName(chip.chip_name)}
              </text>
              <text x={150} y={17} fontSize={14} fontWeight={600} fill={TEXT}>
                {truncate(chip.manager_name, 26)}
              </text>
              <text
                x={CONTENT_WIDTH - 220}
                y={17}
                fontSize={14}
                fontWeight={700}
                fill={HIGH_TEXT}
                textAnchor="end"
              >
                {chip.points_effect !== null ? `+${chip.points_effect} pts` : ""}
              </text>
              <text
                x={CONTENT_WIDTH - 40}
                y={17}
                fontSize={13}
                fontWeight={600}
                fill={TEXT_MUTED}
                textAnchor="end"
              >
                {chip.resulting_rank !== null ? `Finished ${ordinal(chip.resulting_rank)}` : ""}
              </text>
            </g>
          ))}
        </Card>
      )}

      <text x={MARGIN} y={posterHeight - 16} fontSize={11} fontWeight={600} fill={TEXT_MUTED}>
        {`Live FPL data · ${data.league_name} · GW${data.gameweek}`}
      </text>
    </svg>
  );
});

RoundupPoster.displayName = "RoundupPoster";
