import { useState } from "react";
import { expectedPointsColour } from "../lib/colours";
import { BreakdownPopover } from "./BreakdownPopover";

export interface CardFixture {
  gameweek: number;
  opponentShortName: string | null;
  isHome: boolean | null;
  expectedPoints: number | null;
}

interface PlayerCardProps {
  playerId: number;
  name: string;
  price: number | null;
  fixtures: CardFixture[];
  horizon: "next" | "three";
  isCaptain?: boolean;
  isVice?: boolean;
  lowConfidence?: boolean;
  clickable?: boolean;
  selected?: boolean;
  disabledReason?: string | null;
  onClick?: () => void;
}

function fixtureLabel(fixture: CardFixture | undefined): string {
  if (!fixture || fixture.opponentShortName === null || fixture.isHome === null) return "—";
  return `${fixture.opponentShortName} (${fixture.isHome ? "H" : "A"})`;
}

export function PlayerCard({
  playerId,
  name,
  price,
  fixtures,
  horizon,
  isCaptain,
  isVice,
  lowConfidence,
  clickable,
  selected,
  disabledReason,
  onClick,
}: PlayerCardProps) {
  const [hovered, setHovered] = useState(false);
  const primary = fixtures[0];

  return (
    <div
      className={[
        "player-card",
        clickable && !disabledReason ? "clickable" : "",
        selected ? "selected" : "",
        disabledReason ? "disabled" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      onClick={disabledReason ? undefined : onClick}
      title={disabledReason ?? undefined}
    >
      <div className="player-card-name">
        {name}
        {isCaptain && <span className="badge captain-badge">C</span>}
        {isVice && <span className="badge vice-badge">V</span>}
        {lowConfidence && <span className="badge low-confidence-badge" title="Positional baseline — no history">!</span>}
      </div>
      <div className="player-card-price">{price !== null ? `£${(price / 10).toFixed(1)}m` : "—"}</div>

      {horizon === "next" ? (
        <div className="player-card-points" style={{ background: expectedPointsColour(primary?.expectedPoints ?? null) }}>
          <div className="ep-value">{primary?.expectedPoints !== null && primary?.expectedPoints !== undefined ? primary.expectedPoints.toFixed(1) : "—"}</div>
          <div className="fixture-label">{fixtureLabel(primary)}</div>
        </div>
      ) : (
        <div className="player-card-points-triple">
          {fixtures.slice(0, 3).map((fixture) => (
            <div
              key={fixture.gameweek}
              className="ep-mini"
              style={{ background: expectedPointsColour(fixture.expectedPoints) }}
            >
              <div className="ep-mini-value">{fixture.expectedPoints !== null ? fixture.expectedPoints.toFixed(1) : "—"}</div>
              <div className="ep-mini-fixture">{fixtureLabel(fixture)}</div>
            </div>
          ))}
        </div>
      )}

      {hovered && (
        <div className="popover-anchor">
          <BreakdownPopover playerId={playerId} />
        </div>
      )}
    </div>
  );
}
