import { useState } from "react";
import { expectedPointsColour, expectedPointsTextColour } from "../lib/colours";
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
  /** When set, show this specific gameweek's points instead of following ``horizon``. */
  pinnedGameweek?: number;
  isCaptain?: boolean;
  isVice?: boolean;
  lowConfidence?: boolean;
  /** Shows a small "x" in the top corner on hover, for marking this player for removal from the
   * squad; omit to hide it (there's currently no case where a card is shown without it). */
  onRemove?: () => void;
  /** Shows small "C"/"VC" pills in the top-left corner on hover; omit both to hide them entirely
   * (only starting-XI players can be captain/vice, so bench cards never get these). */
  onCaptain?: () => void;
  onVice?: () => void;
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
  pinnedGameweek,
  isCaptain,
  isVice,
  lowConfidence,
  onRemove,
  onCaptain,
  onVice,
}: PlayerCardProps) {
  const [hovered, setHovered] = useState(false);
  const primary =
    pinnedGameweek !== undefined
      ? fixtures.find((fixture) => fixture.gameweek === pinnedGameweek) ?? fixtures[0]
      : fixtures[0];

  return (
    <div
      className="player-card"
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
    >
      {hovered && onRemove && (
        <button
          type="button"
          className="card-remove-button"
          title="Remove from squad"
          onClick={(e) => {
            e.stopPropagation();
            onRemove();
          }}
        >
          ×
        </button>
      )}

      {hovered && (onCaptain || onVice) && (
        <div className="card-captain-actions">
          {onCaptain && (
            <button
              type="button"
              className="card-captain-button"
              disabled={isCaptain}
              title="Make captain (doubles their expected points)"
              onClick={(e) => {
                e.stopPropagation();
                onCaptain();
              }}
            >
              C
            </button>
          )}
          {onVice && (
            <button
              type="button"
              className="card-vice-button"
              disabled={isVice}
              title="Make vice captain"
              onClick={(e) => {
                e.stopPropagation();
                onVice();
              }}
            >
              VC
            </button>
          )}
        </div>
      )}

      <div className="player-card-name">
        <span className="player-card-name-text">{name}</span>
        {isCaptain && <span className="badge captain-badge">C</span>}
        {isVice && <span className="badge vice-badge">V</span>}
        {lowConfidence && <span className="badge low-confidence-badge" title="Positional baseline — no history">!</span>}
      </div>
      <div className="player-card-price">{price !== null ? `£${(price / 10).toFixed(1)}m` : "—"}</div>

      {horizon === "next" || pinnedGameweek !== undefined ? (
        <div
          className="player-card-points"
          style={{
            background: expectedPointsColour(primary?.expectedPoints ?? null),
            color: expectedPointsTextColour(primary?.expectedPoints ?? null),
          }}
        >
          <div className="ep-value">{primary?.expectedPoints !== null && primary?.expectedPoints !== undefined ? primary.expectedPoints.toFixed(1) : "—"}</div>
          <div className="fixture-label">{fixtureLabel(primary)}</div>
        </div>
      ) : (
        <div className="player-card-points-triple">
          {fixtures.slice(0, 3).map((fixture) => (
            <div
              key={fixture.gameweek}
              className="ep-mini"
              style={{
                background: expectedPointsColour(fixture.expectedPoints),
                color: expectedPointsTextColour(fixture.expectedPoints),
              }}
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
