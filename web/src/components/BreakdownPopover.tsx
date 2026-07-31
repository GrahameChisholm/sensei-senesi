import { useEffect, useState } from "react";
import { api, PlayerDetailOut } from "../api";

const COMPONENT_LABELS: Record<string, string> = {
  appearance: "Appearance",
  goals: "Goals",
  assists: "Assists",
  clean_sheet: "Clean sheet",
  goals_conceded: "Goals conceded",
  defensive_contribution: "Defensive contribution",
  saves: "Saves",
  bonus: "Bonus",
  cards: "Cards",
  penalty_misses: "Penalty misses",
  own_goals: "Own goals",
};

export function BreakdownPopover({ playerId }: { playerId: number }) {
  const [detail, setDetail] = useState<PlayerDetailOut | null>(null);

  useEffect(() => {
    let cancelled = false;
    void api.getPlayer(playerId).then((result) => {
      if (!cancelled) setDetail(result);
    });
    return () => {
      cancelled = true;
    };
  }, [playerId]);

  if (!detail) {
    return (
      <div className="breakdown-popover">
        <p>Loading…</p>
      </div>
    );
  }

  const lines = Object.entries(detail.breakdown).filter(
    ([key, value]) => key !== "total" && Math.abs(value) > 0.001,
  );

  return (
    <div className="breakdown-popover">
      {detail.low_confidence && (
        <p className="low-confidence-note">Positional baseline — no history for this player</p>
      )}
      <table>
        <tbody>
          {lines.map(([key, value]) => (
            <tr key={key}>
              <td>{COMPONENT_LABELS[key] ?? key}</td>
              <td className={value < 0 ? "negative" : undefined}>{value.toFixed(1)}</td>
            </tr>
          ))}
          <tr className="total-row">
            <td>Total</td>
            <td>{detail.breakdown.total.toFixed(1)}</td>
          </tr>
        </tbody>
      </table>
      {detail.floor !== null && detail.ceiling !== null && (
        <p className="floor-ceiling">
          Floor {detail.floor.toFixed(1)} · Ceiling {detail.ceiling.toFixed(1)}
          {detail.prob_big_haul !== null && ` · P(10+) ${(detail.prob_big_haul * 100).toFixed(0)}%`}
        </p>
      )}
    </div>
  );
}
