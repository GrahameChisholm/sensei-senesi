import { TransferPlanOut, TransferSuggestionOut } from "../api";

interface TransferBannerProps {
  suggestion: TransferSuggestionOut | null;
  loading: boolean;
  error: string | null;
  transfers: number;
  onTransfersChange: (transfers: number) => void;
  onApply: (plan: TransferPlanOut) => void;
  applying: boolean;
}

const MAX_TRANSFER_OPTIONS = [1, 2, 3];

/** Projected finish is shown to one decimal, so a change smaller than this cannot be drawn. A
 * manager comfortably clear of their league (or out of reach of it) has a finish that barely
 * moves whatever they do, and rendering "1.0 → 1.0" as a headline metric would claim a result
 * where there is none. Below this the banner says the finish is unchanged instead. */
const RANK_DISPLAY_EPSILON = 0.05;

function signed(value: number, digits: number = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

function money(tenths: number): string {
  return `£${(Math.abs(tenths) / 10).toFixed(1)}m`;
}

/** Effective ownership read as one word. Above 1 the field is captaining him, so buying him is
 * pure catch-up; near 0 nobody owns him and buying him is where a rank gain has to come from. */
function ownershipLabel(eoMultiplier: number | null): string | null {
  if (eoMultiplier === null) return null;
  if (eoMultiplier >= 1.0) return "template";
  if (eoMultiplier >= 0.5) return "popular";
  if (eoMultiplier > 0.1) return "low owned";
  return "differential";
}

/** Says which criterion actually decided the ranking, not which one it would have used. When the
 * projected finish does not move, the league had nothing to separate the candidates with and
 * expected points broke the tie, so claiming the league drove the choice would be false. */
function VarianceNote({
  suggestion,
  rankMoved,
}: {
  suggestion: TransferSuggestionOut;
  rankMoved: boolean;
}) {
  if (suggestion.n_rivals === 0) {
    return (
      <span className="transfer-banner-note">
        No mini league configured, so these are ranked on expected points alone.
      </span>
    );
  }

  const league = `${suggestion.league_name} (${suggestion.n_rivals} ${
    suggestion.n_rivals === 1 ? "rival" : "rivals"
  })`;

  if (!rankMoved) {
    return (
      <span className="transfer-banner-note">
        Your projected finish in {league} is settled enough that no move on offer shifts it, so
        these are ordered on expected points.
      </span>
    );
  }

  return (
    <span className="transfer-banner-note">
      Ranked by projected finish in {league}.{" "}
      {suggestion.variance_preference === "increase"
        ? "You are projected behind, so differentials that widen your swings are worth more than their points alone."
        : "You are projected ahead, so covering what your rivals own protects the lead better than chasing points."}
    </span>
  );
}

function MoveRow({ move }: { move: TransferPlanOut["moves"][number] }) {
  const ownership = ownershipLabel(move.in_eo_multiplier);
  return (
    <div className="transfer-move">
      <span className="transfer-move-position">{move.position}</span>
      <span className="transfer-move-out">{move.out_name}</span>
      <span className="transfer-move-arrow">→</span>
      <span className="transfer-move-in">{move.in_name}</span>
      {ownership && <span className="transfer-move-tag">{ownership}</span>}
      {move.price_delta !== 0 && (
        <span className="transfer-move-price">
          {move.price_delta > 0 ? "+" : "−"}
          {money(move.price_delta)}
        </span>
      )}
    </div>
  );
}

/** A thin strip under the pitch: the best set of transfers at the manager's chosen count, the
 * expected points it gains, where it moves their projected league finish, and one click to apply
 * it. Everything shown is computed server side by features.transfer_planner, this component only
 * formats it. */
export function TransferBanner({
  suggestion,
  loading,
  error,
  transfers,
  onTransfersChange,
  onApply,
  applying,
}: TransferBannerProps) {
  const plan = suggestion?.plans[0] ?? null;
  const pending = loading || (suggestion === null && error === null);
  const marginal = suggestion?.marginal_points_gains ?? [];
  // What the last transfer in the chosen plan added on its own, so a manager can see whether the
  // second or third move is carrying its weight. Only meaningful past the first.
  const lastMarginal = transfers > 1 ? marginal[transfers - 1] : undefined;
  const rankMoved =
    plan !== null && Math.abs(plan.expected_final_rank_delta) >= RANK_DISPLAY_EPSILON;

  return (
    <div className="transfer-banner">
      <div className="transfer-banner-main">
        <div className="transfer-banner-heading">
          <span className="transfer-banner-title">Best transfer</span>
          <div className="transfer-banner-count">
            {MAX_TRANSFER_OPTIONS.map((count) => (
              <button
                key={count}
                type="button"
                className={count === transfers ? "count-active" : undefined}
                onClick={() => onTransfersChange(count)}
                aria-pressed={count === transfers}
              >
                {count}
              </button>
            ))}
          </div>
        </div>

        {error && <span className="transfer-banner-empty">{error}</span>}
        {/* Pending covers the first render too, before the fetch effect has set `loading`, so the
            empty state never flashes in front of a suggestion that is merely on its way. */}
        {!error && pending && <span className="transfer-banner-empty">Solving…</span>}
        {!error && !pending && !plan && (
          <span className="transfer-banner-empty">
            No transfer improves this squad at your budget.
          </span>
        )}

        {!error && !pending && suggestion && plan && (
          <>
            <div className="transfer-moves">
              {plan.moves.map((move) => (
                <MoveRow key={move.out_player_id} move={move} />
              ))}
            </div>

            {/* Metrics and Apply wrap together as one right-aligned group, so a narrow column
                never strands the button on a line of its own away from the numbers it acts on. */}
            <div className="transfer-banner-right">
              <div className="transfer-metrics">
                <span className="transfer-metric">
                  <strong className={plan.expected_points_delta >= 0 ? "gain" : "loss"}>
                    {signed(plan.expected_points_delta)}
                  </strong>
                  <span className="transfer-metric-label">
                    xP over {suggestion.gameweeks.length === 1 ? "GW" : "GWs"}{" "}
                    {suggestion.gameweeks.join(", ")}
                  </span>
                </span>

                {suggestion.n_rivals > 0 &&
                  (rankMoved ? (
                    <span className="transfer-metric">
                      <strong className={plan.expected_final_rank_delta <= 0 ? "gain" : "loss"}>
                        {suggestion.current_expected_final_rank.toFixed(1)} →{" "}
                        {plan.expected_final_rank.toFixed(1)}
                      </strong>
                      <span className="transfer-metric-label">projected finish</span>
                    </span>
                  ) : (
                    <span className="transfer-metric">
                      <strong>{suggestion.current_expected_final_rank.toFixed(1)}</strong>
                      <span className="transfer-metric-label">projected finish, unmoved</span>
                    </span>
                  ))}

                <span className="transfer-metric">
                  <strong>{money(plan.budget_remaining)}</strong>
                  <span className="transfer-metric-label">left in the bank</span>
                </span>
              </div>

              <button
                type="button"
                className="btn-primary transfer-apply"
                onClick={() => onApply(plan)}
                disabled={applying}
              >
                {applying ? "Applying…" : "Apply"}
              </button>
            </div>
          </>
        )}
      </div>

      {suggestion && !pending && !error && (
        <div className="transfer-banner-footer">
          <VarianceNote suggestion={suggestion} rankMoved={rankMoved} />
          {lastMarginal !== undefined && (
            <span className="transfer-banner-note">
              Transfer {transfers} adds {signed(lastMarginal)} xP on its own.
            </span>
          )}
        </div>
      )}
    </div>
  );
}
