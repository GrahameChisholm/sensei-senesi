import { TransferRecommendationOut } from "../api";

interface TransferRecommendationBannerProps {
  recommendation: TransferRecommendationOut | null;
  onApply: () => void;
  applying: boolean;
}

/** The one headline transfer suggestion for whichever squad is currently loaded, always
 * rendered (including an explicit empty state) so its absence never reads as broken. Applying
 * is a single click, no confirmation step -- this page treats every transfer as immediate. */
export function TransferRecommendationBanner({ recommendation, onApply, applying }: TransferRecommendationBannerProps) {
  if (!recommendation) {
    return (
      <div className="transfer-recommendation-banner empty">
        <span>No transfer recommended this gameweek.</span>
      </div>
    );
  }

  return (
    <div className="transfer-recommendation-banner">
      <div className="transfer-recommendation-text">
        <span className="transfer-recommendation-swap">
          {recommendation.sell_player_name} <span className="arrow">→</span> {recommendation.buy_player_name}
        </span>
        <span className="transfer-recommendation-reasoning">{recommendation.reasoning}</span>
      </div>
      <button className="btn-primary" onClick={onApply} disabled={applying}>
        Apply transfer
      </button>
    </div>
  );
}
