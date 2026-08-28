interface GameweekSelectorProps {
  gameweeks: number[];
  currentGameweek: number;
  selected: number | null;
  onSelect: (gameweek: number | null) => void;
}

/** A pill per gameweek already in the app's projection horizon, letting a manager preview a future
 * gameweek's projected points/team without changing anything saved. Shared across the Team
 * Selection and Fixtures pages so the selection carries over between them. */
export function GameweekSelector({
  gameweeks,
  currentGameweek,
  selected,
  onSelect,
}: GameweekSelectorProps) {
  if (gameweeks.length <= 1) return null;

  return (
    <div className="gameweek-selector">
      {gameweeks.map((gameweek) => {
        const isCurrent = gameweek === currentGameweek;
        const active = selected === null ? isCurrent : selected === gameweek;
        return (
          <button
            key={gameweek}
            type="button"
            aria-pressed={active}
            className={active ? "active" : ""}
            title={isCurrent ? "Back to the current gameweek" : `Preview gameweek ${gameweek}`}
            onClick={() => onSelect(isCurrent ? null : gameweek)}
          >
            GW{gameweek}
            {isCurrent && " · Now"}
          </button>
        );
      })}
    </div>
  );
}
