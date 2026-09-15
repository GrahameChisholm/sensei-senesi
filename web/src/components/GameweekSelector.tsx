interface GameweekSelectorProps {
  gameweeks: number[];
  currentGameweek: number;
  selected: number | null;
  onSelect: (gameweek: number | null) => void;
  /** How many transfers each gameweek's own plan has made, keyed by gameweek -- shown as a small
   * badge per pill. A gameweek missing from this map (still loading) shows no badge. */
  transferCounts: Record<number, number>;
}

/** A pill per gameweek already in the app's projection horizon, letting a manager preview a future
 * gameweek's projected points/team without changing anything saved. Shared across the Team
 * Selection and Fixtures pages so the selection carries over between them.
 *
 * On the Team Selection page this doubles as the week-on-week simulator's own gameweek switch:
 * each pill is also which gameweek's squad plan add/remove/captain/etc. edits land on, and the
 * badge shows how many transfers that gameweek's plan has made of its own (0 while it's still
 * live-following an earlier gameweek). */
export function GameweekSelector({
  gameweeks,
  currentGameweek,
  selected,
  onSelect,
  transferCounts,
}: GameweekSelectorProps) {
  if (gameweeks.length <= 1) return null;

  return (
    <div className="gameweek-selector">
      {gameweeks.map((gameweek) => {
        const isCurrent = gameweek === currentGameweek;
        const active = selected === null ? isCurrent : selected === gameweek;
        const transfers = transferCounts[gameweek];
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
            {!!transfers && (
              <span className="transfer-count-badge">
                {transfers} transfer{transfers === 1 ? "" : "s"}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}
