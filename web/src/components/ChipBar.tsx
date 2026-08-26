export type ActiveChip = "bench_boost" | "triple_captain" | null;

interface ChipBarProps {
  activeChip: ActiveChip;
  onChange: (chip: ActiveChip) => void;
}

const CHIPS: { key: "bench_boost" | "triple_captain"; label: string; hint: string }[] = [
  { key: "bench_boost", label: "Bench Boost", hint: "Count bench points toward the total" },
  { key: "triple_captain", label: "Triple Captain", hint: "Triple the captain's points instead of doubling" },
];

/** Bench Boost/Triple Captain as plain, always-available toggles: no scarcity, no "used this
 * half" tracking, no separate play step -- toggling one just changes how the predicted points
 * total (and, for Bench Boost, an auto-build) are computed. Only one can be active at a time,
 * since the points-preview call only ever takes a single chip. */
export function ChipBar({ activeChip, onChange }: ChipBarProps) {
  return (
    <div className="chip-bar">
      {CHIPS.map(({ key, label, hint }) => {
        const active = activeChip === key;
        return (
          <button
            key={key}
            type="button"
            aria-pressed={active}
            className={active ? "chip-active" : ""}
            title={hint}
            onClick={() => onChange(active ? null : key)}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
