import { SquadOut } from "../api";

const CHIPS: { key: string; label: string }[] = [
  { key: "wildcard", label: "Wildcard" },
  { key: "free_hit", label: "Free Hit" },
  { key: "bench_boost", label: "Bench Boost" },
  { key: "triple_captain", label: "Triple Captain" },
];

interface ChipBarProps {
  squad: SquadOut;
  previewChip: string | null;
  onPreview: (chip: string | null) => void;
  onPlay: (chip: string) => void;
}

export function ChipBar({ squad, previewChip, onPreview, onPlay }: ChipBarProps) {
  return (
    <div className="chip-bar">
      {CHIPS.map(({ key, label }) => {
        const available = squad.chips_available.includes(key);
        const active = squad.active_chip === key;
        const previewing = previewChip === key;
        return (
          <div key={key} className={["chip", active ? "chip-active" : "", !available ? "chip-used" : ""].join(" ")}>
            <button
              disabled={!available && !active}
              className={previewing ? "chip-previewing" : ""}
              onClick={() => onPreview(previewing ? null : key)}
              title={available ? "Preview this chip's effect on your predicted points" : "Already used this half of the season"}
            >
              {label}
              {active && " (active)"}
            </button>
            {previewing && available && (
              <button className="play-chip-button" onClick={() => onPlay(key)}>
                Play {label}
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
