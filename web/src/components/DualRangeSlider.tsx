interface DualRangeSliderProps {
  min: number;
  max: number;
  step: number;
  valueMin: number;
  valueMax: number;
  onChange: (min: number, max: number) => void;
  ariaLabelMin?: string;
  ariaLabelMax?: string;
}

/** One track, two independently-draggable handles -- the two native range inputs sit exactly on
 * top of each other (see the CSS: the track itself is transparent and ignores pointer events, only
 * each input's own thumb re-enables them), rather than the two separate, visually distinct tracks
 * this replaces. The clamp-so-the-handles-can't-cross logic lives here so every caller gets it for
 * free instead of duplicating it. */
export function DualRangeSlider({
  min,
  max,
  step,
  valueMin,
  valueMax,
  onChange,
  ariaLabelMin,
  ariaLabelMax,
}: DualRangeSliderProps) {
  const percent = (value: number) => ((value - min) / (max - min)) * 100;

  return (
    <div className="dual-range-slider">
      <div className="dual-range-track" />
      <div
        className="dual-range-track-fill"
        style={{ left: `${percent(valueMin)}%`, right: `${100 - percent(valueMax)}%` }}
      />
      <input
        type="range"
        className="dual-range-input"
        min={min}
        max={max}
        step={step}
        value={valueMin}
        aria-label={ariaLabelMin}
        onChange={(e) => onChange(Math.min(Number(e.target.value), valueMax), valueMax)}
      />
      <input
        type="range"
        className="dual-range-input"
        min={min}
        max={max}
        step={step}
        value={valueMax}
        aria-label={ariaLabelMax}
        onChange={(e) => onChange(valueMin, Math.max(Number(e.target.value), valueMin))}
      />
    </div>
  );
}
