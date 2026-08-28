interface GameweekStepperProps {
  value: number;
  onChange: (value: number) => void;
  min: number;
  max: number;
}

// One bound of a gameweek window, as a +/- pair -- `min`/`max` is typically the window's other
// bound (clamped to the app's own valid gameweek range), so stepping one bound past the other is
// simply disabled rather than needing to be clamped after the fact.
export function GameweekStepper({ value, onChange, min, max }: GameweekStepperProps) {
  return (
    <span className="gameweek-stepper">
      <button
        type="button"
        className="gameweek-stepper-button"
        onClick={() => onChange(Math.max(min, value - 1))}
        disabled={value <= min}
      >
        −
      </button>
      <span className="gameweek-stepper-value">GW{value}</span>
      <button
        type="button"
        className="gameweek-stepper-button"
        onClick={() => onChange(Math.min(max, value + 1))}
        disabled={value >= max}
      >
        +
      </button>
    </span>
  );
}
