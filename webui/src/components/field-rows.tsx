"use client";

import { useId } from "react";
import { Badge, NumberInput, Slider } from "yunui";

/**
 * The two settings-panel rows that every playground page needs: a label with a
 * live value readout, over (or beside) the control that changes it.
 *
 * They lived as near-identical local copies in `completions` and `audio`, and
 * both had the same defect: the label was a `<span>` or a bare `<label>` with
 * nothing tying it to the control. axe reported seven unnamed sliders and a
 * pile of unnamed number fields across the app — a screen reader announced
 * "slider, 0.7" with no idea what it set. Naming happens here now, once, so a
 * page cannot forget it.
 */

interface RowProps {
  label: string;
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
}

/** Label + value readout above a slider. */
export function SliderRow({
  label,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.01,
  variant = "plain",
}: RowProps & { min?: number; max?: number; step?: number; variant?: "plain" | "badge" }) {
  const id = useId();
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between text-sm">
        <label htmlFor={id} className={variant === "badge" ? "font-medium" : "text-muted-foreground"}>
          {label}
        </label>
        {variant === "badge" ? (
          <Badge variant="info">{value}</Badge>
        ) : (
          <span className="font-mono tabular-nums">{value}</span>
        )}
      </div>
      {/* Radix puts the role on an inner thumb, so `htmlFor` alone cannot reach
          it — the Slider needs its own name. */}
      <Slider
        id={id}
        label={label}
        value={[value]}
        min={min}
        max={max}
        step={step}
        onValueChange={(v) => onChange(v[0] ?? value)}
      />
    </div>
  );
}

/** Label beside a compact numeric field. */
export function NumberRow({ label, value, onChange, min, max, step }: RowProps) {
  const id = useId();
  return (
    <div className="flex items-center justify-between gap-3 text-sm">
      <label htmlFor={id} className="text-muted-foreground">
        {label}
      </label>
      <div className="w-28 shrink-0">
        <NumberInput id={id} value={value} onChange={onChange} min={min} max={max} step={step} />
      </div>
    </div>
  );
}
