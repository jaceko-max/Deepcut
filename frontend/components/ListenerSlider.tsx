"use client";

import { formatCompact } from "@/lib/format";
import { SLIDER_MAX, SLIDER_MIN, sliderToListeners } from "@/lib/slider";

const TICKS = [
  { t: 0, label: "10K" },
  { t: 50, label: "100K" },
  { t: 100, label: "1M" },
];

export function ListenerSlider({ value, onChange }: { value: number; onChange: (t: number) => void }) {
  const listeners = sliderToListeners(value);
  return (
    <div>
      <div className="flex items-baseline justify-between gap-3">
        <label htmlFor="listeners" className="font-medium">
          Underground depth
        </label>
        <output htmlFor="listeners" className="display num text-2xl font-semibold text-accent-text">
          ≤ {formatCompact(listeners)} listeners
        </output>
      </div>
      <input
        id="listeners"
        className="depth mt-1"
        type="range"
        min={SLIDER_MIN}
        max={SLIDER_MAX}
        step={1}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        style={{ ["--pos" as string]: `${value}%` }}
        aria-valuetext={`at most ${formatCompact(listeners)} Last.fm listeners`}
      />
      <div className="relative -mt-1 h-5 text-xs text-muted" aria-hidden>
        {TICKS.map((k) => (
          <span
            key={k.t}
            className="absolute -translate-x-1/2 first:translate-x-0 last:-translate-x-full"
            style={{ left: `${k.t}%` }}
          >
            {k.label}
          </span>
        ))}
      </div>
      <p className="mt-2 max-w-prose text-sm text-muted">
        Only artists with at most this many Last.fm listeners make the list. Lower means more underground; slide
        left to go deeper.
      </p>
    </div>
  );
}
