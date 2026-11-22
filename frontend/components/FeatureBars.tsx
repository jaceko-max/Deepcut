import type { FeatureSummary } from "@/lib/types";

const ROWS: { key: Exclude<keyof FeatureSummary, "bpm">; label: string }[] = [
  { key: "pulse_clarity", label: "Pulse" },
  { key: "onset_density", label: "Drum density" },
  { key: "regularity", label: "Regularity" },
  { key: "syncopation", label: "Syncopation" },
  { key: "brightness", label: "Brightness" },
  { key: "low_end", label: "Low end" },
  { key: "energy", label: "Energy" },
  { key: "dynamics", label: "Dynamics" },
];

/** 0..1 meters. `compare` draws a tick showing the playlist vibe's value. */
export function FeatureBars({ features, compare }: { features: FeatureSummary; compare?: FeatureSummary }) {
  return (
    <dl className="grid grid-cols-[6.5rem_1fr] items-center gap-x-3 gap-y-1.5 text-sm sm:grid-cols-[7.5rem_1fr]">
      {ROWS.map(({ key, label }) => {
        const v = clamp(features[key]);
        const c = compare ? clamp(compare[key]) : null;
        return (
          <div key={key} className="contents">
            <dt className="text-muted">{label}</dt>
            <dd
              role="meter"
              aria-label={label}
              aria-valuemin={0}
              aria-valuemax={1}
              aria-valuenow={Number(v.toFixed(2))}
              className="relative h-2 rounded-full bg-surface-2"
            >
              <span className="absolute inset-y-0 left-0 rounded-full bg-accent" style={{ width: `${v * 100}%` }} />
              {c !== null && (
                <span
                  aria-hidden
                  className="absolute -inset-y-1 w-0.5 bg-fg"
                  style={{ left: `calc(${c * 100}% - 1px)` }}
                  title={`Playlist vibe: ${c.toFixed(2)}`}
                />
              )}
              <span className="sr-only">{v.toFixed(2)}</span>
            </dd>
          </div>
        );
      })}
    </dl>
  );
}

function clamp(n: number) {
  return Math.min(1, Math.max(0, n));
}
