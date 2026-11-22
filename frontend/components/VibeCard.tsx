import { formatBpm, formatPercent } from "@/lib/format";
import type { Vibe } from "@/lib/types";
import { FeatureBars } from "./FeatureBars";

export function VibeCard({ vibe }: { vibe: Vibe }) {
  return (
    <article className="rounded-xl border border-line bg-surface p-5">
      <header className="flex items-start justify-between gap-4">
        <div>
          <h3 className="display text-lg font-semibold leading-snug">{vibe.label}</h3>
          <p className="num mt-1 text-sm text-muted">
            {vibe.track_count} tracks · {formatBpm(vibe.bpm)} BPM
          </p>
        </div>
        <span className="display num shrink-0 text-3xl font-semibold text-accent-text">{formatPercent(vibe.share)}</span>
      </header>
      <div className="mt-4">
        <FeatureBars features={vibe.features} />
      </div>
      {vibe.example_tracks.length > 0 && (
        <p className="mt-4 text-sm text-muted">
          Like {vibe.example_tracks.map((t) => `${t.title} by ${t.artist}`).join(", ")}
        </p>
      )}
    </article>
  );
}
