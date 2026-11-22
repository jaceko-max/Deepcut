import { formatCompact } from "@/lib/format";
import type { JobResponse } from "@/lib/types";
import { PreviewProvider } from "./PreviewPlayer";
import { ResultCard } from "./ResultCard";
import { Warnings } from "./Warnings";

export function ResultList({ job }: { job: JobResponse }) {
  const vibes = job.playlist_profile?.vibes ?? [];
  const multi = vibes.length > 1;
  const s = job.stats;
  return (
    <section aria-labelledby="results-h" className="space-y-4">
      <div>
        <h2 id="results-h" className="display text-2xl font-semibold">
          {job.results.length} deep cuts
        </h2>
        {s && (
          <p className="num text-sm text-muted">
            Filtered {s.filtered_mainstream} mainstream artists above {formatCompact(job.params.max_listeners)} listeners
            · {s.candidates_analyzed} candidates listened to
            {s.filtered_in_playlist > 0 ? ` · ${s.filtered_in_playlist} already in your playlist` : ""}
          </p>
        )}
      </div>

      <Warnings warnings={job.warnings} />

      {job.results.length === 0 ? (
        <div className="rounded-xl border border-dashed border-line p-6 text-center">
          <p className="font-medium">Nothing came up this deep.</p>
          <p className="mt-1 text-sm text-muted">Raise the listener cap a little and run it again.</p>
        </div>
      ) : (
        <PreviewProvider>
          <ol className="space-y-3">
            {job.results.map((r) => (
              <ResultCard key={r.deezer_id} result={r} vibe={vibes.find((v) => v.id === r.vibe_id)} showVibe={multi} />
            ))}
          </ol>
        </PreviewProvider>
      )}
    </section>
  );
}
