import type { JobResponse, JobStage } from "@/lib/types";
import { Warnings } from "./Warnings";

const STEPS: { stage: JobStage; label: string }[] = [
  { stage: "fetching_playlist", label: "Reading your playlist" },
  { stage: "matching_previews", label: "Finding audio previews" },
  { stage: "analyzing_playlist", label: "Listening to your tracks" },
  { stage: "finding_candidates", label: "Digging for similar artists" },
  { stage: "analyzing_candidates", label: "Listening to candidates" },
  { stage: "ranking", label: "Ranking matches" },
];

export function JobProgress({
  job,
  waking = false,
  retrying = false,
}: {
  job: JobResponse | null;
  waking?: boolean;
  retrying?: boolean;
}) {
  const progress = job?.progress ?? 0;
  const pct = Math.round(progress * 100);
  const idx = job ? STEPS.findIndex((s) => s.stage === job.stage) : -1;

  return (
    <section aria-labelledby="progress-h" className="rounded-2xl border border-line bg-surface p-5 sm:p-7">
      <h2 id="progress-h" className="display text-2xl font-semibold">
        {waking ? "Waking up server…" : job?.playlist ? `Listening to “${job.playlist.name}”` : "Starting up…"}
      </h2>
      {waking && (
        <p className="mt-1 text-sm text-muted">
          The free server sleeps when idle and can take up to a minute to wake. Hang tight.
        </p>
      )}

      <div
        role="progressbar"
        aria-label="Analysis progress"
        aria-valuemin={0}
        aria-valuemax={100}
        aria-valuenow={pct}
        className="mt-5 h-3 overflow-hidden rounded-full bg-surface-2"
      >
        <div className="progress-fill h-full rounded-full transition-[width] duration-700" style={{ width: `${Math.max(3, pct)}%` }} />
      </div>
      <p className="num mt-2 flex justify-between text-sm text-muted" aria-live="polite">
        <span>{job?.message ?? (job?.stage === "queued" ? "Waiting for a free worker" : "")}</span>
        <span>{pct}%</span>
      </p>
      {retrying && <p className="mt-2 text-sm text-danger">Connection hiccup. Retrying…</p>}
      {job && <Warnings warnings={job.warnings} />}

      <ol className="mt-5 space-y-2">
        {STEPS.map((s, i) => {
          const state = i < idx ? "done" : i === idx ? "active" : "todo";
          return (
            <li key={s.stage} className="flex items-center gap-3 text-sm" aria-current={state === "active" ? "step" : undefined}>
              <span
                aria-hidden
                className={`grid size-5 place-items-center rounded-full border text-[11px] ${
                  state === "done"
                    ? "border-accent bg-accent text-accent-ink"
                    : state === "active"
                      ? "border-accent-text text-accent-text"
                      : "border-line text-transparent"
                }`}
              >
                {state === "done" ? "✓" : state === "active" ? "•" : "·"}
              </span>
              <span className={state === "todo" ? "text-muted" : "text-fg"}>
                {s.label}
                <span className="sr-only">{state === "done" ? " (done)" : state === "active" ? " (in progress)" : ""}</span>
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}
