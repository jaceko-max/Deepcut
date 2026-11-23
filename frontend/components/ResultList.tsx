"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { copyText, downloadFile, toCsv, toPlainText } from "@/lib/exporters";
import { formatCompact } from "@/lib/format";
import { EMPTY_FEEDBACK, type Feedback, loadFeedback, rerank, saveFeedback, undoLast, vote } from "@/lib/rerank";
import { moreFromStart } from "@/lib/results";
import { buildShareUrl } from "@/lib/share";
import type { JobResponse, Result } from "@/lib/types";
import { countByVibe, filterByVibe, parseVibeParam, type VibeSelection, withVibeParam } from "@/lib/vibe";
import { PreviewProvider } from "./PreviewPlayer";
import { ResultCard } from "./ResultCard";
import { VibeFilter } from "./VibeFilter";
import { Warnings } from "./Warnings";

const actionBtn =
  "inline-flex min-h-11 items-center rounded-lg border border-line px-4 text-sm hover:border-accent disabled:opacity-60";

export function ResultList({ job, readOnly = false }: { job: JobResponse; readOnly?: boolean }) {
  const vibes = useMemo(() => job.playlist_profile?.vibes ?? [], [job.playlist_profile]);
  const multi = vibes.length > 1;
  const s = job.stats;
  const [vibe, setVibe] = useState<VibeSelection>(null);
  const [fb, setFb] = useState<Feedback>(EMPTY_FEEDBACK);
  const [toast, setToast] = useState<string | null>(null);
  const [shareUrl, setShareUrl] = useState<string | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);

  // ?vibe= (alongside ?job=) and saved feedback are read after mount and on back/forward.
  useEffect(() => {
    const read = () => {
      setVibe(parseVibeParam(window.location.search, vibes));
      if (!readOnly) setFb(loadFeedback(job.job_id));
    };
    read();
    window.addEventListener("popstate", read);
    return () => window.removeEventListener("popstate", read);
  }, [vibes, job.job_id, readOnly]);

  useEffect(() => () => clearTimeout(toastTimer.current), []);

  const say = useCallback((msg: string) => {
    setToast(msg);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 4000);
  }, []);

  const selectVibe = (v: VibeSelection) => {
    setVibe(v);
    window.history.replaceState(null, "", withVibeParam(window.location.href, v));
  };

  const changeFeedback = (next: Feedback) => {
    setFb(next);
    if (!readOnly) saveFeedback(job.job_id, next);
  };

  const refined = useMemo(() => rerank(job.results, job.reserve ?? [], fb), [job.results, job.reserve, fb]);
  const items = refined.items;
  const shown = filterByVibe(items, vibe);
  const split = moreFromStart(shown);
  const counts = useMemo(() => countByVibe(items), [items]);
  const refinedNote = refined.liked + refined.disliked > 0;

  async function share() {
    const url = await buildShareUrl(
      { playlist: job.playlist, vibes, results: items, maxListeners: job.params.max_listeners },
      window.location.href,
    );
    if (await copyText(url)) {
      setShareUrl(null);
      say("Link copied. It opens these results without the server.");
    } else {
      setShareUrl(url);
      say("Couldn't copy automatically. Copy the link below.");
    }
  }

  async function copyList() {
    say((await copyText(toPlainText(items))) ? `Copied ${items.length} tracks.` : "Couldn't copy the list.");
  }

  const card = (r: Result) => (
    <ResultCard
      key={r.deezer_id}
      result={r}
      vibe={vibes.find((v) => v.id === r.vibe_id)}
      showVibe={multi}
      feedback={fb.votes[r.deezer_id]}
      onVote={readOnly ? undefined : (v) => changeFeedback(vote(fb, r.deezer_id, v))}
      fresh={refined.pulledIn.has(r.deezer_id)}
    />
  );

  return (
    <section aria-labelledby="results-h" className="space-y-4">
      <div>
        <h2 id="results-h" className="display text-2xl font-semibold">
          {items.length} picks {readOnly ? "shared with you" : "for you"}
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

      {items.length === 0 && !refinedNote ? (
        <div className="rounded-xl border border-dashed border-line p-6 text-center">
          <p className="font-medium">Nothing came up this deep.</p>
          <p className="mt-1 text-sm text-muted">Raise the listener cap a little and run it again.</p>
        </div>
      ) : (
        <>
          <div className="flex flex-wrap gap-2">
            <button type="button" onClick={() => void share()} className={actionBtn}>
              Share results
            </button>
            <button type="button" onClick={() => void copyList()} className={actionBtn}>
              Copy list
            </button>
            <button type="button" onClick={() => downloadFile("deepcut-picks.txt", toPlainText(items) + "\n", "text/plain")} className={actionBtn}>
              Download .txt
            </button>
            <button type="button" onClick={() => downloadFile("deepcut-picks.csv", toCsv(items), "text/csv")} className={actionBtn}>
              Download .csv
            </button>
          </div>
          {shareUrl && (
            <input
              readOnly
              value={shareUrl}
              aria-label="Share link"
              onFocus={(e) => e.currentTarget.select()}
              className="min-h-11 w-full rounded-lg border border-line bg-bg px-3 text-sm"
            />
          )}

          {multi && <VibeFilter vibes={vibes} value={vibe} counts={counts} onChange={selectVibe} />}

          {refinedNote && (
            <div role="status" className="flex flex-wrap items-center gap-x-3 rounded-lg border border-line bg-surface-2 px-3 text-sm">
              <p className="py-2">
                Refined from your feedback: {refined.liked} liked, {refined.disliked} dropped
                {refined.pulledIn.size > 0 ? `, ${refined.pulledIn.size} new` : ""}.
              </p>
              <button type="button" onClick={() => changeFeedback(undoLast(fb))} className="min-h-11 underline underline-offset-4 hover:text-accent-text">
                Undo last
              </button>
              <button type="button" onClick={() => changeFeedback(EMPTY_FEEDBACK)} className="min-h-11 underline underline-offset-4 hover:text-accent-text">
                Reset
              </button>
            </div>
          )}

          {shown.length === 0 ? (
            <p className="rounded-xl border border-dashed border-line p-6 text-center text-sm text-muted">
              No picks left in this vibe. Choose another vibe or reset your feedback.
            </p>
          ) : (
            <PreviewProvider>
              <ol className="space-y-3">{shown.slice(0, split).map(card)}</ol>
              {split < shown.length && (
                <>
                  <h3 className="display pt-4 text-lg font-semibold">More from these artists</h3>
                  <ol start={shown[split].rank} className="space-y-3">
                    {shown.slice(split).map(card)}
                  </ol>
                </>
              )}
            </PreviewProvider>
          )}
        </>
      )}

      <div role="status" aria-live="polite" className="pointer-events-none fixed inset-x-4 bottom-4 z-50 mx-auto max-w-md">
        {toast && <p className="rounded-lg border border-accent bg-surface px-4 py-3 text-sm shadow-lg">{toast}</p>}
      </div>
    </section>
  );
}
