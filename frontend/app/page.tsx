"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { AnalyzeForm } from "@/components/AnalyzeForm";
import { BeatGrid } from "@/components/BeatGrid";
import { ErrorState } from "@/components/ErrorState";
import { Footer } from "@/components/Footer";
import { JobProgress } from "@/components/JobProgress";
import { PlaylistProfile } from "@/components/PlaylistProfile";
import { ResultList } from "@/components/ResultList";
import { ThemeToggle } from "@/components/ThemeToggle";
import { Warnings } from "@/components/Warnings";
import { ApiError, createJob, isMockMode, ping } from "@/lib/api";
import { decodeShare, shareTokenFromHash } from "@/lib/share";
import type { AnalyzeRequest, JobResponse } from "@/lib/types";
import { useJob } from "@/lib/useJob";

function setJobParam(id: string | null) {
  const url = new URL(window.location.href);
  if (id) url.searchParams.set("job", id);
  else url.searchParams.delete("job");
  url.searchParams.delete("vibe"); // the vibe filter belongs to one job
  url.hash = "";
  window.history.pushState(null, "", url);
}

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [waking, setWaking] = useState(false);
  const [submitError, setSubmitError] = useState<ApiError | null>(null);
  const lastReq = useRef<AnalyzeRequest | null>(null);
  const { job, phase, retrying } = useJob(jobId);
  const [shared, setShared] = useState<JobResponse | null>(null);
  const [shareFailed, setShareFailed] = useState(false);

  // Read ?job= on mount and on back/forward; wake the server with a health ping.
  useEffect(() => {
    const read = () => {
      setJobId(new URLSearchParams(window.location.search).get("job"));
      const token = shareTokenFromHash(window.location.hash);
      setShareFailed(false);
      if (!token) return setShared(null);
      decodeShare(token).then(setShared, () => {
        setShared(null);
        setShareFailed(true);
      });
    };
    read();
    void ping();
    window.addEventListener("popstate", read);
    window.addEventListener("hashchange", read);
    return () => {
      window.removeEventListener("popstate", read);
      window.removeEventListener("hashchange", read);
    };
  }, []);

  const start = useCallback(async (req: AnalyzeRequest) => {
    lastReq.current = req;
    setSubmitError(null);
    setSubmitting(true);
    const t = setTimeout(() => setWaking(true), 5000);
    try {
      const acc = await createJob(req);
      setJobParam(acc.job_id);
      setJobId(acc.job_id);
    } catch (e) {
      setSubmitError(e instanceof ApiError ? e : new ApiError(0, "network_error", "Network error"));
    } finally {
      clearTimeout(t);
      setWaking(false);
      setSubmitting(false);
    }
  }, []);

  const reset = useCallback(() => {
    setJobParam(null);
    setJobId(null);
    setShared(null);
    setShareFailed(false);
    setSubmitError(null);
  }, []);

  const showForm = !jobId && !submitting && !shared && !shareFailed;
  const running = (jobId && (phase === "polling" || phase === "idle")) || submitting;

  return (
    <>
      <header className="mx-auto flex max-w-3xl items-center justify-between px-4 pt-5">
        <button type="button" onClick={reset} className="display min-h-11 text-xl font-bold tracking-tight">
          Deepcut
        </button>
        <div className="flex items-center gap-3">
          {isMockMode() && <span className="rounded-full border border-line px-3 py-1 text-xs text-muted">Demo data</span>}
          <ThemeToggle />
        </div>
      </header>

      <main className="mx-auto max-w-3xl space-y-8 px-4 pt-10">
        {showForm && (
          <>
            <section className="space-y-5">
              <BeatGrid className="max-w-sm" />
              <h1 className="display text-balance text-5xl font-bold leading-[0.95] sm:text-7xl">
                Dig past the algorithm.
              </h1>
              <p className="max-w-xl text-lg text-muted">
                Paste a Spotify playlist or a list of tracks. Deepcut listens to its beat flow (tempo, groove, low end) and finds tracks that
                move the same way from artists almost nobody has heard.
              </p>
            </section>
            {submitError && (
              <ErrorState
                code={submitError.code}
                message={submitError.message}
                retryAfter={submitError.retryAfter}
                onRetry={() => (lastReq.current ? void start(lastReq.current) : setSubmitError(null))}
              />
            )}
            <AnalyzeForm onSubmit={start} busy={submitting} />
          </>
        )}

        {shared && (
          <>
            <section className="space-y-1">
              <h1 className="display text-3xl font-semibold">{shared.playlist?.name ?? "Shared picks"}</h1>
              <p className="text-sm text-muted">
                A saved copy of someone&apos;s Deepcut results. It opens without the server, and the play buttons still work.
              </p>
            </section>
            <ResultList job={shared} readOnly />
            <button type="button" onClick={reset} className="display min-h-12 rounded-lg border border-line px-6 font-semibold hover:border-accent">
              Dig your own playlist
            </button>
          </>
        )}
        {shareFailed && <ErrorState code="share_invalid" onRetry={reset} retryLabel="Start a new search" />}

        {running && <JobProgress job={job} waking={waking} retrying={retrying} />}

        {jobId && job && phase !== "expired" && job.playlist_profile && phase !== "done" && (
          <PlaylistProfile playlist={job.playlist} profile={job.playlist_profile} />
        )}

        {jobId && phase === "done" && job && (
          <>
            {job.playlist_profile && <PlaylistProfile playlist={job.playlist} profile={job.playlist_profile} />}
            <ResultList job={job} />
            <button type="button" onClick={reset} className="display min-h-12 rounded-lg border border-line px-6 font-semibold hover:border-accent">
              Dig another playlist
            </button>
          </>
        )}

        {phase === "failed" && job?.error && (
          <>
            <ErrorState
              code={job.error.code}
              message={job.error.message}
              onRetry={reset}
              retryLabel="Try another playlist"
            />
            <Warnings warnings={job.warnings} />
          </>
        )}
        {phase === "expired" && <ErrorState code="job_not_found" onRetry={reset} retryLabel="Start a new search" />}
        {phase === "unreachable" && <ErrorState code="network_error" onRetry={reset} />}
      </main>
      <Footer />
    </>
  );
}
