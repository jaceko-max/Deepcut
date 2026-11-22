"use client";

import { useEffect, useState } from "react";
import { ApiError, getJob } from "./api";
import type { JobResponse } from "./types";

export type JobPhase = "idle" | "polling" | "done" | "failed" | "expired" | "unreachable";

export const POLL_FAST_MS = 1500;
export const POLL_SLOW_MS = 3000;
export const SLOW_AFTER_MS = 30_000;
const MAX_CONSECUTIVE_NETWORK_FAILURES = 5;

export interface UseJobState {
  job: JobResponse | null;
  phase: JobPhase;
  /** Set when a request failed but we are still retrying. */
  retrying: boolean;
}

/**
 * Polls GET /api/jobs/{id}. Stops on done/error. 404 -> "expired".
 * Interval is 1.5 s, widening to 3 s once polling has lasted 30 s.
 */
export function useJob(jobId: string | null): UseJobState {
  const [state, setState] = useState<UseJobState & { forId: string | null }>({
    job: null,
    phase: "idle",
    retrying: false,
    forId: null,
  });

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const startedAt = Date.now();
    let failures = 0;

    const tick = async () => {
      let next: UseJobState | null = null;
      let again = false;
      try {
        const job = await getJob(jobId);
        failures = 0;
        if (job.status === "done") next = { job, phase: "done", retrying: false };
        else if (job.status === "error") next = { job, phase: "failed", retrying: false };
        else {
          next = { job, phase: "polling", retrying: false };
          again = true;
        }
      } catch (e) {
        if (e instanceof ApiError && e.status === 404) {
          next = { job: null, phase: "expired", retrying: false };
        } else {
          failures += 1;
          if (failures >= MAX_CONSECUTIVE_NETWORK_FAILURES) {
            next = { job: null, phase: "unreachable", retrying: false };
          } else {
            again = true;
            next = null;
            setState((s) => (cancelled ? s : { ...s, retrying: true }));
          }
        }
      }
      if (cancelled) return;
      if (next) {
        const n = next;
        setState({ ...n, forId: jobId });
      }
      if (again) {
        const delay = Date.now() - startedAt >= SLOW_AFTER_MS ? POLL_SLOW_MS : POLL_FAST_MS;
        timer = setTimeout(tick, delay);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [jobId]);

  if (!jobId) return { job: null, phase: "idle", retrying: false };
  if (state.forId !== jobId) return { job: null, phase: "polling", retrying: false };
  return { job: state.job, phase: state.phase, retrying: state.retrying };
}
