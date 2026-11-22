"""In-memory job store with admission control, and the job runner."""
from __future__ import annotations

import asyncio
import logging
import secrets
import time
from datetime import datetime, timezone
from typing import Awaitable, Callable

from app.config import Settings
from app.models import (JobError, JobParams, JobResponse, PlaylistInfoOut, PlaylistProfile,
                        Result, Stats)

log = logging.getLogger("deepcut.jobs")


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class ServerBusy(Exception):
    pass


class JobStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, JobResponse] = {}
        self._touched: dict[str, float] = {}

    def queued_count(self) -> int:
        return sum(1 for j in self.jobs.values() if j.status == "queued")

    def create(self, max_listeners: int, limit: int) -> JobResponse:
        if self.queued_count() >= self.settings.max_queued_jobs:
            raise ServerBusy()
        jid = secrets.token_hex(8)
        t = now_iso()
        job = JobResponse(job_id=jid, status="queued", stage="queued", progress=0.0, message="Queued",
                          created_at=t, updated_at=t, params=JobParams(max_listeners=max_listeners, limit=limit))
        self.jobs[jid] = job
        self._touched[jid] = time.monotonic()
        return job

    def get(self, jid: str) -> JobResponse | None:
        return self.jobs.get(jid)

    def _touch(self, job: JobResponse) -> None:
        job.updated_at = now_iso()
        self._touched[job.job_id] = time.monotonic()

    def update(self, jid: str, **fields) -> None:
        job = self.jobs.get(jid)
        if not job:
            return
        if "progress" in fields:
            fields["progress"] = round(max(job.progress, min(1.0, float(fields["progress"]))), 4)
        for k, v in fields.items():
            setattr(job, k, v)
        self._touch(job)

    def has_result_id(self, deezer_id: int) -> bool:
        """True if a track with this Deezer id appears in the results of a job currently held."""
        return any(r.deezer_id == deezer_id for j in self.jobs.values() for r in (j.results or []))

    def sweep(self) -> int:
        cutoff = time.monotonic() - self.settings.job_ttl_seconds
        dead = [j for j, t in self._touched.items() if t < cutoff and self.jobs[j].status in ("done", "error")]
        for j in dead:
            self.jobs.pop(j, None)
            self._touched.pop(j, None)
        return len(dead)


class Reporter:
    """What the pipeline uses to publish progress into the store."""

    def __init__(self, store: JobStore, job_id: str):
        self.store, self.job_id = store, job_id

    def stage(self, stage: str, progress: float, message: str | None = None) -> None:
        self.store.update(self.job_id, status="running", stage=stage, progress=progress, message=message)

    def progress(self, progress: float, message: str | None = None) -> None:
        self.store.update(self.job_id, progress=progress, message=message)

    def set_playlist(self, p: PlaylistInfoOut) -> None:
        self.store.update(self.job_id, playlist=p)

    def set_profile(self, p: PlaylistProfile) -> None:
        self.store.update(self.job_id, playlist_profile=p)

    def warn(self, msg: str) -> None:
        job = self.store.get(self.job_id)
        if job:
            self.store.update(self.job_id, warnings=[*job.warnings, msg])

    def done(self, results: list[Result], stats: Stats) -> None:
        self.store.update(self.job_id, status="done", stage="done", progress=1.0, results=results,
                          stats=stats, message=f"Found {len(results)} matches")

    def fail(self, code: str, message: str) -> None:
        self.store.update(self.job_id, status="error", stage="error", message=message,
                          error=JobError(code=code, message=message))


class JobRunner:
    def __init__(self, store: JobStore, settings: Settings):
        self.store, self.settings = store, settings
        self.sem = asyncio.Semaphore(settings.max_concurrent_jobs)
        self.tasks: set[asyncio.Task] = set()

    def submit(self, job_id: str, work: Callable[[Reporter], Awaitable[None]]) -> None:
        task = asyncio.create_task(self._run(job_id, work))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _run(self, job_id: str, work: Callable[[Reporter], Awaitable[None]]) -> None:
        rep = Reporter(self.store, job_id)
        async with self.sem:
            rep.stage("fetching_playlist", 0.0, "Starting")
            try:
                await asyncio.wait_for(work(rep), timeout=self.settings.job_timeout_seconds)
            except asyncio.TimeoutError:
                rep.fail("internal_error", "The analysis timed out. Try a smaller playlist or run it again.")
            except Exception as e:  # pipeline maps expected errors itself
                from app.pipeline import PipelineError

                if isinstance(e, PipelineError):
                    if e.detail:
                        log.warning("job %s failed (%s): %s", job_id, e.code, e.detail)
                    rep.fail(e.code, e.message)
                else:
                    log.exception("job %s crashed", job_id)
                    rep.fail("internal_error", "Something went wrong while analyzing. Please try again.")
