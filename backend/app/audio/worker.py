"""Process pool for CPU-bound analysis, with recovery when a worker process dies."""
from __future__ import annotations

import asyncio
import logging
from concurrent.futures import BrokenExecutor, Executor, ProcessPoolExecutor, ThreadPoolExecutor
from typing import Callable

from app.audio.features import extract_features

log = logging.getLogger("deepcut.worker")


def analyze_bytes(data: bytes) -> dict:
    """Top-level (picklable) entry point run inside worker processes."""
    return extract_features(data)


def make_executor(workers: int) -> Executor:
    if workers <= 0:
        return ThreadPoolExecutor(max_workers=1)
    return ProcessPoolExecutor(max_workers=workers)


class AnalysisPool:
    """Wraps an executor. If it breaks (BrokenProcessPool, e.g. OOM-killed worker) it is rebuilt once and the
    clip retried. If the retry breaks too, `failed` is set so /api/health can return 503 and the host restarts us."""

    def __init__(self, workers: int, factory: Callable[[int], Executor] = make_executor,
                 fn: Callable[[bytes], dict] = analyze_bytes):
        self.workers, self.factory, self.fn = workers, factory, fn
        self.failed = False
        self.rebuilds = 0
        self._ex: Executor = factory(workers)

    def _rebuild(self, broken: Executor) -> None:
        if self._ex is not broken:  # another task already replaced it
            return
        self.rebuilds += 1
        log.error("analysis pool broke; rebuilding (rebuild #%d)", self.rebuilds)
        try:
            broken.shutdown(wait=False, cancel_futures=True)
        except Exception:
            log.exception("shutting down broken pool failed")
        try:
            self._ex = self.factory(self.workers)
        except Exception:
            log.exception("rebuilding analysis pool failed")
            self.failed = True
            raise

    async def analyze(self, data: bytes) -> dict:
        loop = asyncio.get_running_loop()
        for attempt in (0, 1):
            ex = self._ex
            try:
                out = await loop.run_in_executor(ex, self.fn, data)
            except BrokenExecutor:
                if attempt == 1:
                    self.failed = True
                    raise
                self._rebuild(ex)
                continue
            self.failed = False
            return out
        raise AssertionError("unreachable")

    def shutdown(self, wait: bool = False, cancel_futures: bool = True) -> None:
        self._ex.shutdown(wait=wait, cancel_futures=cancel_futures)


async def analyze_async(executor, data: bytes) -> dict:
    if isinstance(executor, AnalysisPool):
        return await executor.analyze(data)
    return await asyncio.get_running_loop().run_in_executor(executor, analyze_bytes, data)
