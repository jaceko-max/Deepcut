"""Async token bucket and per-IP job limiter."""
from __future__ import annotations

import asyncio
import time
from collections import defaultdict, deque


class TokenBucket:
    """Allows `capacity` acquisitions per `per` seconds (smooth refill)."""

    def __init__(self, capacity: int, per: float):
        self.capacity = float(capacity)
        self.rate = capacity / per
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self._lock: asyncio.Lock | None = None

    async def acquire(self) -> None:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            while True:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
                self.updated = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                await asyncio.sleep((1 - self.tokens) / self.rate)


class IPJobLimiter:
    def __init__(self, limit: int = 3, window: float = 600.0, clock=time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)

    def check(self, ip: str) -> int | None:
        """Record a hit. Return None if allowed, else seconds until a slot frees."""
        now = self.clock()
        q = self.hits[ip]
        while q and now - q[0] >= self.window:
            q.popleft()
        if len(q) >= self.limit:
            return max(1, int(q[0] + self.window - now) + 1)
        q.append(now)
        return None

    def release(self, ip: str) -> None:
        """Give back the slot taken by the most recent successful check()."""
        q = self.hits.get(ip)
        if q:
            q.pop()
            if not q:
                del self.hits[ip]

    def sweep(self) -> None:
        now = self.clock()
        for ip in list(self.hits):
            q = self.hits[ip]
            while q and now - q[0] >= self.window:
                q.popleft()
            if not q:
                del self.hits[ip]
