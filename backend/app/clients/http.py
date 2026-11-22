"""Shared httpx wrapper: token bucket, retries with backoff + jitter, hooks."""
from __future__ import annotations

import asyncio
import random
from typing import Callable

import httpx

from app.ratelimit import TokenBucket

BACKOFFS: tuple[float, ...] = (0.5, 1.0, 2.0)  # tests monkeypatch to zeros


class UpstreamError(Exception):
    pass


class HttpClient:
    def __init__(
        self,
        client: httpx.AsyncClient,
        bucket: TokenBucket | None = None,
        retry_on_body: Callable[[httpx.Response], bool] | None = None,
    ):
        self.client = client
        self.bucket = bucket
        self.retry_on_body = retry_on_body

    async def get(self, url: str, params: dict | None = None, headers: dict | None = None) -> httpx.Response:
        return await self.request("GET", url, params=params, headers=headers)

    async def request(self, method: str, url: str, **kw) -> httpx.Response:
        last: str = "unknown"
        for attempt in range(len(BACKOFFS) + 1):
            if self.bucket:
                await self.bucket.acquire()
            retry_after = None
            try:
                resp = await self.client.request(method, url, **kw)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last = f"{type(e).__name__}"
            else:
                if resp.status_code == 429 or resp.status_code >= 500:
                    last = f"HTTP {resp.status_code}"
                    ra = resp.headers.get("retry-after")
                    if ra and ra.isdigit():
                        retry_after = min(float(ra), 5.0)
                elif self.retry_on_body and self.retry_on_body(resp):
                    last = "rate limited (body)"
                else:
                    return resp
            if attempt < len(BACKOFFS):
                delay = BACKOFFS[attempt]
                delay = retry_after if retry_after is not None else delay + random.uniform(0, delay * 0.25)
                await asyncio.sleep(delay)
        raise UpstreamError(f"{method} {url.split('?')[0]} failed after retries: {last}")
