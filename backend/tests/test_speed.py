"""Speed work: clip analysis is deduplicated and can start early (prefetch) without double downloads."""
import asyncio
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx

from app.audio.features import FEATURE_VERSION
from app.clients.deezer import DeezerTrack
from app.pipeline import MAX_CANDIDATE_CLIPS, Pipeline
from tests.test_pipeline_e2e import run_once


class MemCache:
    def __init__(self):
        self.store = {}

    async def get_features(self, key):
        return self.store.get(key)

    async def put_features(self, key, raw):
        self.store[key] = raw


class CountingFetcher:
    def __init__(self, body):
        self.body, self.calls = body, Counter()

    async def get(self, url):
        self.calls[url] += 1
        await asyncio.sleep(0.01)
        return httpx.Response(200, content=self.body)


def make_pipeline(body):
    ex = ThreadPoolExecutor(1)
    svc = SimpleNamespace(cache=MemCache(), fetcher=CountingFetcher(body), executor=ex)
    return Pipeline(svc, "x", 500_000, 10, None), svc, ex


def dz(i):
    return DeezerTrack(id=i, title=f"t{i}", artist="a", artist_id=1, album=None, artwork_url=None,
                       preview_url=f"https://cdn/{i}.mp3", link="l")


async def test_concurrent_requests_for_one_clip_download_once(wavs):
    p, svc, ex = make_pipeline(wavs["cand90"])
    try:
        a, b = await asyncio.gather(p._features("dz:1:v", "https://cdn/1.mp3"), p._features("dz:1:v", "https://cdn/1.mp3"))
        assert a is b and a is not None
        assert svc.fetcher.calls["https://cdn/1.mp3"] == 1
    finally:
        ex.shutdown()


async def test_prefetched_clips_are_reused_by_the_final_stage(wavs):
    p, svc, ex = make_pipeline(wavs["cand90"])
    try:
        p._prefetch_clips([dz(1), dz(2)])
        p._prefetch_clips([dz(2)])  # already in flight
        await asyncio.gather(*p._inflight.values())
        raw = await p._features(f"dz:1:v{FEATURE_VERSION}", "https://cdn/1.mp3")  # the later "analysis" stage
        assert raw is not None
        assert svc.fetcher.calls == Counter({"https://cdn/1.mp3": 1, "https://cdn/2.mp3": 1})
    finally:
        ex.shutdown()


async def test_prefetch_respects_the_clip_budget(wavs):
    p, svc, ex = make_pipeline(wavs["cand90"])
    try:
        p._prefetch_clips([dz(i) for i in range(MAX_CANDIDATE_CLIPS + 20)])
        assert len(p._inflight) == MAX_CANDIDATE_CLIPS
        for t in p._inflight.values():
            t.cancel()
    finally:
        ex.shutdown(wait=False, cancel_futures=True)


async def test_pipeline_downloads_each_clip_once(settings, wavs):
    c = Counter()
    job = await run_once(settings, wavs, c)
    assert job.status == "done", job.error
    # 3 analysed candidate previews (one per artist after contamination filters), each fetched exactly once
    assert c["dz"] == job.stats.candidates_analyzed
