#!/usr/bin/env python
"""Live smoke test against the real services (not part of the offline suite).

  python scripts/smoke_live.py [PLAYLIST_URL] [--max-listeners N] [--limit N]
      Full pipeline (needs LASTFM_API_KEY in the environment).

  python scripts/smoke_live.py [PLAYLIST_URL] --playlist-only [--clips N]
      Only reads the playlist and analyzes a few Spotify preview MP3s; no Last.fm key needed.
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import httpx  # noqa: E402

from app.audio.features import extract_features  # noqa: E402
from app.clients.http import HttpClient  # noqa: E402
from app.clients.spotify import SpotifyClient, parse_playlist_id  # noqa: E402
from app.config import Settings  # noqa: E402
from app.jobs import JobStore, Reporter  # noqa: E402
from app.pipeline import build_services, run_analysis  # noqa: E402

DEFAULT = "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M"


async def playlist_only(pid: str, clips: int) -> None:
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as c:
        http = HttpClient(c)
        t0 = time.perf_counter()
        data = await SpotifyClient(http, Settings()).fetch_playlist(pid)
        print(f"playlist {data.name!r} by {data.owner}: {len(data.tracks)} tracks via {data.source} "
              f"({time.perf_counter() - t0:.2f}s)")
        n = 0
        for t in data.tracks:
            if not t.preview_url:
                continue
            t0 = time.perf_counter()
            r = await http.get(t.preview_url)
            t_dl = time.perf_counter() - t0
            t0 = time.perf_counter()
            f = extract_features(r.content)
            t_an = time.perf_counter() - t0
            print(f"  {t.primary_artist[:24]:24} - {t.title[:30]:30} bpm_raw={f['bpm_raw']:6.1f} "
                  f"folded={f['bpm_folded']:6.1f} pulse={f['pulse_clarity']:.2f} reg={f['regularity']:.2f} "
                  f"dl={t_dl:.2f}s analyze={t_an:.2f}s ({len(r.content) // 1024} KB)")
            n += 1
            if n >= clips:
                break


async def full(pid: str, max_listeners: int, limit: int) -> None:
    settings = Settings.from_env()
    settings.cache_dir = settings.cache_dir or tempfile.mkdtemp()
    if not settings.lastfm_api_key:
        sys.exit("LASTFM_API_KEY is required for the full pipeline (or use --playlist-only)")
    svc = build_services(settings)
    store = JobStore(settings)
    job = store.create(max_listeners, limit)
    rep = Reporter(store, job.job_id)
    t0 = time.perf_counter()
    try:
        await run_analysis(svc, pid, max_listeners, limit, rep)
    except Exception as e:  # noqa: BLE001
        print("FAILED:", repr(e))
    finally:
        await svc.aclose()
    j = store.get(job.job_id)
    print(f"status={j.status} in {time.perf_counter() - t0:.1f}s stats={j.stats} warnings={j.warnings}")
    if j.playlist_profile:
        for v in j.playlist_profile.vibes:
            print(f"  vibe {v.id}: {v.label} ({v.share:.0%}, {v.track_count} tracks)")
    for r in j.results:
        print(f"  #{r.rank:2} {r.match_score:3} {r.artist} - {r.title} [{r.listeners:,} listeners, {r.bpm} BPM] {r.why}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("playlist", nargs="?", default=DEFAULT)
    ap.add_argument("--playlist-only", action="store_true")
    ap.add_argument("--clips", type=int, default=4)
    ap.add_argument("--max-listeners", type=int, default=200000)
    ap.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()
    pid = parse_playlist_id(a.playlist)
    if not pid:
        sys.exit("not a Spotify playlist link")
    if a.playlist_only:
        asyncio.run(playlist_only(pid, a.clips))
    else:
        asyncio.run(full(pid, a.max_listeners, a.limit))


if __name__ == "__main__":
    main()
