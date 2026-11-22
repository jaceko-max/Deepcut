"""End-to-end analysis pipeline (PLAN.md section 4)."""
from __future__ import annotations

import asyncio
import logging
import random
import re
import statistics
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx
import numpy as np

from app.audio.decode import AudioError
from app.audio.features import FEATURE_VERSION
from app.audio.worker import AnalysisPool, analyze_async
from app.cache import Cache
from app.clients.deezer import DeezerClient, DeezerTrack, deezer_body_rate_limited
from app.clients.http import HttpClient, UpstreamError
from app.clients.lastfm import LastfmClient
from app.clients.spotify import (PlaylistData, PlaylistNotFound, PlaylistTrack, PlaylistUnavailable,
                                 SpotifyClient)
from app.config import Settings
from app.jobs import Reporter
from app.models import (FeatureSummary, PlaylistInfoOut, PlaylistProfile, Result, Stats, TrackRef,
                        Vibe as VibeOut)
from app.ratelimit import TokenBucket
from app.scoring.explain import explain, vibe_label
from app.scoring.normalize import MfccScaler, Norm, normalize
from app.scoring.score import (Scored, candidate_distance, compute_d_ref, distance, match_score,
                               select_diverse)
from app.scoring.vibes import build_vibes, centroid_of
from app.textutil import clean_title, norm_name, norm_title

log = logging.getLogger("deepcut.pipeline")

MAX_PLAYLIST_TRACKS = 200
SAMPLE_SIZE = 40
MIN_LISTENERS = 500
LISTENER_BATCH = 40
MAX_LISTENER_LOOKUPS = 240
JUNK_TAGS = {"seen live", "favorites", "favourites", "female vocalists", "male vocalists", "spotify",
             "albums i own", "all", "under 2000 listeners"}


class PipelineError(Exception):
    def __init__(self, code: str, message: str, detail: str | None = None):
        super().__init__(message)
        self.code, self.message, self.detail = code, message, detail  # detail is logged, never shown to users


@dataclass
class Services:
    settings: Settings
    cache: Cache
    spotify: SpotifyClient
    lastfm: LastfmClient
    deezer: DeezerClient
    fetcher: HttpClient
    executor: object
    client: httpx.AsyncClient
    preview_deezer: DeezerClient | None = None  # separate Deezer bucket for /api/preview lookups

    async def aclose(self) -> None:
        await self.client.aclose()
        self.executor.shutdown(wait=False, cancel_futures=True)


def build_services(settings: Settings) -> Services:
    client = httpx.AsyncClient(timeout=10.0, follow_redirects=True,
                               headers={"User-Agent": settings.http_user_agent},
                               limits=httpx.Limits(max_connections=40))
    cache = Cache(settings.cache_dir)
    plain = HttpClient(client)
    lfm = HttpClient(client, TokenBucket(5 * settings.rate_limit_scale, 1.0))
    dz = HttpClient(client, TokenBucket(40 * settings.rate_limit_scale, 5.0), retry_on_body=deezer_body_rate_limited)
    dz_preview = HttpClient(client, TokenBucket(10 * settings.rate_limit_scale, 5.0), retry_on_body=deezer_body_rate_limited)
    return Services(settings, cache, SpotifyClient(plain, settings),
                    LastfmClient(lfm, cache, settings.lastfm_api_key), DeezerClient(dz, cache),
                    plain, AnalysisPool(settings.analysis_workers), client, DeezerClient(dz_preview, cache))


def sample_tracks(tracks: list[PlaylistTrack], n: int = SAMPLE_SIZE, seed: int = 42) -> list[PlaylistTrack]:
    """Stratified round-robin sample by primary artist."""
    if len(tracks) <= n:
        return list(tracks)
    rng = random.Random(seed)
    groups: "OrderedDict[str, list[PlaylistTrack]]" = OrderedDict()
    for t in tracks:
        groups.setdefault(norm_name(t.primary_artist), []).append(t)
    lists = list(groups.values())
    for g in lists:
        rng.shuffle(g)
    rng.shuffle(lists)
    out: list[PlaylistTrack] = []
    i = 0
    while len(out) < n:
        progressed = False
        for g in lists:
            if i < len(g) and len(out) < n:
                out.append(g[i])
                progressed = True
        if not progressed:
            break
        i += 1
    return out


def spotify_search_url(artist: str, title: str) -> str:
    return "https://open.spotify.com/search/" + quote(f"{artist} {title}", safe="-_.!~*'()")


@dataclass
class Cand:
    name: str
    hits: list = field(default_factory=list)       # (affinity, source)
    tracks: list = field(default_factory=list)     # (affinity, title) from similar_track
    listeners: int = 0

    @property
    def affinity(self) -> float:
        if not self.hits:
            return 0.0
        sources = {s for _, s in self.hits}
        return min(1.0, max(a for a, _ in self.hits) + 0.1 * (len(sources) - 1))

    @property
    def source(self) -> str:
        pri = {"similar_artist": 2, "similar_track": 1, "tag": 0}
        return max(self.hits, key=lambda h: (h[0], pri[h[1]]))[1]


class Pipeline:
    def __init__(self, svc: Services, playlist_id: str, max_listeners: int, limit: int, rep: Reporter):
        self.s, self.pid, self.max_listeners, self.limit, self.rep = svc, playlist_id, max_listeners, limit, rep
        self.dl_sem = asyncio.Semaphore(8)

    # -- audio features -------------------------------------------------
    async def _features(self, key: str, url: str) -> dict | None:
        hit = await self.s.cache.get_features(key)
        if hit and hit.get("version") == FEATURE_VERSION:
            return hit
        try:
            async with self.dl_sem:
                resp = await self.s.fetcher.get(url)
            if resp.status_code != 200 or not resp.content:
                return None
            raw = await analyze_async(self.s.executor, resp.content)  # AnalysisPool rebuilds a dead pool itself
        except (UpstreamError, AudioError, httpx.HTTPError):
            return None
        except Exception:
            log.exception("feature extraction failed for %s", key)
            return None
        await self.s.cache.put_features(key, raw)
        return raw

    async def _gather_progress(self, coros, stage: str, lo: float, hi: float, label: str):
        total = len(coros)
        done = 0
        results = [None] * total

        async def run(i, c):
            nonlocal done
            results[i] = await c
            done += 1
            self.rep.stage(stage, lo + (hi - lo) * done / max(1, total), f"{label} {done}/{total}")

        await asyncio.gather(*(run(i, c) for i, c in enumerate(coros)))
        return results

    # -- main -----------------------------------------------------------
    async def run(self) -> None:
        rep, s = self.rep, self.s
        if not s.settings.lastfm_api_key:
            raise PipelineError("upstream_error", "The service is temporarily unavailable. Please try again later.",
                                "LASTFM_API_KEY is not set")

        # 1. playlist
        rep.stage("fetching_playlist", 0.01, "Reading playlist")
        try:
            data: PlaylistData = await s.spotify.fetch_playlist(self.pid)
        except PlaylistNotFound:
            raise PipelineError("playlist_not_found", "Playlist not found. Make sure it is public.")
        except PlaylistUnavailable as e:
            raise PipelineError("playlist_unavailable", "Could not read this playlist. Make sure it is public and try again.", str(e))
        except UpstreamError as e:
            raise PipelineError("upstream_error", "Spotify is not responding. Please try again in a minute.", str(e))
        tracks = data.tracks[:MAX_PLAYLIST_TRACKS]
        if not tracks:
            raise PipelineError("playlist_empty", "This playlist has no tracks.")
        sampled = sample_tracks(tracks)
        playlist_artists = {norm_name(a) for t in tracks for a in t.artists if a}
        playlist_titles = {norm_title(t.title) for t in tracks}
        rep.stage("fetching_playlist", 0.05, f"Found {len(tracks)} tracks")

        # 2. audio sources for playlist tracks
        rep.stage("matching_previews", 0.05, "Finding previews")
        src: list[tuple[PlaylistTrack, str | None, str | None]] = [(t, None, None) for t in sampled]

        async def resolve(t: PlaylistTrack):
            if t.preview_url:
                return t, f"sp:{t.spotify_id}:v{FEATURE_VERSION}", t.preview_url
            try:
                dz = await s.deezer.find_track(t.primary_artist, clean_title(t.title), t.duration_ms, t.isrc)
            except UpstreamError:
                dz = None
            return (t, f"dz:{dz.id}:v{FEATURE_VERSION}", dz.preview_url) if dz else (t, None, None)

        src = await self._gather_progress([resolve(t) for t in sampled], "matching_previews", 0.05, 0.15, "Matching")

        # 3. analyze playlist tracks
        usable = [(t, k, u) for t, k, u in src if k and u]
        raws = await self._gather_progress([self._features(k, u) for _, k, u in usable],
                                           "analyzing_playlist", 0.15, 0.45, "Analyzing track")
        pl = [(t, r) for (t, _, _), r in zip(usable, raws) if r]
        if len(pl) < 5:
            raise PipelineError("no_previews", "Not enough tracks with audio previews to analyze (need at least 5).")
        if len(pl) < 15:
            rep.warn(f"Only {len(pl)} tracks could be analyzed, so the profile is less reliable.")
        rep.set_playlist(PlaylistInfoOut(id=data.id, name=data.name, owner=data.owner, image_url=data.image_url,
                                         total_tracks=len(data.tracks), analyzed_tracks=len(pl), source=data.source))

        scaler = MfccScaler.fit([r for _, r in pl])
        norms = [normalize(r, scaler) for _, r in pl]
        vibes = build_vibes(norms)
        d_ref = compute_d_ref(norms, vibes)
        rep.set_profile(self._profile(pl, norms, vibes))

        # 4/5. seeds + Last.fm
        rep.stage("finding_candidates", 0.45, "Finding similar artists")
        cands = await self._lastfm_candidates(tracks, pl, norms, vibes)
        considered = len(cands)

        # 6. filters
        before = len(cands)
        cands = {k: c for k, c in cands.items() if k not in playlist_artists and not (set(k.split(" and ")) & playlist_artists)}
        filtered_in_playlist = before - len(cands)
        by_affinity = sorted(cands.values(), key=lambda c: -c.affinity)
        rep.stage("finding_candidates", 0.50, "Checking listener counts")
        sem = asyncio.Semaphore(8)

        async def info(c: Cand):
            async with sem:
                try:
                    c.listeners = await s.lastfm.artist_listeners(c.name) or 0
                except UpstreamError:
                    c.listeners = 0

        wanted_artists = max(30, 2 * self.limit)
        ranked: list[Cand] = []
        keep: list[Cand] = []
        for start in range(0, min(len(by_affinity), MAX_LISTENER_LOOKUPS), LISTENER_BATCH):
            batch = by_affinity[start:start + LISTENER_BATCH]
            await asyncio.gather(*(info(c) for c in batch))
            ranked += batch
            keep += [c for c in batch if MIN_LISTENERS <= c.listeners <= self.max_listeners]
            if len(keep) >= wanted_artists:
                break
        filtered_mainstream = sum(1 for c in ranked if c.listeners > self.max_listeners)
        if not keep:
            raise PipelineError("no_candidates", "No underground candidates found. Try raising the listener cap.")

        # 7. candidate tracks
        rep.stage("finding_candidates", 0.52, "Looking up previews")
        tsem = asyncio.Semaphore(8)
        no_preview = 0

        async def tracks_for(c: Cand) -> list[DeezerTrack]:
            nonlocal no_preview
            async with tsem:
                found: list[DeezerTrack] = []
                try:
                    for _, title in sorted(c.tracks, key=lambda x: -x[0])[:2]:
                        t = await s.deezer.find_track(c.name, title)
                        if t:
                            found.append(t)
                    if not found:
                        aid = await s.deezer.find_artist_id(c.name)
                        if aid:
                            found = (await s.deezer.artist_top(aid, 3, c.name))[:2]
                except UpstreamError:
                    pass
                # Deezer returns features and loose search hits; the listener cap was checked for c.name only.
                found = [t for t in found
                         if norm_name(t.artist) == norm_name(c.name)
                         and norm_name(t.artist) not in playlist_artists
                         and norm_title(t.title) not in playlist_titles]
                if not found:
                    no_preview += 1
                return found

        per = await asyncio.gather(*(tracks_for(c) for c in keep))
        budget = min(150, max(80, 5 * self.limit))
        pool: list[tuple[Cand, DeezerTrack]] = []
        seen_ids: set[int] = set()
        for c, ts in zip(keep, per):
            for t in ts[:2]:
                if t.id not in seen_ids:
                    seen_ids.add(t.id)
                    pool.append((c, t))
        pool = pool[:budget]
        if not pool:
            raise PipelineError("no_candidates", "No candidate tracks with previews were found.")

        # 8. analyze candidates
        craws = await self._gather_progress(
            [self._features(f"dz:{t.id}:v{FEATURE_VERSION}", t.preview_url) for _, t in pool],
            "analyzing_candidates", 0.55, 0.95, "Analyzing candidate")
        good = [(c, t, r) for (c, t), r in zip(pool, craws) if r]
        if not good:
            raise PipelineError("no_candidates", "None of the candidate previews could be analyzed.")

        # 9-11. rank
        rep.stage("ranking", 0.95, "Ranking matches")
        scored: list[Scored] = []
        info_by_key: dict[int, tuple] = {}
        for c, t, r in good:
            n = normalize(r, scaler)
            vid, dc = candidate_distance(n, norms, vibes)
            sc = Scored(key=t.id, artist=t.artist or c.name, vibe_id=vid, score=match_score(dc, d_ref), affinity=c.affinity)
            scored.append(sc)
            info_by_key[t.id] = (c, t, r, n)
        chosen = select_diverse(scored, self.limit, {v.id: v.share for v in vibes})
        results: list[Result] = []
        for rank, sc in enumerate(chosen, 1):
            c, t, r, n = info_by_key[sc.key]
            v = vibes[sc.vibe_id]
            summ = n.summary()
            results.append(Result(
                rank=rank, deezer_id=t.id, title=t.title, artist=t.artist or c.name, album=t.album,
                artwork_url=t.artwork_url, preview_url=t.preview_url, preview_proxy_url=f"/api/preview/{t.id}",
                deezer_url=t.link, spotify_search_url=spotify_search_url(t.artist or c.name, t.title),
                listeners=c.listeners, match_score=sc.score, bpm=summ["bpm"], vibe_id=v.id,
                why=explain(n, n.bpm_raw, v.centroid, v.bpm), source=c.source, features=FeatureSummary(**summ)))
        stats = Stats(candidates_considered=considered, candidates_analyzed=len(good),
                      filtered_mainstream=filtered_mainstream, filtered_in_playlist=filtered_in_playlist,
                      no_preview=no_preview)
        rep.done(results, stats)

    # -- helpers --------------------------------------------------------
    def _profile(self, pl, norms: list[Norm], vibes) -> PlaylistProfile:
        folded = [n.bpm_folded for n in norms]
        med = float(statistics.median(folded))
        lo, hi = float(np.percentile(folded, 10)), float(np.percentile(folded, 90))
        overall = centroid_of(norms)
        out = []
        for v in vibes:
            cen = v.centroid
            ordered = sorted(v.members, key=lambda j: distance(norms[j], cen))[:3]
            out.append(VibeOut(
                id=v.id, label=vibe_label(cen, v.bpm, overall if len(vibes) > 1 else None), share=round(v.share, 3), bpm=round(v.bpm, 1),
                track_count=len(v.members), features=FeatureSummary(**cen.summary(bpm=v.bpm)),
                example_tracks=[TrackRef(title=pl[j][0].title, artist=pl[j][0].primary_artist) for j in ordered]))
        return PlaylistProfile(bpm_median=round(med, 1), bpm_range=(round(lo, 1), round(hi, 1)),
                               features=FeatureSummary(**overall.summary(bpm=med)), vibes=out)

    async def _lastfm_candidates(self, tracks, pl, norms, vibes) -> dict[str, Cand]:
        lf = self.s.lastfm
        counts = Counter(a for t in tracks for a in t.artists if a)
        seed_artists = [a for a, _ in counts.most_common(15)]
        seed_tracks: list[tuple[str, str]] = []
        seen = set()
        for v in vibes:
            q = max(1, round(20 * v.share))
            for j in sorted(v.members, key=lambda j: distance(norms[j], v.centroid))[:q]:
                t = pl[j][0]
                k = (norm_name(t.primary_artist), norm_title(t.title))
                if k not in seen:
                    seen.add(k)
                    seed_tracks.append((t.primary_artist, clean_title(t.title)))
        seed_tracks = seed_tracks[:20]

        cands: dict[str, Cand] = {}

        def add(name: str, aff: float, source: str, title: str | None = None):
            if not name:
                return
            c = cands.setdefault(norm_name(name), Cand(name))
            c.hits.append((aff, source))
            if title:
                c.tracks.append((aff, title))

        failures = 0
        sem = asyncio.Semaphore(8)

        async def guard(coro):
            nonlocal failures
            async with sem:
                try:
                    return await coro
                except UpstreamError as e:
                    failures += 1
                    self._last_err = str(e)
                    return None

        sim_a = await asyncio.gather(*(guard(lf.artist_similar(a, 50)) for a in seed_artists))
        for res in sim_a:
            for name, m in res or []:
                add(name, m, "similar_artist")
        self.rep.progress(0.48, "Finding similar tracks")
        sim_t = await asyncio.gather(*(guard(lf.track_similar(a, t, 30)) for a, t in seed_tracks))
        for res in sim_t:
            for artist, title, m in res or []:
                add(artist, m, "similar_track", title)
        self.rep.progress(0.50, "Reading tags")
        tag_res = await asyncio.gather(*(guard(lf.artist_top_tags(a)) for a in seed_artists[:10]))
        tagsum: Counter = Counter()
        for res in tag_res:
            for name, cnt in res or []:
                n = name.strip().lower()
                if n in JUNK_TAGS or re.fullmatch(r"\d{4}", n) or re.search(r"under \d+ listeners", n):
                    continue
                tagsum[n] += cnt
        top_tags = [t for t, _ in tagsum.most_common(5)]
        tag_pages = await asyncio.gather(*(guard(lf.tag_top_artists(t, 50, p)) for t in top_tags for p in (2, 3)))
        for res in tag_pages:
            for name in res or []:
                add(name, 0.3, "tag")
        if not cands:
            if failures:
                raise PipelineError("upstream_error", "Last.fm is not responding. Please try again in a minute.",
                                getattr(self, "_last_err", ""))
            raise PipelineError("no_candidates", "Last.fm returned no similar artists for this playlist.")
        if failures:
            self.rep.warn("Some Last.fm lookups failed, so the candidate pool may be smaller than usual.")
        return cands


async def run_analysis(svc: Services, playlist_id: str, max_listeners: int, limit: int, rep: Reporter) -> None:
    await Pipeline(svc, playlist_id, max_listeners, limit, rep).run()
