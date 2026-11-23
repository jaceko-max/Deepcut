"""End-to-end analysis pipeline (PLAN.md section 4)."""
from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import statistics
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from urllib.parse import quote

import httpx
import numpy as np

from app.audio.decode import AudioError
from app.audio.embed import enabled as embed_enabled, moods_enabled
from app.audio.features import FEATURE_VERSION
from app.audio.worker import AnalysisPool, analyze_async
from app.cache import Cache
from app.clients.deezer import DeezerClient, DeezerTrack, deezer_body_rate_limited
from app.clients.http import HttpClient, UpstreamError
from app.clients.lastfm import LastfmClient
from app.clients.listenbrainz import ListenBrainzClient
from app.clients.spotify import (PlaylistData, PlaylistNotFound, PlaylistTrack, PlaylistUnavailable,
                                 SpotifyClient)
from app.config import Settings
from app.filters import allowed_categories, artist_is_tribute, is_contaminated
from app.jobs import Reporter
from app.pasted import pasted_playlist, pick_reserve
from app.models import (FeatureSummary, PlaylistInfoOut, PlaylistProfile, Result, Stats, TrackRef,
                        Vibe as VibeOut)
from app.ratelimit import TokenBucket
from app.scoring.explain import deep_cut_why, explain, vibe_labels, with_mood, with_tags
from app.scoring.mood import mood_words, word_scores
from app.scoring.normalize import MfccScaler, Norm, normalize
from app.scoring.score import (Scored, candidate_distance, compute_d_ref, distance, envelope_ratio, match_score,
                               mood_envelope, near_dup_threshold, order_by_score, own_scores, select_diverse)
from app.scoring.tags import (build_profile, shared_tags, similarity, tag_vector, top_profile_tags,
                               JUNK_TAGS as _TAG_JUNK)
from app.scoring.vibes import build_vibes, centroid_of
from app.signing import preview_path
from app.textutil import clean_title, norm_name, norm_title, strict_name

log = logging.getLogger("deepcut.pipeline")

MAX_PLAYLIST_TRACKS = 200
SAMPLE_SIZE = 40
MIN_LISTENERS = 15_000     # floor on Last.fm listeners for NEW artists: below this, results are hard to find
MIN_LISTENERS_MAX_SHARE = 0.25  # ...but never above a quarter of the cap (deep 10K-50K caps still work)
# Popularity mix: about a third each of deep cuts, "rising" artists (RISING_MIN..cap listeners) and
# "underground" artists (floor..RISING_MIN). Quotas fill by score and backfill from other tiers.
RISING_MIN = 100_000
RISING_MIN_CAP_SHARE = 1 / 3  # low caps: the rising band is the top two thirds below the cap
LISTENER_BATCH = 40
MAX_LISTENER_LOOKUPS = 120
MAX_LISTENER_LOOKUPS_HARD = 160
JUNK_TAGS = _TAG_JUNK  # kept for backwards compatibility; see app/scoring/tags.py
SEED_TAG_ARTISTS = 25      # playlist artists whose Last.fm tags define the genre profile
LB_SEEDS = 10              # seed artists sent to ListenBrainz similar-artists
LB_ROWS_PER_SEED = 100
LB_TIMEOUT = 30.0          # whole ListenBrainz stage; on timeout we continue Last.fm-only
LB_USER_SCALE = 10         # Last.fm listeners ~= 10-15x ListenBrainz users (Radiohead, Beach House, ...)
LB_MAINSTREAM_SLACK = 2.0  # LB is a secondary signal: only reject gross mismatches
UNDERGROUND_HOP_SOURCES = 8    # kept artists whose ListenBrainz neighbours are searched for underground artists
UNDERGROUND_HOP_LOOKUPS = 16   # Last.fm listener + tag lookups spent on them (~6 s at Last.fm's 5 req/s)
UNDERGROUND_HOP_TIMEOUT = 20.0
TAG_MIN_SIM = 0.12         # hard reject below this cosine to the playlist/vibe tag profile
TAG_FULL_SIM = 0.60        # cosine treated as a perfect genre fit in the blended score
UNTAGGED_MIN_AFFINITY = 0.6  # artists with no usable tags survive only with strong similarity evidence
UNTAGGED_TAG_SCORE = 0.25
# Displayed match_score = 0.80*audio + 0.20*100*tag_fit, where audio = 0.53 mood + 0.29 style + 0.18 DSP
# (app/scoring/score.py): ~42% mood, 23% style, 14% rhythm/tone, 20% Last.fm genre tags. Benchmark: tags at
# 0.35 / 0.20 / 0.15 gave held-out-artist AUC 0.735 / 0.722 / 0.717 and the same R@20 (0.26): genre is kept
# as a light outlier check, mood leads. Without mood heads the old 0.65/0.35 blend applies.
W_AUDIO, W_TAGS = 0.80, 0.20
W_AUDIO_NO_MOOD, W_TAGS_NO_MOOD = 0.65, 0.35
# Quality floor: candidates below it are dropped instead of padding the list (a short list gets a note).
# The floor follows the playlist's own coherence: at most QUALITY_FLOOR, or what the playlist's own tracks
# score against the rest of it (QUALITY_OWN_PCT percentile of leave-one-out audio, with a neutral tag fit).
QUALITY_FLOOR = 70
QUALITY_OWN_PCT = 25
QUALITY_MIN = 55
OWN_TAG_FIT = 0.75
DEEZER_FAN_SLACK = 2.0     # reject when Deezer nb_fan > 2x the listener cap
TRACKS_PER_ARTIST = 5      # Deezer tracks fetched per candidate artist (after filters)
CLIPS_PER_ARTIST = 2       # analyzed per artist when the budget allows
MAX_CANDIDATE_CLIPS = 72  # was 80; the lower 15K floor keeps more artists, this holds cold-search time
SEED_TRACKS = 12           # track.getSimilar seeds (often empty for new releases; ListenBrainz covers)
# "Deep cuts": lesser-known tracks by the playlist's own core artists (the listener cap does not apply:
# the artist is known, the track is what's obscure).
DEEP_CUT_ARTISTS = 8       # playlist artists (by track count) searched for deep cuts
DEEP_CUT_MIN_TRACKS = 2    # an artist needs >= 2 playlist tracks to count as "an artist you like"
DEEP_CUT_TOP_FETCH = 50    # Deezer /artist/{id}/top?limit=50, ordered by popularity with a `rank` field
DEEP_CUT_SKIP_TOP = 5      # the artist's 5 most popular tracks (by list position and by rank) are hits
DEEP_CUT_RANK_RATIO = 0.60 # Deezer rank is log-like (hits ~800K, obscure ~300K): <= 60% of the artist's best
DEEP_CUT_LFM_RATIO = 0.30  # Last.fm: a deep cut has <= 30% of the listeners of the artist's top track
DEEP_CUT_LFM_TOP = 200     # artist.getTopTracks limit
DEEP_CUT_LFM_MIN_TRACKS = 10  # fewer Last.fm tracks than this: fall back to the Deezer rank ratio
DEEP_CUT_CLIPS = 3         # analyzed per artist (<= 24 extra clips)
DEEP_CUT_PER_ARTIST = 2    # kept per artist in the results
DEEP_CUT_SHARE = (0.25, 1 / 3, 0.40)  # (min, target, max) share of results
DEEP_CUT_MIN_SCORE = 55    # a deep cut below this (or below the quality floor) is not worth a slot
DEEP_CUT_AFFINITY = 0.5    # neutral rank_score weight (no similarity edge involved)
DEEP_CUT_SKIP = re.compile(r"remix|\bmix\)|interlude|\bskit\b|\bintro\b|\boutro\b|a ?cappella|acapella|"
                           r"instrumental|\bdemo\b|\bedit\b|\bversion\b|\breprise\b", re.I)


def min_listeners(cap: int) -> int:
    """Listener floor for new artists: results should be not-mainstream but still findable (15K at the
    default 500K cap, so the underground tier 15K-100K exists)."""
    return int(min(MIN_LISTENERS_MAX_SHARE * cap, MIN_LISTENERS))


def rising_min(cap: int) -> int:
    """Listener count separating "underground" from "rising" artists under this cap."""
    return int(max(min_listeners(cap), min(RISING_MIN, RISING_MIN_CAP_SHARE * cap)))


def popularity_tier(listeners: int, cap: int) -> str:
    return "rising" if listeners >= rising_min(cap) else "underground"


def tier_quotas(n_discovery: int) -> dict[str, int]:
    """Discovery slots split evenly between rising and underground (rising gets the odd one)."""
    return {"rising": n_discovery - n_discovery // 2, "underground": n_discovery // 2}


def interleave_tiers(cands: list, cap: int) -> list:
    """Alternate rising and underground candidates (each keeping its order) so both tiers get analysis budget."""
    groups = {"rising": [], "underground": []}
    for c in cands:
        groups[popularity_tier(c.listeners, cap)].append(c)
    out, a, b = [], groups["rising"], groups["underground"]
    for i in range(max(len(a), len(b))):
        out += a[i:i + 1] + b[i:i + 1]
    return out


def quality_floor(own: list[int], w_audio: float, w_tags: float) -> int:
    """Minimum displayed score for a result (see QUALITY_FLOOR)."""
    if not own:
        return QUALITY_FLOOR
    own_level = w_audio * float(np.percentile(own, QUALITY_OWN_PCT)) + w_tags * 100 * OWN_TAG_FIT
    return int(round(max(QUALITY_MIN, min(QUALITY_FLOOR, own_level))))


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
    listenbrainz: ListenBrainzClient | None = None

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
    lb = HttpClient(client, TokenBucket(5 * settings.rate_limit_scale, 1.0))
    return Services(settings, cache, SpotifyClient(plain, settings),
                    LastfmClient(lfm, cache, settings.lastfm_api_key), DeezerClient(dz, cache),
                    plain, AnalysisPool(settings.analysis_workers), client, DeezerClient(dz_preview, cache),
                    ListenBrainzClient(lb, cache))


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


NON_ARTIST = re.compile(r"various artists|soundtrack|\bost\b|original (motion picture|cast|score)|grand theft auto|"
                        r"\bcast of\b|^disney\b|^(the )?(video ?game|movie|film)\b", re.I)


def pick_seeds(tracks: list[PlaylistTrack], pl, norms, vibes, small: int = 15, large: int = 20) -> tuple[list[str], Counter]:
    """Seed artists: PRIMARY artists only (featured credits like SZA on someone else's single are not
    what the playlist is about), non-artists dropped (soundtracks, game/film titles, tribute acts).
    Ties - the norm on singleton-heavy playlists - are broken by how close the artist's analyzed track
    sits to its vibe centroid instead of playlist order; singleton-heavy playlists get more seeds."""
    counts = Counter(t.primary_artist for t in tracks
                     if t.primary_artist and not NON_ARTIST.search(t.primary_artist) and not artist_is_tribute(t.primary_artist))
    best: dict[str, float] = {}
    for v in vibes:
        for j in v.members:
            a = pl[j][0].primary_artist
            best[a] = min(best.get(a, 9.0), distance(norms[j], v.centroid))
    order = sorted(counts, key=lambda a: (-counts[a], best.get(a, 9.0), a))
    singles = sum(1 for n in counts.values() if n == 1) / max(1, len(counts))
    return order[:large if singles >= 0.6 else small], counts


def deep_cut_artists(counts: Counter, n: int = DEEP_CUT_ARTISTS) -> list[str]:
    """Core playlist artists (primary, non-soundtrack/tribute: `counts` from pick_seeds) with at least
    DEEP_CUT_MIN_TRACKS tracks, most tracks first."""
    return [a for a in sorted(counts, key=lambda a: (-counts[a], a)) if counts[a] >= DEEP_CUT_MIN_TRACKS][:n]


@dataclass
class LastfmPopularity:
    """Per-track Last.fm listeners for one artist (artist.getTopTracks), indexed by normalised title."""
    listeners: dict[str, int]
    top: int          # listeners of the artist's most-listened track
    floor: int        # smallest listener count in the (possibly truncated) list

    @classmethod
    def build(cls, rows: list[tuple[str, int, int]]) -> "LastfmPopularity | None":
        idx: dict[str, int] = {}
        for title, listeners, _ in rows:
            k = norm_title(title)
            if k and listeners > idx.get(k, -1):
                idx[k] = listeners
        if len(idx) < DEEP_CUT_LFM_MIN_TRACKS or max(idx.values()) <= 0:
            return None
        return cls(idx, max(idx.values()), min(idx.values()))

    def ratio(self, title: str) -> float:
        """Track listeners / the artist's top track. Titles missing from the list are rarer than its last
        entry, so they get the floor ratio (an upper bound)."""
        n = self.listeners.get(norm_title(title))
        return (self.floor if n is None else n) / self.top


def pick_deep_cuts(artist: str, top: list[DeezerTrack], playlist_titles: set[str], playlist_isrcs: set[str],
                   allowed: set[str] = frozenset(), n: int = DEEP_CUT_CLIPS,
                   lfm: "LastfmPopularity | None" = None) -> list[DeezerTrack]:
    """Lesser-known tracks by `artist` from its Deezer top list (popularity order): not in the playlist,
    not among its DEEP_CUT_SKIP_TOP hits (by position or by rank), clean (no live/holiday/sped-up/remix/
    interlude...), one version per base title, and obscure enough: with Last.fm data, track listeners
    <= DEEP_CUT_LFM_RATIO x the artist's top track (Last.fm counts unique listeners, which separates album
    tracks from hits far better than Deezer's log-like rank); without it, Deezer rank <= DEEP_CUT_RANK_RATIO
    x the artist's best."""
    own = [t for t in top if norm_name(t.artist) == norm_name(artist)]  # drop features on others' songs
    if not own:
        return []
    ranks = [t.rank for t in own if t.rank]
    best = max(ranks) if ranks else None
    hits = set(t.id for t in own[:DEEP_CUT_SKIP_TOP])
    hits |= {t.id for t in sorted((t for t in own if t.rank), key=lambda t: -t.rank)[:DEEP_CUT_SKIP_TOP]}
    # Versions of hits / playlist tracks ("Poison" on another album, "X (Remix)") are not deep cuts.
    seen = {norm_title(t.title) for t in own if t.id in hits} | set(playlist_titles)
    out: list[DeezerTrack] = []
    for t in own:
        base = norm_title(t.title)
        if t.id in hits or base in seen or (t.isrc and t.isrc in playlist_isrcs):
            continue
        if lfm is not None:
            if lfm.ratio(t.title) > DEEP_CUT_LFM_RATIO:
                continue
        elif best is None or not t.rank or t.rank > DEEP_CUT_RANK_RATIO * best:
            continue
        if DEEP_CUT_SKIP.search(t.title) or is_contaminated(t.title, t.album, t.artist, allowed):
            continue  # a rejected live take must not block the studio version further down
        seen.add(base)
        out.append(t)
        if len(out) >= n:
            break
    return out


def deep_cut_slots(limit: int) -> tuple[int, int, int]:
    """(min, target, max) number of deep cuts in a list of `limit` results."""
    lo, mid, hi = DEEP_CUT_SHARE
    return math.ceil(lo * limit), round(mid * limit), math.floor(hi * limit)


def merge_deep_cuts(discovery: list[Scored], deep: list[Scored], limit: int) -> list[Scored]:
    """Final list: discoveries (already diversity-selected, best first) plus deep cuts at about
    DEEP_CUT_SHARE of the list, <= DEEP_CUT_PER_ARTIST per artist, one per artist first. Ordered by
    score with at most one track per artist in the top 10. Deep cuts backfill a short discovery list."""
    rounds: list[list[Scored]] = [[] for _ in range(DEEP_CUT_PER_ARTIST)]
    per: dict[str, int] = {}
    for d in sorted(deep, key=lambda x: -x.rank_score):
        a = d.artist.lower()
        if per.get(a, 0) < DEEP_CUT_PER_ARTIST:
            rounds[per.get(a, 0)].append(d)
            per[a] = per.get(a, 0) + 1
    ordered_deep = [d for r in rounds for d in r]
    target = min(len(ordered_deep), deep_cut_slots(limit)[1])
    disc = discovery[:limit - target]
    k = min(len(ordered_deep), limit - len(disc))
    return order_by_score(disc + ordered_deep[:k])[:limit]


def interleave_families(cands: list, weight: dict[str, int]) -> list:
    """Round-robin across seed families, keeping each family's internal order. Families of artists with
    more playlist tracks get more turns per round (sqrt of their track count)."""
    groups: dict = {}
    for c in cands:
        groups.setdefault(c.family, []).append(c)
    order = sorted(groups, key=lambda f: -weight.get(f, 1) if f else 0)
    turns = {f: max(1, round(math.sqrt(weight.get(f, 1)))) if f else 1 for f in order}
    out: list = []
    while any(groups.values()):
        for f in order:
            take, groups[f] = groups[f][:turns[f]], groups[f][turns[f]:]
            out += take
    return out


def candidate_support(c: "Cand", weight: dict[str, int]) -> float:
    """Unsaturated evidence for a candidate: the similarity each distinct seed gave it, summed, with Last.fm
    seeds weighted by sqrt(their playlist track count), plus a bonus when Last.fm and ListenBrainz agree.
    `affinity` caps at 1.0, so on most playlists dozens of candidates tie at the top; ordering the
    rate-limited listener lookups by support instead puts artists that many playlist artists point at
    first. Held-out benchmark (README): held-out playlist artists among the scored candidates 0.40 -> 0.42,
    in the top 20 0.23 -> 0.28, at no extra API cost."""
    tot = 0.0
    for seed, aff in c.seed_aff.items():
        w = 1.0 if seed.startswith("lb:") else math.sqrt(weight.get(seed, 1))
        tot += w * aff
    if not c.seed_aff:  # tag-only candidates
        tot = max((a for a, _ in c.hits), default=0.0)
    return tot + (0.5 if len(c.origins) > 1 else 0.0)


def spotify_search_url(artist: str, title: str) -> str:
    return "https://open.spotify.com/search/" + quote(f"{artist} {title}", safe="-_.!~*'()")


@dataclass
class Cand:
    name: str
    hits: list = field(default_factory=list)       # (affinity, source)
    tracks: list = field(default_factory=list)     # (affinity, title) from similar_track
    listeners: int = 0
    origins: set = field(default_factory=set)      # "lastfm" / "listenbrainz"
    mbid: str | None = None
    tags: dict | None = None                       # cleaned Last.fm tag vector (None = not fetched)
    tag_sim: float | None = None                   # best cosine vs playlist/vibe profiles (None = untagged)
    seeds: set = field(default_factory=set)        # distinct seed artists pointing at this candidate
    lb_users: int | None = None                    # ListenBrainz total_user_count, if known
    seed_aff: dict = field(default_factory=dict)   # seed artist -> strongest affinity it gave

    @property
    def family(self) -> str | None:
        """The playlist artist that points at this candidate most strongly."""
        return max(self.seed_aff, key=self.seed_aff.get) if self.seed_aff else None

    @property
    def affinity(self) -> float:
        if not self.hits:
            return 0.0
        sources = {s for _, s in self.hits}
        both = 0.15 if len(self.origins) > 1 else 0.0  # found by Last.fm AND ListenBrainz
        multi = min(0.2, 0.05 * max(0, len(self.seeds) - 1))  # several seeds agree: strong coherence signal
        return min(1.0, max(a for a, _ in self.hits) + 0.1 * (len(sources) - 1) + both + multi)

    @property
    def source(self) -> str:
        pri = {"similar_artist": 2, "similar_track": 1, "tag": 0}
        return max(self.hits, key=lambda h: (h[0], pri[h[1]]))[1]


@dataclass
class DeepCutArtist(Cand):
    """A playlist artist whose lesser-known tracks are offered as deep cuts (scored like a candidate)."""
    best_rank: int | None = None
    track_ratio: dict = field(default_factory=dict)  # deezer id -> Last.fm listeners / artist's top track


class Pipeline:
    def __init__(self, svc: Services, playlist_id: str, max_listeners: int, limit: int, rep: Reporter):
        self.s, self.pid, self.max_listeners, self.limit, self.rep = svc, playlist_id, max_listeners, limit, rep
        self._artist_tags: dict[str, dict] = {}
        self.dl_sem = asyncio.Semaphore(8)
        self._ticks: dict[str, int] = {}
        self.pasted: list[dict] | None = None  # pasted track list instead of a Spotify playlist
        # In-process instrumentation for benchmarks (scratch scripts set a dict); never part of API output.
        self.trace: dict | None = None
        self._want_style = embed_enabled()  # cached clips without the style profile are re-analysed
        self._want_mood = moods_enabled()
        self._vibe_words: dict[int, list[str]] = {}   # ...and, with the mood heads installed, without mood outputs

    # -- audio features -------------------------------------------------
    async def _features(self, key: str, url: str) -> dict | None:
        hit = await self.s.cache.get_features(key)
        if (hit and hit.get("version") == FEATURE_VERSION and (hit.get("style") or not self._want_style)
                and (hit.get("mood") or not self._want_mood)):
            return hit
        try:
            async with self.dl_sem:
                resp = await self.s.fetcher.get(url)
            if (resp.status_code != 200 or not resp.content) and key.startswith("dz:"):
                # Deezer preview URLs are signed for ~15 minutes, but search/top-list responses are cached
                # for a day: on a feature-cache miss the cached URL is often stale (HTTP 403). Fetch a fresh
                # URL for the track id once and retry.
                fresh = await self.s.deezer.get_track(int(key.split(":")[1]))
                if fresh and fresh.preview_url != url:
                    async with self.dl_sem:
                        resp = await self.s.fetcher.get(fresh.preview_url)
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
        self._dc_task: asyncio.Task | None = None
        try:
            await self._run()
        finally:
            if self._dc_task:
                self._dc_task.cancel()

    async def _run(self) -> None:
        rep, s = self.rep, self.s
        if not s.settings.lastfm_api_key:
            raise PipelineError("upstream_error", "The service is temporarily unavailable. Please try again later.",
                                "LASTFM_API_KEY is not set")

        # 1. playlist
        t_start = time.monotonic()
        marks: dict[str, float] = {}
        rep.stage("fetching_playlist", 0.01, "Reading playlist")
        try:
            data: PlaylistData = (pasted_playlist(self.pasted) if self.pasted
                                  else await s.spotify.fetch_playlist(self.pid))
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

        if self.trace is not None:
            keys = [k for (_, k, _), r in zip(usable, raws) if r]
            self.trace["playlist"] = [(t.primary_artist, t.title, r, k) for (t, r), k in zip(pl, keys)]
        scaler = MfccScaler.fit([r for _, r in pl])
        norms = [normalize(r, scaler) for _, r in pl]
        vibes = build_vibes(norms)
        d_ref = compute_d_ref(norms, vibes)
        rep.set_profile(self._profile(pl, norms, vibes))
        # Mood-first scoring needs mood vectors on the playlist; otherwise the pre-mood blend applies.
        has_mood = all(n.mood is not None for n in norms)
        w_audio, w_tags = (W_AUDIO, W_TAGS) if has_mood else (W_AUDIO_NO_MOOD, W_TAGS_NO_MOOD)
        radii = mood_envelope(norms, vibes)
        floor_score = quality_floor(own_scores(norms, vibes, d_ref), w_audio, w_tags)
        dup_d = near_dup_threshold(norms, [norm_name(t.primary_artist) for t, _ in pl])
        log.info("mood=%s envelope=%s quality floor=%d near-dup distance=%.3f", has_mood,
                 radii and {k: round(v, 3) for k, v in radii.items()}, floor_score, dup_d)
        if self.trace is not None:
            self.trace["floor"], self.trace["envelope"], self.trace["dup_d"] = floor_score, radii, dup_d

        # 4/5. seeds + Last.fm + ListenBrainz, genre-tag profile
        marks["playlist_audio"] = time.monotonic() - t_start
        rep.stage("finding_candidates", 0.45, "Finding similar artists")
        seeds, counts = pick_seeds(tracks, pl, norms, vibes)
        allowed = allowed_categories([t.title for t in tracks])
        # Deep cuts (Deezer + one cached Last.fm call per core artist) run alongside the candidate search.
        dc_task = self._dc_task = asyncio.create_task(self._deep_cuts(deep_cut_artists(counts), tracks, allowed))
        floor = min_listeners(self.max_listeners)
        cap = self.max_listeners
        profiles = await self._tag_profiles(counts, seeds, pl, vibes)
        cands, seed_mbids = await self._candidates(tracks, pl, norms, vibes, seeds, profiles)
        considered = len(cands)
        if self.trace is not None:
            self.trace["pool"] = {k: {"name": c.name, "affinity": c.affinity, "sources": sorted({h[1] for h in c.hits}),
                                      "origins": sorted(c.origins), "family": c.family, "seed_aff": c.seed_aff} for k, c in cands.items()}

        # 6. filters: playlist artists, tribute acts, listener cap (Last.fm + ListenBrainz), genre tags
        before = len(cands)
        cands = {k: c for k, c in cands.items()
                 if k not in playlist_artists and not (set(k.split(" and ")) & playlist_artists)
                 and not (c.mbid and c.mbid in seed_mbids)}
        filtered_in_playlist = before - len(cands)
        cands = {k: c for k, c in cands.items() if not artist_is_tribute(c.name)}
        filtered_mainstream = filtered_genre = 0
        lb_pop_ok = s.listenbrainz is not None
        if lb_pop_ok:
            # ListenBrainz candidates arrive with MBIDs: drop obvious mainstream artists in a few batched
            # POSTs before spending rate-limited Last.fm lookups on them (Today's Top Hits: 222 of 240
            # Last.fm lookups used to hit mainstream artists).
            pre = await self._lb_mainstream([c for c in cands.values() if c.mbid])
            if pre is None:
                lb_pop_ok = False
            else:
                filtered_mainstream += len(pre)
                cands = {k: c for k, c in cands.items() if k not in pre}
        def lookup_order(c: Cand):
            # Spend rate-limited Last.fm lookups where they are likely to pass the cap: artists that
            # ListenBrainz already puts under the cap first, unknown next, borderline (1-2x cap) last.
            if c.lb_users is None:
                group = 1
            else:
                group = 0 if c.lb_users * LB_USER_SCALE <= self.max_listeners else 2
            return group, -candidate_support(c, family_weight)
        family_weight = {norm_name(a): n for a, n in counts.items()}
        # Interleave seed families so one dominant artist's neighbours can't use up the whole lookup budget;
        # the stable sort then restores the ListenBrainz-popularity grouping.
        by_affinity = sorted(interleave_families(sorted(cands.values(), key=lookup_order), family_weight),
                             key=lambda c: lookup_order(c)[0])
        # Popularity mix: among artists ListenBrainz already puts under the cap, alternate predicted-rising and
        # predicted-underground ones, or the lookup budget goes to the best-connected (most popular) artists
        # and the underground tier stays empty (user benchmark playlist: 1 of 120 lookups was 15K-100K).
        g0 = [c for c in by_affinity if lookup_order(c)[0] == 0]
        split = {"rising": [], "underground": []}
        for c in g0:
            split[popularity_tier(c.lb_users * LB_USER_SCALE, self.max_listeners)].append(c)
        a, b = split["rising"], split["underground"]
        by_affinity = [c for i in range(max(len(a), len(b))) for c in a[i:i + 1] + b[i:i + 1]] + by_affinity[len(g0):]
        sem = asyncio.Semaphore(8)
        n_lookups = min(len(by_affinity), MAX_LISTENER_LOOKUPS_HARD)

        async def info(c: Cand):
            async with sem:
                try:
                    listeners, mbid, resolved = await s.lastfm.artist_info(c.name)
                    if resolved and strict_name(resolved) != strict_name(c.name):
                        # Autocorrect can land on a tiny duplicate page ("Florence + The Machine" ->
                        # "Florence   The Machine", 20k listeners). Check the exact name too; keep the
                        # larger count so the listener cap stays conservative.
                        l2, m2, _ = await s.lastfm.artist_info(c.name, autocorrect=0)
                        if (l2 or 0) > (listeners or 0):
                            listeners, mbid = l2, m2 or mbid
                    c.listeners = listeners or 0
                    c.mbid = c.mbid or mbid
                except UpstreamError:
                    c.listeners = 0
                self._tick("Checking listener counts", 0.48, 0.51, min(n_lookups, MAX_LISTENER_LOOKUPS))

        async def tags_of(c: Cand, total: int):
            async with sem:
                try:
                    c.tags = tag_vector(await s.lastfm.artist_top_tags(c.name))
                except UpstreamError:
                    c.tags = {}
                self._tick("Comparing genre tags", 0.51, 0.53, total)

        # 7. candidate tracks (contamination filters, up to TRACKS_PER_ARTIST per artist). Deezer lookups
        # start as soon as an artist passes the listener caps, concurrently with its Last.fm tag lookup
        # (Deezer and Last.fm have separate rate limits); genre-rejected artists' lookups are cancelled.
        tsem = asyncio.Semaphore(8)
        no_preview = contaminated = 0

        deezer_big = 0

        async def tracks_for(c: Cand) -> list[DeezerTrack]:
            nonlocal no_preview, contaminated, deezer_big
            async with tsem:
                found: list[DeezerTrack] = []
                try:
                    aid, fans = await s.deezer.find_artist(c.name)
                    if fans and fans > DEEZER_FAN_SLACK * self.max_listeners:
                        # Last.fm undercounts some scenes (Latin, Disney, ...): Deezer fans as a third cap
                        deezer_big += 1
                        self._tick("Looking up previews", 0.53, 0.55, len(dz_tasks))
                        return []
                    for _, title in sorted(c.tracks, key=lambda x: -x[0])[:2]:
                        t = await s.deezer.find_track(c.name, title)
                        if t:
                            found.append(t)
                    if aid:
                        found += await s.deezer.artist_top(aid, 10, c.name)
                except UpstreamError:
                    pass
                # Deezer returns features and loose search hits; the listener cap was checked for c.name only.
                out, titles, ids = [], set(), set()
                for t in found:
                    nt = norm_title(clean_title(t.title))
                    if (t.id in ids or nt in titles or norm_name(t.artist) != norm_name(c.name)
                            or norm_name(t.artist) in playlist_artists or norm_title(t.title) in playlist_titles):
                        continue
                    if is_contaminated(t.title, t.album, t.artist, allowed):
                        contaminated += 1
                        continue
                    ids.add(t.id)
                    titles.add(nt)
                    out.append(t)
                if not out:
                    no_preview += 1
                self._tick("Looking up previews", 0.53, 0.55, len(dz_tasks))
                return out[:TRACKS_PER_ARTIST]

        dz_tasks: dict[int, asyncio.Task] = {}
        wanted_artists = max(30, 2 * self.limit)
        ranked: list[Cand] = []
        keep: list[Cand] = []
        async def process(batch: list[Cand]) -> None:
            """Listener counts -> caps -> (Deezer lookups started) -> genre tags -> keep."""
            nonlocal filtered_mainstream, filtered_genre, lb_pop_ok
            await asyncio.gather(*(info(c) for c in batch))
            ranked.extend(batch)
            filtered_mainstream += sum(1 for c in batch if c.listeners > self.max_listeners)
            ok = [c for c in batch if floor <= c.listeners <= self.max_listeners]
            if lb_pop_ok and ok:  # second mainstream signal (MBIDs learned from Last.fm getInfo)
                big = await self._lb_mainstream([c for c in ok if c.mbid])
                if big is None:
                    lb_pop_ok = False
                else:
                    filtered_mainstream += len(big)
                    ok = [c for c in ok if norm_name(c.name) not in big]
            for c in ok:
                dz_tasks[id(c)] = asyncio.create_task(tracks_for(c))
            todo = [c for c in ok if c.tags is None]
            tags_total = self._ticks.get("Comparing genre tags", 0) + len(todo)
            await asyncio.gather(*(tags_of(c, tags_total) for c in todo))
            for c in ok:
                if c.tags:
                    c.tag_sim = max([similarity(c.tags, p) for p in profiles["all"]] or [0.0])
                    if profiles["all"] and c.tag_sim < TAG_MIN_SIM:
                        filtered_genre += 1
                        dz_tasks.pop(id(c)).cancel()
                        continue
                elif profiles["all"] and c.affinity < UNTAGGED_MIN_AFFINITY and profiles["untagged_share"] < 0.3:
                    filtered_genre += 1
                    dz_tasks.pop(id(c)).cancel()
                    continue
                keep.append(c)

        def n_under() -> int:
            return sum(1 for c in keep if popularity_tier(c.listeners, cap) == "underground")

        # Popular seeds' neighbours are mostly popular too. When the first batch shows the underground tier
        # will be thin, a second hop runs alongside the remaining Last.fm lookups: ListenBrainz neighbours of
        # the artists kept so far (separate rate limit), pre-filtered to the underground band with one batched
        # ListenBrainz popularity call. Their Last.fm checks come out of the normal lookup budget, so the hop
        # costs no extra Last.fm time.
        hop_task: asyncio.Task | None = None
        budget = MAX_LISTENER_LOOKUPS
        start = 0
        try:
            while start < n_lookups:
                size = min(LISTENER_BATCH, max(1, budget - len(ranked))) if len(ranked) < budget else LISTENER_BATCH
                await process(by_affinity[start:start + size])
                start += size
                if (hop_task is None and s.listenbrainz is not None
                        and n_under() < (wanted_artists // 3) * len(ranked) / max(budget, 1)):
                    hop_task = asyncio.create_task(
                        self._underground_hop(list(keep), cands, playlist_artists, seed_mbids, cap))
                    budget = MAX_LISTENER_LOOKUPS - UNDERGROUND_HOP_LOOKUPS
                # Popularity mix: keep looking (within the normal budget) until the underground tier is stocked too.
                if len(keep) >= wanted_artists and (n_under() >= wanted_artists // 3 or len(ranked) >= budget):
                    break
                # Normal budget is MAX_LISTENER_LOOKUPS; keep going (up to the hard cap) only while we
                # still could not fill the requested number of results.
                # (A running hop is expected to keep about half of its pre-filtered picks.)
                hop_keep = UNDERGROUND_HOP_LOOKUPS // 2 if hop_task is not None else 0
                if len(ranked) >= budget and len(keep) + hop_keep >= self.limit + 5:
                    break
            if hop_task is not None:
                extra = await hop_task
                if extra:
                    await process(extra)
        finally:
            if hop_task is not None and not hop_task.done():
                hop_task.cancel()
        self.rep.progress(0.53, "Checked listener counts and genre tags")
        if self.trace is not None:
            self.trace["looked_up"] = {norm_name(c.name): c.listeners for c in ranked}
            self.trace["keep"] = [norm_name(c.name) for c in keep]
        log.info("candidates: considered=%d looked_up=%d kept=%d mainstream=%d off_genre=%d",
                 considered, len(ranked), len(keep), filtered_mainstream, filtered_genre)
        if not keep:
            for t in dz_tasks.values():
                t.cancel()
            raise PipelineError("no_candidates", "No underground candidates found. Try raising the listener cap.")
        keep.sort(key=lambda c: -(c.affinity * (0.5 + (c.tag_sim if c.tag_sim is not None else 0.2))))
        keep = interleave_tiers(interleave_families(keep, family_weight), cap)

        try:
            per = await asyncio.gather(*(dz_tasks[id(c)] for c in keep))
        finally:
            for t in dz_tasks.values():
                t.cancel()
        filtered_mainstream += deezer_big
        # Round-robin: every artist's best-known track first, then 2nd and 3rd tracks while budget allows.
        budget = MAX_CANDIDATE_CLIPS
        pool: list[tuple[Cand, DeezerTrack]] = []
        seen_ids: set[int] = set()
        for rnd in range(CLIPS_PER_ARTIST):
            for c, ts in zip(keep, per):
                if rnd < len(ts) and ts[rnd].id not in seen_ids and len(pool) < budget:
                    seen_ids.add(ts[rnd].id)
                    pool.append((c, ts[rnd]))
        log.info("candidate clips=%d (artists=%d, contaminated tracks dropped=%d)", len(pool), len(keep), contaminated)
        if not pool:
            raise PipelineError("no_candidates", "No candidate tracks with previews were found.")

        # 8. analyze candidates (+ deep cuts, analyzed and cached like any other clip)
        marks["candidates"] = time.monotonic() - t_start
        dc_pool = [(c, t) for c, ts in await dc_task for t in ts if t.id not in seen_ids]
        clips = pool + dc_pool
        craws = await self._gather_progress(
            [self._features(f"dz:{t.id}:v{FEATURE_VERSION}", t.preview_url) for _, t in clips],
            "analyzing_candidates", 0.55, 0.95, "Analyzing candidate")
        good = [(c, t, r) for (c, t), r in zip(pool, craws) if r]
        dc_good = [(c, t, r) for (c, t), r in zip(dc_pool, craws[len(pool):]) if r]
        if not good:
            raise PipelineError("no_candidates", "None of the candidate previews could be analyzed.")

        # 9-11. rank: blended audio + genre-tag score; keep each artist's best-matching tracks
        marks["candidate_audio"] = time.monotonic() - t_start
        rep.stage("ranking", 0.95, "Ranking matches")
        scored: list[Scored] = []
        info_by_key: dict[int, tuple] = {}
        off_mood = below_floor = 0
        for c, t, r in good:
            n = normalize(r, scaler)
            vid, dc = candidate_distance(n, norms, vibes)
            audio = match_score(dc, d_ref)
            tsim, fit = self._tag_fit(c, vid, profiles)
            score = int(round(w_audio * audio + w_tags * 100 * fit))
            env = envelope_ratio(n, vibes, radii)
            tier = popularity_tier(c.listeners, cap)
            reject = "mood" if env is not None and env > 1.0 else "floor" if score < floor_score else None
            self._trace_scored(c, t, r, audio, tsim, fit, score, vid, deep=False, n=n, env=env, tier=tier,
                               reject=reject)
            if reject == "mood":  # outside the playlist's mood envelope: an outlier, whatever its genre
                off_mood += 1
                continue
            if reject == "floor":
                below_floor += 1
                continue
            sc = Scored(key=t.id, artist=t.artist or c.name, vibe_id=vid, score=score, affinity=c.affinity,
                        family=c.family, tier=tier)
            scored.append(sc)
            info_by_key[t.id] = (c, t, r, n, audio, tsim)
        log.info("d_ref=%.3f audio scores: %s; rejected off-mood=%d below-floor=%d", d_ref,
                 sorted((info_by_key[x.key][4] for x in scored), reverse=True)[:40], off_mood, below_floor)
        # Best-of-N per artist: keep each artist's best-matching analyzed track (not its most popular).
        # Second tracks only come back if there are too few distinct artists to fill the list.
        best: dict[str, Scored] = {}
        for sc in sorted(scored, key=lambda x: -x.rank_score):
            best.setdefault(sc.artist.lower(), sc)
        if len(best) >= self.limit:
            scored = list(best.values())
        deep: list[Scored] = []
        for c, t, r in dc_good:
            c.tags = self._artist_tags.get(c.name)  # playlist tag profile lookups (seed artists come first)
            n = normalize(r, scaler)
            vid, dc = candidate_distance(n, norms, vibes)
            audio = match_score(dc, d_ref)
            tsim, fit = self._tag_fit(c, vid, profiles)
            score = int(round(w_audio * audio + w_tags * 100 * fit))
            env = envelope_ratio(n, vibes, radii)
            log.info("deep cut candidate: %s - %s score=%d audio=%d env=%s deezer_rank=%s artist_best_rank=%s lfm_ratio=%s",
                     c.name, t.title, score, audio, env and round(env, 2), t.rank, c.best_rank, c.track_ratio.get(t.id))
            reject = ("mood" if env is not None and env > 1.0
                      else "floor" if score < max(DEEP_CUT_MIN_SCORE, floor_score) else None)
            self._trace_scored(c, t, r, audio, tsim, fit, score, vid, deep=True, n=n, env=env, tier="deep_cut",
                               reject=reject)
            if reject:
                continue
            deep.append(Scored(key=t.id, artist=t.artist or c.name, vibe_id=vid, score=score,
                               affinity=DEEP_CUT_AFFINITY, family=None, tier="deep_cut"))
            info_by_key[t.id] = (c, t, r, n, audio, tsim)
        n_deep = min(sum(min(DEEP_CUT_PER_ARTIST, k) for k in Counter(d.artist.lower() for d in deep).values()),
                     deep_cut_slots(self.limit)[1])
        total = sum(family_weight.values()) or 1
        n_disc = self.limit - n_deep

        def too_close(a: Scored, b: Scored) -> bool:  # near-duplicate sound: keep only the better pick
            return distance(info_by_key[a.key][3], info_by_key[b.key][3]) < dup_d

        chosen = select_diverse(scored, n_disc, {v.id: v.share for v in vibes},
                                {f: n / total for f, n in family_weight.items()},
                                tier_quota=tier_quotas(n_disc), too_close=too_close)
        chosen = merge_deep_cuts(chosen, deep, self.limit)
        deep_keys = {d.key for d in deep}
        distinct = len({s.artist.lower() for s in chosen})
        if not chosen:
            raise PipelineError("no_candidates", "No candidates matched this playlist's mood closely enough. "
                                "Try raising the listener cap.")
        if len(chosen) < self.limit:
            rep.warn(f"Only {len(chosen)} strong matches for this playlist's mood under this listener cap. "
                     "Slide Underground depth to the right for more.")
        elif distinct < 0.75 * self.limit:
            rep.warn(f"Only {distinct} artists under this listener cap matched your playlist. "
                     "Slide Underground depth to the right for more variety.")
        def to_result(rank: int, sc: Scored) -> Result:
            c, t, r, n, audio, tsim = info_by_key[sc.key]
            v = vibes[sc.vibe_id]
            summ = n.summary()
            why = explain(n, n.bpm_raw, v.centroid, v.bpm)
            if has_mood:
                why = with_mood(why, mood_words(n.mz, n.tz), self._vibe_words.get(v.id, []))
            if sc.key in deep_keys:
                why = deep_cut_why(c.name, why)
            elif c.tags:
                prof = profiles["vibe"].get(v.id) or (profiles["all"][0] if profiles["all"] else {})
                why = with_tags(why, shared_tags(c.tags, prof) or shared_tags(c.tags, profiles["all"][0] if profiles["all"] else {}))
            res = Result(
                rank=rank, deezer_id=t.id, title=t.title, artist=t.artist or c.name, album=t.album,
                artwork_url=t.artwork_url, preview_url=t.preview_url, preview_proxy_url=preview_path(self.s.settings.preview_secret, t.id),
                deezer_url=t.link, spotify_search_url=spotify_search_url(t.artist or c.name, t.title),
                listeners=c.listeners, match_score=max(0, min(100, sc.score)), bpm=summ["bpm"], vibe_id=v.id,
                why=why, source="deep_cut" if sc.key in deep_keys else c.source, features=FeatureSummary(**summ),
                tier=sc.tier)
            log.info("result %d: %s - %s score=%d audio=%d tag_sim=%s", rank, t.artist, t.title, sc.score, audio,
                     None if tsim is None else round(tsim, 3))
            return res

        results = [to_result(rank, sc) for rank, sc in enumerate(chosen, 1)]
        # Reserve pool for client-side refine: next-best scored (already analyzed and filtered) leftovers.
        reserve = [to_result(rank, sc) for rank, sc in enumerate(
            pick_reserve([*scored, *deep], {x.key for x in chosen}, {x.artist.lower() for x in chosen}),
            len(results) + 1)]
        if self.trace is not None:
            self.trace["results"] = [{"artist": x.artist, "title": x.title, "score": x.match_score, "source": x.source,
                                      "listeners": x.listeners, "deezer_id": x.deezer_id, "tier": x.tier,
                                      "why": x.why, "mood_words": mood_words(info_by_key[x.deezer_id][3].mz,
                                                                             info_by_key[x.deezer_id][3].tz)}
                                     for x in results]
        stats = Stats(candidates_considered=considered, candidates_analyzed=len(good),
                      filtered_mainstream=filtered_mainstream, filtered_in_playlist=filtered_in_playlist,
                      no_preview=no_preview)
        log.info("timings (cumulative s): %s, total=%.1f, clips=%d",
                 {k: round(v, 1) for k, v in marks.items()}, time.monotonic() - t_start, len(pool))
        rep.done(results, stats, reserve)

    # -- helpers --------------------------------------------------------
    def _trace_scored(self, c, t, r, audio, tsim, fit, score, vid, deep: bool, n: Norm | None = None,
                      env: float | None = None, tier: str | None = None, reject: str | None = None) -> None:
        if self.trace is None:
            return
        self.trace.setdefault("scored", []).append({
            "env": env, "tier": tier, "reject": reject,
            "mood": None if n is None or n.mood is None else [round(float(x), 4) for x in n.mood],
            "artist": t.artist or c.name, "cand": c.name, "title": t.title, "deezer_id": t.id, "raw": r,
            "audio": audio, "tag_sim": tsim, "fit": fit, "score": score, "vibe": vid, "deep": deep,
            "affinity": c.affinity, "family": c.family, "listeners": c.listeners, "rank": t.rank,
            "lfm_ratio": getattr(c, "track_ratio", {}).get(t.id)})

    async def _deep_cuts(self, artists: list[str], tracks: list[PlaylistTrack], allowed: set[str]
                         ) -> list[tuple["DeepCutArtist", list[DeezerTrack]]]:
        """[(artist, deep-cut tracks)] for the playlist's core artists. Never raises: deep cuts are a bonus."""
        if not artists:
            return []
        s = self.s
        titles = {norm_title(t.title) for t in tracks}
        isrcs = {t.isrc for t in tracks if t.isrc}
        sem = asyncio.Semaphore(4)

        async def one(a: str):
            async with sem:
                try:
                    aid, _ = await s.deezer.find_artist(a)
                    if not aid:
                        return None
                    top = await s.deezer.artist_top(aid, DEEP_CUT_TOP_FETCH, a)
                    try:
                        lfm = LastfmPopularity.build(await s.lastfm.artist_top_tracks(a, DEEP_CUT_LFM_TOP))
                    except UpstreamError:
                        lfm = None
                    picks = pick_deep_cuts(a, top, titles, isrcs, allowed, lfm=lfm)
                    if not picks:
                        return None
                    try:
                        listeners = (await s.lastfm.artist_info(a))[0] or 0
                    except UpstreamError:
                        listeners = 0
                except UpstreamError as e:
                    log.warning("deep cuts for %s failed: %s", a, e)
                    return None
                own = [t.rank for t in top if t.rank and norm_name(t.artist) == norm_name(a)]
                c = DeepCutArtist(a, listeners=listeners, best_rank=max(own) if own else None,
                                  track_ratio={t.id: round(lfm.ratio(t.title), 3) for t in picks} if lfm else {})
                return c, picks

        try:
            res = await asyncio.gather(*(one(a) for a in artists))
        except Exception:  # noqa: BLE001 - never fail a job because of deep cuts
            log.exception("deep cuts failed")
            return []
        out = [r for r in res if r]
        log.info("deep cuts: %s", {c.name: [t.title for t in ts] for c, ts in out})
        return out

    async def _underground_hop(self, keep: list[Cand], cands: dict[str, Cand], playlist_artists: set[str],
                               seed_mbids: set[str], cap: int) -> list[Cand]:
        """New candidates one ListenBrainz hop beyond the kept artists whose ListenBrainz user count predicts
        the underground band (floor..rising_min Last.fm listeners, with slack for the ~10-15x scale), best
        first, at most UNDERGROUND_HOP_LOOKUPS. Never raises: the hop is a bonus."""
        lb = self.s.listenbrainz
        srcs = [c for c in sorted(keep, key=lambda c: -c.affinity) if c.mbid][:UNDERGROUND_HOP_SOURCES]
        if lb is None or not srcs:
            return []

        async def work() -> list[Cand]:
            res = await asyncio.gather(*(lb.similar_artists(c.mbid) for c in srcs), return_exceptions=True)
            new: dict[str, Cand] = {}
            for src, rows in zip(srcs, res):
                if isinstance(rows, BaseException) or not rows or rows[0][2] <= 0:
                    continue
                top = rows[0][2]
                for amb, name, score in rows[:LB_ROWS_PER_SEED]:
                    k = norm_name(name or "")
                    if (not k or k in cands or k in playlist_artists or amb in seed_mbids
                            or artist_is_tribute(name) or NON_ARTIST.search(name)):
                        continue
                    aff = 0.9 * math.sqrt(max(0.0, score) / top) * src.affinity  # a hop further: weaker evidence
                    c = new.setdefault(k, Cand(name, mbid=amb))
                    c.hits.append((aff, "similar_artist"))
                    c.origins.add("listenbrainz")
                    if src.family:
                        c.seeds.add(src.family)
                        c.seed_aff[src.family] = max(c.seed_aff.get(src.family, 0.0), aff)
            if not new:
                return []
            pop = await lb.popularity([c.mbid for c in new.values()])
            lo = 0.5 * min_listeners(cap) / LB_USER_SCALE
            hi = 1.5 * rising_min(cap) / LB_USER_SCALE
            picks = [c for c in new.values() if c.mbid in pop and lo <= pop[c.mbid][0] <= hi]
            for c in picks:
                c.lb_users = pop[c.mbid][0]
                cands[norm_name(c.name)] = c
            picks.sort(key=lambda c: -c.affinity)
            log.info("underground hop: %d sources, %d new neighbours, %d in the band", len(srcs), len(new), len(picks))
            return picks[:UNDERGROUND_HOP_LOOKUPS]

        try:
            return await asyncio.wait_for(work(), UNDERGROUND_HOP_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - never fail a job because of the hop
            log.warning("underground hop failed: %s", e)
            return []

    def _tick(self, label: str, lo: float, hi: float, total: int) -> None:
        """Advance progress smoothly inside [lo, hi] with an 'x/total' counter (monotonic via the store)."""
        done = self._ticks.get(label, 0) + 1
        self._ticks[label] = done
        total = max(total, done, 1)
        self.rep.progress(lo + (hi - lo) * done / total, f"{label} {done}/{total}")

    async def _lb_mainstream(self, cs: list[Cand]) -> set[str] | None:
        """norm names of candidates whose ListenBrainz user count marks them as mainstream;
        None if ListenBrainz failed (callers then skip the check)."""
        if not cs:
            return set()
        try:
            pop = await asyncio.wait_for(self.s.listenbrainz.popularity([c.mbid for c in cs]), 20)
        except Exception as e:  # noqa: BLE001 - never fail a job because of ListenBrainz
            log.warning("ListenBrainz popularity failed: %s", e)
            return None
        for c in cs:
            if c.mbid in pop:
                c.lb_users = pop[c.mbid][0]
        cap = self.max_listeners * LB_MAINSTREAM_SLACK
        return {norm_name(c.name) for c in cs if c.lb_users is not None and c.lb_users * LB_USER_SCALE > cap}

    def _tag_fit(self, c: Cand, vid: int, profiles: dict) -> tuple[float | None, float]:
        """(tag similarity used for ranking, 0..1 genre fit). Similarity blends the whole-playlist profile
        (70%) with the candidate's nearest audio vibe profile (30%): a niche corner of a mixed playlist
        (two Finnish rap tracks in Today's Top Hits) may pass the hard filter but should rank lower."""
        if not profiles["all"]:
            return None, 0.5  # no playlist tags at all: neutral, audio decides
        if not c.tags:  # small artists are often untagged; neutral when the playlist itself is mostly untagged
            return None, UNTAGGED_TAG_SCORE if profiles["untagged_share"] < 0.3 else 0.5
        whole = similarity(c.tags, profiles["all"][0])
        vp = profiles["vibe"].get(vid)
        tsim = 0.7 * whole + 0.3 * similarity(c.tags, vp) if vp else whole
        fit = (tsim - TAG_MIN_SIM) / (TAG_FULL_SIM - TAG_MIN_SIM)
        return tsim, float(min(1.0, max(0.0, fit)))

    async def _tag_profiles(self, counts: Counter, seeds: list[str], pl, vibes) -> dict:
        """Last.fm tag profile of the whole playlist and of each audio vibe.
        profiles["all"] = [whole, *vibe profiles] (hard filter uses the max over these)."""
        lf = self.s.lastfm
        artists = list(dict.fromkeys(seeds + [t.primary_artist for t, _ in pl
                                              if t.primary_artist in counts]))[:SEED_TAG_ARTISTS]
        sem = asyncio.Semaphore(8)

        async def get(a):
            async with sem:
                try:
                    v = tag_vector(await lf.artist_top_tags(a))
                except UpstreamError:
                    v = {}
                self._tick("Reading playlist genre tags", 0.45, 0.46, len(artists))
                return a, v

        artist_tags = {a: v for a, v in await asyncio.gather(*(get(a) for a in artists)) if v}
        self._artist_tags = artist_tags
        whole = build_profile(artist_tags, counts)
        per_vibe: dict[int, dict] = {}
        if len(vibes) > 1:
            for v in vibes:
                vc = Counter(pl[j][0].primary_artist for j in v.members)
                at = {a: artist_tags[a] for a in vc if a in artist_tags}
                if len(at) >= 2:
                    per_vibe[v.id] = build_profile(at, vc)
        log.info("playlist tag profile: %s (tagged artists %d/%d)", top_profile_tags(whole, 8), len(artist_tags), len(artists))
        untagged = 1 - len(artist_tags) / max(1, len(artists))
        return {"all": ([whole] + list(per_vibe.values())) if whole else [], "vibe": per_vibe, "whole": whole,
                "untagged_share": untagged}

    def _profile(self, pl, norms: list[Norm], vibes) -> PlaylistProfile:
        folded = [n.bpm_folded for n in norms]
        med = float(statistics.median(folded))
        lo, hi = float(np.percentile(folded, 10)), float(np.percentile(folded, 90))
        overall = centroid_of(norms)
        labels = vibe_labels([v.centroid for v in vibes], [v.bpm for v in vibes], overall if len(vibes) > 1 else None)
        self._vibe_words: dict[int, list[str]] = {}
        if all(n.mz is not None for n in norms):  # mood words: "late-night · mellow · sensual"
            ref = word_scores(overall.mz, overall.tz) if len(vibes) > 1 else None
            used: set[str] = set()
            for v in vibes:
                words = mood_words(v.centroid.mz, v.centroid.tz, ref=ref, avoid=used)
                used.update(words)
                self._vibe_words[v.id] = words
                labels[v.id] = " · ".join(words) or labels[v.id]
            if len(set(labels)) < len(labels):  # identical mood words: tell siblings apart by tempo
                labels = [f"{lab} · {round(v.bpm)} BPM" for lab, v in zip(labels, vibes)]
        out = []
        for v in vibes:
            cen = v.centroid
            ordered = sorted(v.members, key=lambda j: distance(norms[j], cen))[:3]
            out.append(VibeOut(
                id=v.id, label=labels[v.id], share=round(v.share, 3), bpm=round(v.bpm, 1),
                track_count=len(v.members), features=FeatureSummary(**cen.summary(bpm=v.bpm)),
                example_tracks=[TrackRef(title=pl[j][0].title, artist=pl[j][0].primary_artist) for j in ordered]))
        return PlaylistProfile(bpm_median=round(med, 1), bpm_range=(round(lo, 1), round(hi, 1)),
                               features=FeatureSummary(**overall.summary(bpm=med)), vibes=out)

    async def _candidates(self, tracks, pl, norms, vibes, seed_artists: list[str], profiles: dict) -> tuple[dict[str, Cand], set[str]]:
        """Last.fm similar artists/tracks + tag top artists, merged with ListenBrainz similar artists."""
        lf = self.s.lastfm
        seed_tracks: list[tuple[str, str]] = []
        seen = set()
        for v in vibes:
            q = max(1, round(SEED_TRACKS * v.share))
            for j in sorted(v.members, key=lambda j: distance(norms[j], v.centroid))[:q]:
                t = pl[j][0]
                k = (norm_name(t.primary_artist), norm_title(t.title))
                if k not in seen:
                    seen.add(k)
                    seed_tracks.append((t.primary_artist, clean_title(t.title)))
        seed_tracks = seed_tracks[:SEED_TRACKS]
        title_for = {}
        for t in tracks:
            title_for.setdefault(t.primary_artist, clean_title(t.title))

        cands: dict[str, Cand] = {}

        def add(name: str, aff: float, source: str, title: str | None = None, origin: str = "lastfm",
                mbid: str | None = None, seed: str | None = None):
            if not name:
                return
            c = cands.setdefault(norm_name(name), Cand(name))
            c.hits.append((aff, source))
            if seed:
                k = norm_name(seed)
                c.seeds.add(k)
                c.seed_aff[k] = max(c.seed_aff.get(k, 0.0), aff)
            c.origins.add(origin)
            if mbid and not c.mbid:
                c.mbid = mbid
            if title:
                c.tracks.append((aff, title))

        failures = 0
        sem = asyncio.Semaphore(8)
        top_tags = top_profile_tags(profiles.get("whole") or {}, 3)
        n_calls = len(seed_artists) + len(seed_tracks) + len(top_tags)

        async def guard(coro):
            nonlocal failures
            async with sem:
                try:
                    return await coro
                except UpstreamError as e:
                    failures += 1
                    self._last_err = str(e)
                    return None
                finally:
                    self._tick("Finding similar artists", 0.46, 0.48, n_calls)

        lb_task = asyncio.create_task(self._listenbrainz_candidates(seed_artists, title_for))
        sim_a, sim_t, tag_pages = await asyncio.gather(
            asyncio.gather(*(guard(lf.artist_similar(a, 50, with_mbid=True)) for a in seed_artists)),
            asyncio.gather(*(guard(lf.track_similar(a, t, 30, with_mbid=True)) for a, t in seed_tracks)),
            asyncio.gather(*(guard(lf.tag_top_artists(t, 50, 2)) for t in top_tags)))
        for seed, res in zip(seed_artists, sim_a):
            for name, m, mb in res or []:  # Last.fm MBIDs let the ListenBrainz popularity prefilter
                add(name, m, "similar_artist", seed=seed, mbid=mb)  # skip mainstream artists cheaply
        for (seed, _), res in zip(seed_tracks, sim_t):
            for artist, title, m, mb in res or []:
                add(artist, m, "similar_track", title, seed=seed, mbid=mb)
        for res in tag_pages:
            for name in res or []:
                add(name, 0.3, "tag")
        lb_rows, seed_mbids = await lb_task
        for name, aff, mbid, ref in lb_rows:
            add(name, aff, "similar_artist", origin="listenbrainz", mbid=mbid, seed="lb:" + ref)
        if not cands:
            if failures:
                raise PipelineError("upstream_error", "Last.fm is not responding. Please try again in a minute.",
                                    getattr(self, "_last_err", ""))
            raise PipelineError("no_candidates", "Last.fm returned no similar artists for this playlist.")
        if failures:
            self.rep.warn("Some Last.fm lookups failed, so the candidate pool may be smaller than usual.")
        return cands, seed_mbids

    async def _listenbrainz_candidates(self, seed_artists: list[str], title_for: dict[str, str]):
        """[(name, affinity 0..0.9, mbid)] from ListenBrainz co-listening, plus the seed MBIDs.
        Never raises: any failure degrades to Last.fm-only with a warning."""
        lb = self.s.listenbrainz
        if lb is None:
            return [], set()

        async def work():
            sem = asyncio.Semaphore(4)

            async def mbid_of(a):
                async with sem:
                    try:
                        m = (await self.s.lastfm.artist_info(a))[1]
                    except UpstreamError:
                        m = None
                    if not m:
                        try:
                            m = await lb.lookup_artist_mbid(a, title_for.get(a, ""))
                        except UpstreamError:
                            m = None
                    return m

            mbids = [m for m in await asyncio.gather(*(mbid_of(a) for a in seed_artists[:LB_SEEDS])) if m]
            errors = 0

            async def sim(m):
                nonlocal errors
                async with sem:
                    try:
                        return await lb.similar_artists(m)
                    except UpstreamError as e:
                        errors += 1
                        log.warning("ListenBrainz similar-artists failed for %s: %s", m, e)
                        return []

            res = await asyncio.gather(*(sim(m) for m in dict.fromkeys(mbids)))
            if mbids and errors == len(set(mbids)):
                raise UpstreamError("all ListenBrainz similar-artist calls failed")
            rows = []
            for ref, r in zip(dict.fromkeys(mbids), res):
                r = r[:LB_ROWS_PER_SEED]
                if not r or r[0][2] <= 0:
                    continue
                top = r[0][2]
                for amb, name, score in r:  # per-reference normalization; sqrt keeps the long tail usable
                    rows.append((name, 0.9 * math.sqrt(max(0.0, score) / top), amb, ref))
            return rows, set(mbids)

        try:
            return await asyncio.wait_for(work(), LB_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - ListenBrainz must never fail a job
            log.warning("ListenBrainz unavailable: %s", e)
            self.rep.warn("ListenBrainz was unavailable, so recommendations use Last.fm data only.")
            return [], set()


async def run_analysis(svc: Services, playlist_id: str | None, max_listeners: int, limit: int, rep: Reporter,
                       tracks: list[dict] | None = None) -> None:
    p = Pipeline(svc, playlist_id or "", max_listeners, limit, rep)
    p.pasted = tracks
    await p.run()
