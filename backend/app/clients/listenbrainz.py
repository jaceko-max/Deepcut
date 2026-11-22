"""ListenBrainz client (no API key): artist MBID lookup, similar artists, artist popularity.

Everything is cached in http_cache for 7 days. The labs similar-artists `algorithm` value is
undocumented and changes from time to time: on HTTP 400 the error body lists the permitted values,
so we parse it, pick the closest one to our preference and remember it (cache + instance).
Callers must treat every UpstreamError as "ListenBrainz unavailable" and degrade, never fail a job.
"""
from __future__ import annotations

import difflib
import html
import json
import logging
import re

from app.cache import Cache
from app.clients.http import HttpClient, UpstreamError

log = logging.getLogger("deepcut.listenbrainz")

API = "https://api.listenbrainz.org/1"
LABS = "https://labs.api.listenbrainz.org"
TTL = 7 * 86400
PREFERRED_ALGOS = (
    "session_based_days_7500_session_300_contribution_5_threshold_10_limit_100_filter_True_skip_30",
    "session_based_days_9000_session_300_contribution_5_threshold_15_limit_50_skip_30",
)
ALGO_CACHE_KEY = "lb:similar_artists_algorithm"
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def parse_permitted_algorithms(body: str) -> list[str]:
    """Extract permitted algorithm names from the labs API's 400 validation message."""
    text = html.unescape(body or "")
    m = re.search(r"permitted:(.*?)(?:\(type=|$)", text, re.S)
    seg = m.group(1) if m else text
    out: list[str] = []
    for name in re.findall(r"'([A-Za-z0-9_]+)'", seg):
        if name not in out:
            out.append(name)
    return out


def _algo_numbers(name: str) -> dict[str, float]:
    return {k: float(v) for k, v in re.findall(r"([a-z]+)_(\d+)", name)}


def closest_algorithm(permitted: list[str], preferred: str = PREFERRED_ALGOS[0]) -> str | None:
    """Exact match first; otherwise the candidate whose numeric parameters (days, contribution, ...)
    are closest to the preferred one, tie-broken by string similarity."""
    if not permitted:
        return None
    if preferred in permitted:
        return preferred
    want = _algo_numbers(preferred)

    def cost(name: str) -> tuple[float, float]:
        got = _algo_numbers(name)
        c = 0.0
        for k, v in want.items():  # the history window ("days") matters most for similarity quality
            g = got.get(k)
            c += (3.0 if k == "days" else 1.0) * (1.0 if g is None else abs(g - v) / max(v, g, 1.0))
        c += 0.25 * (("filter_True" in name) != ("filter_True" in preferred))
        return c, -difflib.SequenceMatcher(None, name, preferred).ratio()

    return min(permitted, key=cost)


class ListenBrainzClient:
    def __init__(self, http: HttpClient, cache: Cache):
        self.http, self.cache = http, cache
        self.algorithm: str | None = None
        self.lookup_disabled = False  # metadata/lookup started requiring auth (HTTP 401) in 2026

    async def _cached(self, key: str):
        hit = await self.cache.get_http(key)
        return json.loads(hit) if hit is not None else None

    async def _get_algorithm(self) -> str:
        if self.algorithm:
            return self.algorithm
        hit = await self._cached(ALGO_CACHE_KEY)
        self.algorithm = hit if isinstance(hit, str) else PREFERRED_ALGOS[0]
        return self.algorithm

    async def _set_algorithm(self, algo: str) -> None:
        self.algorithm = algo
        await self.cache.put_http(ALGO_CACHE_KEY, json.dumps(algo), TTL)

    async def lookup_artist_mbid(self, artist: str, recording: str) -> str | None:
        """MBID via metadata/lookup (needs a recording name). Returns None when unavailable."""
        if self.lookup_disabled or not artist or not recording:
            return None
        key = "lb:lookup:" + json.dumps([artist.lower(), recording.lower()])
        hit = await self._cached(key)
        if hit is not None:
            return hit.get("mbid")
        resp = await self.http.get(API + "/metadata/lookup/", params={"artist_name": artist, "recording_name": recording})
        if resp.status_code in (401, 403):
            self.lookup_disabled = True
            return None
        if resp.status_code != 200:
            raise UpstreamError(f"ListenBrainz lookup: HTTP {resp.status_code}")
        try:
            data = resp.json()
        except ValueError:
            raise UpstreamError("ListenBrainz lookup: bad response")
        mbids = (data.get("artist_mbids") or []) if isinstance(data, dict) else []
        mbid = mbids[0] if mbids and isinstance(mbids[0], str) else None
        await self.cache.put_http(key, json.dumps({"mbid": mbid}), TTL)
        return mbid

    async def similar_artists(self, mbid: str) -> list[tuple[str, str, float]]:
        """[(artist_mbid, name, raw co-listen score)] for one reference artist, best first."""
        if not UUID_RE.match(mbid or ""):
            return []
        algo = await self._get_algorithm()
        key = f"lb:similar:{mbid}:{algo}"
        hit = await self._cached(key)
        if hit is None:
            url = LABS + "/similar-artists/json"
            resp = await self.http.get(url, params={"artist_mbids": mbid, "algorithm": algo})
            if resp.status_code == 400:
                permitted = parse_permitted_algorithms(resp.text)
                new = closest_algorithm(permitted, algo)
                if not new or new == algo:
                    raise UpstreamError(f"ListenBrainz similar-artists: HTTP 400 ({resp.text[:120]!r})")
                log.warning("ListenBrainz algorithm %s rejected; switching to %s", algo, new)
                await self._set_algorithm(new)
                algo = new
                key = f"lb:similar:{mbid}:{algo}"
                resp = await self.http.get(url, params={"artist_mbids": mbid, "algorithm": algo})
            if resp.status_code != 200:
                raise UpstreamError(f"ListenBrainz similar-artists: HTTP {resp.status_code}")
            try:
                hit = resp.json()
            except ValueError:
                raise UpstreamError("ListenBrainz similar-artists: bad response")
            if not isinstance(hit, list):
                hit = []
            await self.cache.put_http(key, json.dumps(hit), TTL)
        out = []
        for row in hit:
            try:
                if row.get("reference_mbid") not in (None, mbid) or row.get("artist_mbid") == mbid:
                    continue
                out.append((row["artist_mbid"], row["name"], float(row.get("score") or 0)))
            except (KeyError, TypeError, ValueError, AttributeError):
                pass
        out.sort(key=lambda r: -r[2])
        return out

    async def popularity(self, mbids: list[str]) -> dict[str, tuple[int, int]]:
        """{mbid: (total_user_count, total_listen_count)}; batches of 100, cached per MBID."""
        out: dict[str, tuple[int, int]] = {}
        todo = []
        for m in dict.fromkeys(x for x in mbids if UUID_RE.match(x or "")):
            hit = await self._cached("lb:pop:" + m)
            if hit is None:
                todo.append(m)
            elif hit:
                out[m] = (int(hit[0]), int(hit[1]))
        for i in range(0, len(todo), 100):
            chunk = todo[i:i + 100]
            resp = await self.http.request("POST", API + "/popularity/artist", json={"artist_mbids": chunk})
            if resp.status_code != 200:
                raise UpstreamError(f"ListenBrainz popularity: HTTP {resp.status_code}")
            try:
                rows = resp.json()
            except ValueError:
                raise UpstreamError("ListenBrainz popularity: bad response")
            got = {}
            for row in rows if isinstance(rows, list) else []:
                try:
                    got[row["artist_mbid"]] = (int(row.get("total_user_count") or 0), int(row.get("total_listen_count") or 0))
                except (KeyError, TypeError, ValueError, AttributeError):
                    pass
            for m in chunk:
                val = got.get(m)
                await self.cache.put_http("lb:pop:" + m, json.dumps(list(val) if val else []), TTL)
                if val:
                    out[m] = val
        return out
