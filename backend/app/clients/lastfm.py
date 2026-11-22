"""Last.fm client (cached in SQLite)."""
from __future__ import annotations

import json
import logging

from app.cache import Cache
from app.clients.http import HttpClient, UpstreamError

log = logging.getLogger("deepcut.lastfm")
API = "https://ws.audioscrobbler.com/2.0/"
TTL = 7 * 86400


def _listify(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


class LastfmClient:
    def __init__(self, http: HttpClient, cache: Cache, api_key: str):
        self.http, self.cache, self.api_key = http, cache, api_key

    async def _call(self, method: str, **params) -> dict:
        key = "lfm:" + method + ":" + json.dumps(params, sort_keys=True)
        hit = await self.cache.get_http(key)
        if hit is not None:
            return json.loads(hit)
        q = {"method": method, "api_key": self.api_key, "format": "json", **params}
        resp = await self.http.get(API, params=q)
        try:
            data = resp.json()
        except ValueError:
            raise UpstreamError(f"Last.fm {method}: bad response")
        if isinstance(data, dict) and "error" in data:
            code = data.get("error")
            if code == 6:  # not found: stable answer, cache as empty
                data = {}
            elif code in (10, 26):  # invalid / suspended key
                raise UpstreamError(f"Last.fm API key rejected: {data.get('message')}")
            else:  # 8 operation failed, 11/16/29 temporary, anything else: never cache
                if code not in (8, 11, 16, 29):
                    log.warning("Last.fm %s returned error %s: %s", method, code, data.get("message"))
                raise UpstreamError(f"Last.fm {method}: error {code}: {data.get('message')}")
        elif resp.status_code != 200:
            raise UpstreamError(f"Last.fm {method}: HTTP {resp.status_code}")
        await self.cache.put_http(key, json.dumps(data), TTL)
        return data

    async def artist_similar(self, artist: str, limit: int = 50) -> list[tuple[str, float]]:
        d = await self._call("artist.getSimilar", artist=artist, limit=limit, autocorrect=1)
        out = []
        for a in _listify((d.get("similarartists") or {}).get("artist")):
            try:
                out.append((a["name"], float(a.get("match", 0))))
            except (KeyError, ValueError, TypeError):
                pass
        return out

    async def track_similar(self, artist: str, track: str, limit: int = 30) -> list[tuple[str, str, float]]:
        d = await self._call("track.getSimilar", artist=artist, track=track, limit=limit, autocorrect=1)
        out = []
        for t in _listify((d.get("similartracks") or {}).get("track")):
            try:
                out.append((t["artist"]["name"], t["name"], float(t.get("match", 0))))
            except (KeyError, ValueError, TypeError):
                pass
        return out

    async def artist_top_tags(self, artist: str) -> list[tuple[str, int]]:
        d = await self._call("artist.getTopTags", artist=artist, autocorrect=1)
        out = []
        for t in _listify((d.get("toptags") or {}).get("tag")):
            try:
                out.append((t["name"], int(t.get("count", 0))))
            except (KeyError, ValueError, TypeError):
                pass
        return out

    async def tag_top_artists(self, tag: str, limit: int = 50, page: int = 1) -> list[str]:
        d = await self._call("tag.getTopArtists", tag=tag, limit=limit, page=page)
        return [a["name"] for a in _listify((d.get("topartists") or {}).get("artist")) if a.get("name")]

    async def artist_listeners(self, artist: str) -> int | None:
        d = await self._call("artist.getInfo", artist=artist, autocorrect=1)
        stats = (d.get("artist") or {}).get("stats") or {}
        try:
            return int(stats["listeners"])
        except (KeyError, ValueError, TypeError):
            return None
