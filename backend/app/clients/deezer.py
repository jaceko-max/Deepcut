"""Deezer client. Quirks: field-filter search is broken (keyword search + verify),
errors arrive as HTTP 200 JSON, preview URLs are signed and short-lived."""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass

import httpx

from app.cache import Cache
from app.clients.http import HttpClient, UpstreamError
from app.textutil import norm_name, norm_title

API = "https://api.deezer.com"


@dataclass
class DeezerTrack:
    id: int
    title: str
    artist: str
    artist_id: int | None
    album: str | None
    artwork_url: str | None
    preview_url: str
    link: str
    duration: int | None = None


def deezer_body_rate_limited(resp: httpx.Response) -> bool:
    try:
        d = resp.json()
    except ValueError:
        return False
    return isinstance(d, dict) and isinstance(d.get("error"), dict) and d["error"].get("code") == 4


def parse_track(d: dict, fallback_artist: str | None = None) -> DeezerTrack | None:
    if not isinstance(d, dict) or d.get("error") or d.get("readable") is False:
        return None
    preview = d.get("preview") or ""
    if not preview or not d.get("id"):
        return None
    artist = d.get("artist") or {}
    album = d.get("album") or {}
    return DeezerTrack(
        id=int(d["id"]), title=d.get("title") or d.get("title_short") or "",
        artist=artist.get("name") or fallback_artist or "", artist_id=artist.get("id"),
        album=album.get("title") or None,
        artwork_url=album.get("cover_medium") or album.get("cover") or None,
        preview_url=preview, link=d.get("link") or f"https://www.deezer.com/track/{d['id']}",
        duration=d.get("duration"),
    )


class DeezerClient:
    def __init__(self, http: HttpClient, cache: Cache):
        self.http, self.cache = http, cache

    async def _json(self, path: str, params: dict | None, ttl: float | None) -> dict:
        key = "dz:" + path + ":" + json.dumps(params or {}, sort_keys=True)
        if ttl:
            hit = await self.cache.get_http(key)
            if hit is not None:
                return json.loads(hit)
        resp = await self.http.get(API + path, params=params)
        try:
            data = resp.json()
        except ValueError:
            raise UpstreamError(f"Deezer {path}: bad response")
        if not isinstance(data, dict):
            data = {}
        if ttl and "error" not in data:
            await self.cache.put_http(key, json.dumps(data), ttl)
        return data

    async def search_tracks(self, artist: str, title: str, limit: int = 10) -> list[DeezerTrack]:
        d = await self._json("/search", {"q": f"{artist} {title}", "limit": limit}, 86400)
        return [t for t in (parse_track(x) for x in d.get("data", []) or []) if t]

    async def find_track(self, artist: str, title: str, duration_ms: int | None = None,
                         isrc: str | None = None, min_ratio: float = 0.8) -> DeezerTrack | None:
        if isrc:
            d = await self._json(f"/track/isrc:{isrc}", None, 600)
            t = parse_track(d)
            if t:
                return t
        want = f"{norm_name(artist)} {norm_title(title)}"
        best, best_r = None, 0.0
        for t in await self.search_tracks(artist, title):
            if duration_ms and t.duration and abs(t.duration - duration_ms / 1000) > 5:
                continue
            r = difflib.SequenceMatcher(None, want, f"{norm_name(t.artist)} {norm_title(t.title)}").ratio()
            if r > best_r:
                best, best_r = t, r
        return best if best and best_r >= min_ratio else None

    async def find_artist(self, name: str) -> tuple[int | None, int | None]:
        """(Deezer artist id, nb_fan) for an exact (normalized) name match."""
        d = await self._json("/search/artist", {"q": name, "limit": 5}, 86400)
        want = norm_name(name)
        for a in d.get("data", []) or []:
            if norm_name(a.get("name", "")) == want:
                fans = a.get("nb_fan")
                return a.get("id"), (int(fans) if isinstance(fans, (int, float)) else None)
        return None, None

    async def find_artist_id(self, name: str) -> int | None:
        return (await self.find_artist(name))[0]

    async def artist_top(self, artist_id: int, limit: int = 3, artist_name: str | None = None) -> list[DeezerTrack]:
        d = await self._json(f"/artist/{artist_id}/top", {"limit": limit}, 86400)
        out = []
        for x in d.get("data", []) or []:
            t = parse_track(x, fallback_artist=artist_name)
            if t:
                out.append(t)
        return out

    async def get_track(self, deezer_id: int) -> DeezerTrack | None:
        d = await self._json(f"/track/{deezer_id}", None, 600)
        return parse_track(d)
