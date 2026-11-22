"""Playlist reading: public embed page (primary) and optional Web API."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field

import httpx

from app.clients.http import UpstreamError
from app.config import Settings
from app.textutil import split_artists

ID_RE = re.compile(r"^[A-Za-z0-9]{22}$")
EMBED_URL = "https://open.spotify.com/embed/playlist/{id}"
BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


TOKEN_REFRESH_MARGIN = 60.0


class PlaylistNotFound(Exception):
    pass


class PlaylistUnavailable(Exception):
    pass


def parse_playlist_id(text: str) -> str | None:
    """Accept open.spotify.com URLs (optionally /intl-xx/), spotify:playlist:ID, or a bare 22-char id."""
    s = (text or "").strip()
    if ID_RE.match(s):
        return s
    m = re.match(r"^spotify:playlist:([A-Za-z0-9]{22})$", s)
    if m:
        return m.group(1)
    m = re.match(r"^(?:https?://)?open\.spotify\.com/(?:intl-[A-Za-z-]+/)?(?:embed/)?playlist/([A-Za-z0-9]{22})(?:[/?#].*)?$", s)
    return m.group(1) if m else None


@dataclass
class PlaylistTrack:
    spotify_id: str
    title: str
    artists: list[str]
    duration_ms: int | None = None
    preview_url: str | None = None
    isrc: str | None = None

    @property
    def primary_artist(self) -> str:
        return self.artists[0] if self.artists else ""


@dataclass
class PlaylistData:
    id: str
    name: str
    owner: str | None
    image_url: str | None
    tracks: list[PlaylistTrack] = field(default_factory=list)
    source: str = "spotify_embed"  # spotify_embed | spotify_api


def parse_embed_html(html: str, playlist_id: str) -> PlaylistData:
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', html, re.S)
    if not m:
        raise PlaylistUnavailable("Spotify embed page had no data")
    try:
        entity = json.loads(m.group(1))["props"]["pageProps"]["state"]["data"]["entity"]
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise PlaylistUnavailable("Spotify embed page format not recognized") from e
    if not isinstance(entity, dict) or "trackList" not in entity:
        raise PlaylistUnavailable("Spotify embed page format not recognized")
    tracks: list[PlaylistTrack] = []
    for t in entity.get("trackList") or []:
        uri = t.get("uri") or ""
        if not uri.startswith("spotify:track:"):
            continue
        prev = (t.get("audioPreview") or {}).get("url")
        tracks.append(PlaylistTrack(
            spotify_id=uri.split(":")[-1],
            title=(t.get("title") or "").strip(),
            artists=split_artists(t.get("subtitle") or ""),
            duration_ms=t.get("duration"),
            preview_url=prev or None,
        ))
    srcs = ((entity.get("coverArt") or {}).get("sources")) or []
    image = None
    if srcs:
        image = max(srcs, key=lambda s: s.get("width") or 0).get("url")
    return PlaylistData(
        id=playlist_id, name=entity.get("name") or entity.get("title") or "Untitled playlist",
        owner=entity.get("subtitle") or None, image_url=image, tracks=tracks, source="spotify_embed",
    )


class SpotifyClient:
    def __init__(self, http, settings: Settings):
        self.http = http  # HttpClient
        self.settings = settings
        self._token: str | None = None
        self._token_exp: float = 0.0  # time.monotonic() deadline, already shortened by the refresh margin

    async def fetch_playlist(self, playlist_id: str) -> PlaylistData:
        mode = self.settings.spotify_playlist_source
        use_api = mode == "api" or (mode == "auto" and self.settings.has_spotify_credentials)
        if use_api and self.settings.has_spotify_credentials:
            try:
                data = await self._fetch_api(playlist_id)
                if data is not None:
                    return data
            except (httpx.HTTPError, PlaylistUnavailable, UpstreamError, ValueError):
                pass
            if mode == "api":
                raise PlaylistUnavailable("Spotify API did not expose this playlist's tracks")
        return await self._fetch_embed(playlist_id)

    async def _fetch_embed(self, playlist_id: str) -> PlaylistData:
        resp = await self.http.get(EMBED_URL.format(id=playlist_id), headers={"User-Agent": BROWSER_UA, "Accept": "text/html"})
        if resp.status_code == 404:
            raise PlaylistNotFound("Playlist not found or private")
        if resp.status_code != 200:
            raise PlaylistUnavailable(f"Spotify embed returned HTTP {resp.status_code}")
        return parse_embed_html(resp.text, playlist_id)

    async def _get_token(self, force: bool = False) -> str | None:
        if self._token and not force and time.monotonic() < self._token_exp:
            return self._token
        self._token = None
        resp = await self.http.request(
            "POST", "https://accounts.spotify.com/api/token",
            data={"grant_type": "client_credentials"},
            auth=(self.settings.spotify_client_id, self.settings.spotify_client_secret),
        )
        if resp.status_code != 200:
            return None
        try:
            body = resp.json()
            token = body.get("access_token")
            expires_in = float(body.get("expires_in") or 3600)
        except (ValueError, AttributeError, TypeError):
            return None
        if token:
            self._token = token
            self._token_exp = time.monotonic() + max(0.0, expires_in - TOKEN_REFRESH_MARGIN)
        return self._token

    async def _fetch_api(self, playlist_id: str) -> PlaylistData | None:
        """Return None when the API does not expose tracks (caller falls back to embed)."""
        token = await self._get_token()
        if not token:
            return None
        url = f"https://api.spotify.com/v1/playlists/{playlist_id}"
        resp = await self.http.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code == 401:  # token revoked/expired early: refresh once and retry
            token = await self._get_token(force=True)
            if not token:
                return None
            resp = await self.http.get(url, headers={"Authorization": f"Bearer {token}"})
        if resp.status_code != 200:
            return None
        try:
            meta = resp.json()
        except ValueError:
            return None
        if not isinstance(meta, dict):
            return None
        container = meta.get("items") or meta.get("tracks") or {}
        entries = container.get("items") if isinstance(container, dict) else None
        if not entries:
            return None
        tracks: list[PlaylistTrack] = []
        for e in entries:
            t = e.get("item") or e.get("track") or {}
            if not t or t.get("type") not in (None, "track") or not t.get("id"):
                continue
            tracks.append(PlaylistTrack(
                spotify_id=t["id"], title=t.get("name") or "",
                artists=[a.get("name", "") for a in t.get("artists", []) if a.get("name")],
                duration_ms=t.get("duration_ms"), preview_url=t.get("preview_url"),
                isrc=(t.get("external_ids") or {}).get("isrc"),
            ))
        if not tracks:
            return None
        imgs = meta.get("images") or []
        return PlaylistData(
            id=playlist_id, name=meta.get("name") or "Untitled playlist",
            owner=(meta.get("owner") or {}).get("display_name"),
            image_url=imgs[0].get("url") if imgs else None, tracks=tracks, source="spotify_api",
        )
