"""Apple Music playlists: the public share page (https://music.apple.com/<cc>/playlist/<name>/pl.<id>).

No developer token is needed: the page embeds its track list as JSON in
`<script id="serialized-server-data">`. It carries titles and artists but no audio, so tracks come back
without previews and the pipeline finds audio through Deezer, exactly as for pasted track lists.
Unofficial like the Spotify embed scrape: the page format may change.
"""
from __future__ import annotations

import json
import re

from app.clients.http import UpstreamError  # noqa: F401  (re-exported for callers)
from app.clients.spotify import PlaylistData, PlaylistNotFound, PlaylistTrack, PlaylistUnavailable

BROWSER_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
              "Chrome/124.0 Safari/537.36")
URL_RE = re.compile(r"^(?:https?://)?(?:embed\.)?music\.apple\.com/([a-z]{2})/playlist/(?:([^/?#]+)/)?(pl\.[A-Za-z0-9-]{8,64})(?:[/?#].*)?$",
                    re.I)
ID_RE = re.compile(r"^pl\.[A-Za-z0-9-]{8,64}$")
PAGE = "https://music.apple.com/{cc}/playlist/{slug}/{id}"
DATA_RE = re.compile(r'<script type="application/json" id="serialized-server-data">(.*?)</script>', re.S)


def parse_apple_playlist(text: str) -> tuple[str, str] | None:
    """(storefront, playlist id) from an Apple Music playlist link, else None. Only music.apple.com links
    qualify; the fetched URL is rebuilt from the validated parts, never taken from the input."""
    m = URL_RE.match((text or "").strip())
    return (m.group(1).lower(), m.group(3)) if m else None


def apple_page_url(storefront: str, playlist_id: str) -> str:
    return PAGE.format(cc=storefront, slug="playlist", id=playlist_id)


def _artwork(art: dict | None, size: int = 600) -> str | None:
    url = ((art or {}).get("dictionary") or {}).get("url") or ""
    if not url.startswith("https://"):
        return None
    return url.replace("{w}", str(size)).replace("{h}", str(size)).replace("{c}", "bb").replace("{f}", "jpg")


def parse_apple_html(html: str, playlist_id: str) -> PlaylistData:
    m = DATA_RE.search(html)
    if not m:
        raise PlaylistUnavailable("Apple Music page had no data")
    try:
        sections = json.loads(m.group(1))["data"][0]["data"]["sections"]
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        raise PlaylistUnavailable("Apple Music page format not recognized") from e
    header = next((i for s in sections if "header" in str(s.get("id")) for i in s.get("items") or []), {})
    rows = next((s.get("items") or [] for s in sections if s.get("itemKind") == "trackLockup"), None)
    if rows is None:
        raise PlaylistUnavailable("Apple Music page format not recognized")
    tracks: list[PlaylistTrack] = []
    for t in rows:
        desc = t.get("contentDescriptor") or {}
        if desc.get("kind") not in (None, "song"):
            continue
        artists = [(a.get("title") or "").strip() for a in t.get("subtitleLinks") or []]
        artists = [a for a in artists if a]
        title = (t.get("title") or "").strip()
        if not title or not artists:
            continue
        adam = str((desc.get("identifiers") or {}).get("storeAdamID") or len(tracks))
        tracks.append(PlaylistTrack(spotify_id=f"apple{adam}", title=title, artists=artists,
                                    duration_ms=t.get("duration") if isinstance(t.get("duration"), int) else None))
    owner = next((l.get("title") for l in header.get("subtitleLinks") or [] if l.get("title")), None)
    return PlaylistData(id=playlist_id, name=(header.get("title") or "Apple Music playlist").strip(), owner=owner,
                        image_url=_artwork(header.get("artwork")), tracks=tracks, source="apple_music")


class AppleMusicClient:
    def __init__(self, http):
        self.http = http  # HttpClient

    async def fetch_playlist(self, storefront: str, playlist_id: str) -> PlaylistData:
        resp = await self.http.get(apple_page_url(storefront, playlist_id),
                                   headers={"User-Agent": BROWSER_UA, "Accept": "text/html"})
        if resp.status_code in (400, 403, 404):
            raise PlaylistNotFound("Apple Music playlist not found or not public")
        if resp.status_code != 200:
            raise PlaylistUnavailable(f"Apple Music returned HTTP {resp.status_code}")
        return parse_apple_html(resp.text, playlist_id)
