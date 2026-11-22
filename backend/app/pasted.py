"""Pasted track lists: build the PlaylistData the pipeline expects, and pick reserve results."""
from __future__ import annotations

import hashlib

from app.clients.spotify import PlaylistData, PlaylistTrack


def pasted_playlist(tracks: list[dict], name: str = "Pasted playlist") -> PlaylistData:
    """Tracks have no preview URL, so the pipeline's existing Deezer matching finds audio for them."""
    out = []
    for i, t in enumerate(tracks):
        artists = [a.strip() for a in t["artist"].replace(";", ",\u00a0").split(",\u00a0") if a.strip()]
        out.append(PlaylistTrack(spotify_id=f"pasted{i}", title=t["title"].strip(),
                                 artists=artists or [t["artist"].strip()]))
    digest = hashlib.sha1("\n".join(f"{t['artist']}|{t['title']}" for t in tracks).encode()).hexdigest()[:12]
    return PlaylistData(id=f"pasted-{digest}", name=name, owner=None, image_url=None, tracks=out, source="pasted")


def pick_reserve(pool: list, chosen_keys: set, chosen_artists: set[str], n: int = 40) -> list:
    """Best-scored leftovers (already analyzed and filtered), one per artist first, artists in the list excluded
    in the first pass; a second pass fills the remainder with extra tracks by other/used artists."""
    ranked = sorted((s for s in pool if s.key not in chosen_keys), key=lambda s: -s.rank_score)
    out, seen = [], set(chosen_artists)
    for s in ranked:
        a = s.artist.lower()
        if a not in seen:
            seen.add(a)
            out.append(s)
        if len(out) >= n:
            return out
    taken = {s.key for s in out}
    for s in ranked:
        if s.key not in taken and len(out) < n:
            out.append(s)
    return out
