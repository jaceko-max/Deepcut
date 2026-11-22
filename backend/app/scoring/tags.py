"""Genre-tag coherence: Last.fm top-tag vectors, a weighted playlist profile, cosine similarity.

Tag weights are a fixed "IDF-ish" prior instead of an IDF computed from the job's own candidates:
the candidates of a shoegaze playlist are mostly tagged "shoegaze", so a corpus IDF would punish
exactly the tag that matters. Umbrella genres ("pop", "rock", "electronic", ...) get a low weight,
specific genres the full weight of 1.0.
"""
from __future__ import annotations

import math
import re
from collections import Counter

JUNK_TAGS = {
    "seen live", "favorites", "favourites", "favorite", "favourite", "female vocalists", "male vocalists",
    "female vocalist", "male vocalist", "female vocals", "male vocals", "spotify", "albums i own", "all",
    "under 2000 listeners", "beautiful", "awesome", "love", "cool", "good", "amazing", "best", "music",
    "my music", "check out", "to listen", "new", "fun", "sexy", "chill", "catchy", "lovely", "epic",
    "favorite songs", "favourite songs", "loved", "great", "a", "singer", "songwriter", "vocalists",
    "female", "male", "artist", "band", "bands", "musician", "producer", "rapper", "dj", "composer",
    "all time favorites", "heard on pandora", "youtube", "tiktok", "viral", "radio", "billboard",
    "grammy", "grammy winner", "spotify singles",
}
NATIONALITIES = {
    "american", "british", "english", "uk", "usa", "us", "canadian", "australian", "irish", "scottish",
    "welsh", "swedish", "norwegian", "danish", "finnish", "german", "french", "italian", "spanish",
    "japanese", "korean", "brazilian", "mexican", "dutch", "belgian", "icelandic", "polish", "russian",
    "new zealand", "south african", "nigerian", "jamaican", "puerto rico", "puerto rican", "colombian",
    "argentinian", "argentina", "chilean", "latin american", "london", "los angeles", "new york",
    "atlanta", "chicago", "toronto", "california", "texas", "manchester", "brooklyn", "seattle", "detroit",
    "france", "germany", "sweden", "japan", "brazil", "mexico", "canada", "australia", "spain", "finland",
    "united states", "united kingdom", "finnish", "suomi", "latin",
}
SYNONYMS = {
    "hip-hop": "hip hop", "hiphop": "hip hop", "dreampop": "dream pop", "dream-pop": "dream pop",
    "r&b": "rnb", "r and b": "rnb", "rhythm and blues": "rnb", "lo-fi": "lofi", "lo fi": "lofi",
    "post punk": "post-punk", "synth pop": "synthpop", "synth-pop": "synthpop",
    "alt-country": "alt country", "trip-hop": "trip hop", "triphop": "trip hop", "drum n bass": "drum and bass",
    "drum'n'bass": "drum and bass", "dnb": "drum and bass", "k pop": "k-pop", "kpop": "k-pop",
    "electro-pop": "electropop", "electro pop": "electropop", "neo soul": "neo-soul", "nu jazz": "nu-jazz",
    "uk hip-hop": "uk hip hop", "alternative hip-hop": "alternative hip hop", "lofi hip-hop": "lofi hip hop",
    "lo-fi hip hop": "lofi hip hop", "techhouse": "tech house", "tech-house": "tech house",
}
# Low weight = umbrella tag that says little about fit. Missing tags weigh 1.0.
GENERIC_WEIGHTS = {
    "pop": 0.25, "rock": 0.3, "electronic": 0.35, "indie": 0.4, "alternative": 0.35, "dance": 0.4,
    "hip hop": 0.55, "rap": 0.55, "soul": 0.6, "rnb": 0.6, "jazz": 0.7, "folk": 0.6, "metal": 0.6,
    "punk": 0.6, "country": 0.7, "experimental": 0.5, "acoustic": 0.4, "singer-songwriter": 0.45,
    "alternative rock": 0.55, "indie rock": 0.6, "indie pop": 0.6, "electronica": 0.55, "instrumental": 0.45,
    "ambient": 0.75, "classic rock": 0.6, "soundtrack": 0.4, "underground": 0.3, "spanish": 0.6, "oldies": 0.3,
    "chillout": 0.5, "downtempo": 0.75, "psychedelic": 0.65, "pop rock": 0.5, "edm": 0.55, "house": 0.75,
    "trap": 0.75, "latin": 0.7, "blues": 0.7, "funk": 0.7, "christian": 0.8, "world": 0.5,
}
NATIONALITY_WEIGHT = 0.6
MAX_TAGS = 15
MIN_COUNT = 3


def clean_tag(name: str) -> str | None:
    n = re.sub(r"\s+", " ", (name or "").strip().lower())
    n = SYNONYMS.get(n, n)
    if not n or len(n) > 40 or n in JUNK_TAGS:
        return None
    if re.fullmatch(r"(19|20)?\d{2}s?|\d{4}", n):           # years, decades
        return None
    if re.search(r"under \d+ listeners|seen live|favou?rite|my |albums? i |^\d+ of \d+|\d+/10|^best of", n):
        return None
    return n


def tag_vector(tags: list[tuple[str, int]] | None) -> dict[str, float]:
    """Last.fm top tags -> {tag: count/100}, junk removed, top MAX_TAGS kept."""
    v: Counter = Counter()
    for name, cnt in tags or []:
        n = clean_tag(name)
        if n is None:
            continue
        try:
            c = float(cnt)
        except (TypeError, ValueError):
            continue
        if c >= MIN_COUNT:
            v[n] = max(v[n], c / 100.0)
    return dict(v.most_common(MAX_TAGS))


def tag_weight(tag: str) -> float:
    if tag in NATIONALITIES:
        # Kept on purpose: scene/language tags ("finnish", "spanish", "uk") separate Finnish rap from
        # a US/UK pop playlist that also has "rap" in its profile. Moderate weight: they are broad.
        return NATIONALITY_WEIGHT
    return GENERIC_WEIGHTS.get(tag, 1.0)


def weighted(v: dict[str, float]) -> dict[str, float]:
    return {k: x * tag_weight(k) for k, x in v.items()}


def _unit(v: dict[str, float]) -> dict[str, float]:
    n = math.sqrt(sum(x * x for x in v.values()))
    return {k: x / n for k, x in v.items()} if n else {}


def build_profile(artist_tags: dict[str, dict[str, float]], artist_counts: dict[str, float]) -> dict[str, float]:
    """Sum of per-artist unit vectors (IDF-weighted), each weighted by sqrt(tracks in playlist)
    so one dominant artist cannot define the whole profile. Returns a unit vector."""
    prof: Counter = Counter()
    for a, v in artist_tags.items():
        if not v:
            continue
        w = math.sqrt(max(1.0, float(artist_counts.get(a, 1))))
        for k, x in _unit(weighted(v)).items():
            prof[k] += w * x
    return _unit(dict(prof))


def similarity(cand: dict[str, float], profile: dict[str, float]) -> float:
    """Cosine of the candidate's IDF-weighted tag vector with a (unit) profile."""
    if not cand or not profile:
        return 0.0
    c = _unit(weighted(cand))
    return float(max(0.0, sum(x * profile.get(k, 0.0) for k, x in c.items())))


def shared_tags(cand: dict[str, float], profile: dict[str, float], n: int = 2, min_weight: float = 0.5) -> list[str]:
    """The most informative tags the candidate shares with the profile (umbrella tags skipped)."""
    c = weighted(cand)
    common = [(c[k] * profile[k], k) for k in c if k in profile and tag_weight(k) >= min_weight]
    return [k for _, k in sorted(common, reverse=True)[:n]]


def top_profile_tags(profile: dict[str, float], n: int = 5) -> list[str]:
    return [k for k, _ in sorted(profile.items(), key=lambda kv: -kv[1])[:n]]
