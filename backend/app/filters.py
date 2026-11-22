"""Contamination filters for candidate tracks: holiday songs, live takes, karaoke/tribute/cover
versions, sped-up/slowed edits, instrumental versions and film/TV soundtrack tie-ins.

A category is only rejected when the playlist itself does not lean that way (e.g. a Christmas
playlist keeps Christmas songs): see `allowed_categories`.
"""
from __future__ import annotations

import re

_I = re.I
# Patterns applied to the track title.
TITLE_PATTERNS: dict[str, re.Pattern] = {
    "holiday": re.compile(
        r"christmas|x-?mas\b|navidad|weihnacht|\bnoël\b|happy holidays|holiday (song|season|classic|collection|hit|magic|edition)|jingle bell|santa (claus|baby)|"
        r"sleigh|mistletoe|winter wonderland|silent night|let it snow|rudolph|deck the hall|hanukk?ah|"
        r"little drummer boy|feliz navidad|o holy night|frosty the snowman|auld lang syne|"
        r"stocking stuffer|\bstockings?\b|under the (christmas )?tree|\bsanta\b|north pole|reindeer|yuletide|\bsnowman\b", _I),
    "live": re.compile(
        r"[(\[]\s*live\b|[-–—]\s*live\b|\blive (at|from|on|in concert|session|version|acoustic)\b|"
        r"\bunplugged\b|[(\[]\s*(en vivo|en directo|ao vivo)|[-–—]\s*(en vivo|en directo|ao vivo)", _I),
    "karaoke": re.compile(r"karaoke|in the style of|made famous by|backing track|sing-?along version", _I),
    "tribute": re.compile(r"\btribute\b", _I),
    "cover": re.compile(r"[(\[][^)\]]*\bcover\b[^)\]]*[)\]]|[-–—]\s*[^-–—]*\bcover\b|\bcover version\b|"
                        r"originally (performed|recorded|by)", _I),
    "altered": re.compile(r"sped[ -]?up|speed[ -]?up|\bslowed\b|\breverb(ed)?\b.*\bslowed|\b8d\b|nightcore|daycore|"
                          r"super slowed|\btiktok (remix|version)", _I),
    "instrumental": re.compile(r"instrumental version|[(\[]\s*instrumental\s*[)\]]|[-–—]\s*instrumental\s*$", _I),
    "soundtrack": re.compile(r"\(from [\"“']|\[from [\"“']|from the (original )?(motion picture|film|movie|series|"
                             r"netflix|disney|musical|soundtrack)|\bsoundtrack\b|\bost\b|from disney|"
                             r"\bmusic inspired by\b|\binspired by the (motion picture|film|movie|series)\b", _I),
}
# Patterns applied to the album title.
ALBUM_PATTERNS: dict[str, re.Pattern] = {
    "holiday": TITLE_PATTERNS["holiday"],
    "live": re.compile(r"\blive (at|from|in|on)\b|[(\[]\s*live\b|^live$|\blive\s+(19|20)\d\d\b|\bunplugged\b|"
                       r"\b(en vivo|en directo|ao vivo)\b|\bmtv unplugged\b|\blive sessions?\b", _I),
    "karaoke": TITLE_PATTERNS["karaoke"],
    "tribute": re.compile(r"\btribute\b|\bsalute to\b|\bplays the music of\b|\bsongs of\b.*\bperformed by\b", _I),
    "cover": re.compile(r"\bcovers\b|\bcover songs\b|\bcovers? album\b", _I),
    "altered": TITLE_PATTERNS["altered"],
    "soundtrack": re.compile(r"soundtrack|original (motion picture|score|cast)|\bost\b|\bmusic from the\b|\bmusic inspired by\b|"
                             r"\bdisney\b|\bfrom the (motion picture|film|series|netflix)\b|\bthe musical\b", _I),
}
# Artist names that indicate tribute/karaoke/cover acts or cast recordings.
ARTIST_PATTERN = re.compile(
    r"\btribute\b|\bkaraoke\b|cover band|\bcovers\b|in the style of|made famous|\brockabye baby\b|"
    r"vitamin string quartet|piano tribute|lullaby|\bthe hit crew\b|\bsound-?alike|\b8-?bit\b|"
    r"^(the )?(original )?(broadway |london )?cast\b|\bcast of\b|\bcast recording\b|\bdisney\b|"
    r"\bsing-?along\b|\bmusic box\b|"
    r"(experience|revival|legacy|show|tribute)\s*band$", _I)

PLAYLIST_ALLOW_SHARE = 0.08


def track_flags(title: str | None, album: str | None = None) -> set[str]:
    out = set()
    t, a = title or "", album or ""
    for cat, pat in TITLE_PATTERNS.items():
        if pat.search(t):
            out.add(cat)
    for cat, pat in ALBUM_PATTERNS.items():
        if a and pat.search(a):
            out.add(cat)
    return out


def artist_is_tribute(name: str | None) -> bool:
    return bool(name and ARTIST_PATTERN.search(name))


def allowed_categories(titles: list[str], albums: list[str | None] | None = None) -> set[str]:
    """Categories the playlist itself contains at a similar share (>= 8% of its tracks)."""
    if not titles:
        return set()
    albums = albums or [None] * len(titles)
    counts: dict[str, int] = {}
    for t, a in zip(titles, albums):
        for c in track_flags(t, a):
            counts[c] = counts.get(c, 0) + 1
    return {c for c, n in counts.items() if n / len(titles) >= PLAYLIST_ALLOW_SHARE}


def is_contaminated(title: str | None, album: str | None, artist: str | None, allowed: set[str] = frozenset()) -> str | None:
    """Return the rejecting category, or None if the track is clean."""
    if artist_is_tribute(artist) and "tribute" not in allowed:
        return "tribute"
    bad = track_flags(title, album) - set(allowed)
    return sorted(bad)[0] if bad else None
