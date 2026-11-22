from __future__ import annotations

import re
import unicodedata


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def clean_title(title: str) -> str:
    """Strip feat./remaster/version decorations from a track title."""
    t = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", title)
    t = re.sub(r"\s+-\s+.*$", "", t)
    t = t.strip()
    return t or title.strip()


def norm_name(s: str) -> str:
    """Normalize an artist/title for comparison."""
    s = _fold(s).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strict_name(s: str) -> str:
    """Like norm_name but keeps '+', '&' and 'and' distinct from nothing: 'Florence + The Machine'
    != 'Florence The Machine' (a bogus Last.fm autocorrect target), while case/accents still fold."""
    s = _fold(s).lower()
    s = re.sub(r"\s*(\+|&|\band\b)\s*", " and ", s)
    s = re.sub(r"[^a-z0-9 ]+", "", s)
    return re.sub(r"\s+", " ", s).strip()


def norm_title(s: str) -> str:
    return norm_name(clean_title(s))


def split_artists(subtitle: str) -> list[str]:
    # The embed joins credits with ",\u00a0"; a plain ", " belongs to a name like "Tyler, The Creator".
    parts = (subtitle or "").split(",\u00a0")
    return [a for a in (re.sub(r"\s+", " ", p).strip() for p in parts) if a]
