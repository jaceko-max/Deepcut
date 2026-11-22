"""Human-readable 'why' strings and vibe labels."""
from __future__ import annotations

import numpy as np

from app.scoring.normalize import Norm

MAX_LEN = 90


def vibe_label(norm: Norm, bpm: float, ref: Norm | None = None, exclude: set[int] = frozenset(),
               used: list[int] | None = None) -> str:
    """With `ref` (the whole playlist), traits are described relative to it so sibling vibes read differently."""
    used = [] if used is None else used
    t, g = norm.tone, norm.groove
    if ref is None:
        base_t, base_g = [0.5] * len(t), [0.5] * len(g)
        words = [("dark", "bright"), ("light low end", "bass-heavy"), ("sparse drums", "busy drums"),
                 ("straight beat", "syncopated"), ("soft", "loud")]
    else:
        base_t, base_g = ref.tone, ref.groove
        words = [("darker", "brighter"), ("lighter low end", "heavier low end"), ("sparser drums", "busier drums"),
                 ("straighter beat", "more syncopated"), ("softer", "louder")]
    devs = [t[0] - base_t[0], t[2] - base_t[2], g[1] - base_g[1], g[3] - base_g[3], t[4] - base_t[4]]
    cands = sorted(((abs(d), i, words[i][int(d >= 0)]) for i, d in enumerate(devs)
                    if i not in exclude), key=lambda c: -c[0])
    used.extend(c[1] for c in cands[:2])
    return " · ".join([f"{round(bpm)} BPM"] + [c[2] for c in cands[:2]])


def vibe_labels(cens: list[Norm], bpms: list[float], ref: Norm | None) -> list[str]:
    """Labels for sibling vibes. With 2 vibes the deviations from the overall centroid are mirror
    images, so each later vibe skips the traits an earlier one already used and is described by its
    own next most distinctive traits."""
    if ref is None or len(cens) < 2:
        return [vibe_label(c, b, ref) for c, b in zip(cens, bpms)]
    used: list[int] = []
    return [vibe_label(c, b, ref, exclude=set(used), used=used) for c, b in zip(cens, bpms)]


def _tempo_clause(cand_raw: float, vibe_bpm: float) -> str:
    r = cand_raw / vibe_bpm
    x = round(vibe_bpm)
    if abs(r - 1) <= 0.03:
        return f"Same {x} BPM pocket"
    if abs(r / 0.5 - 1) <= 0.06:
        return f"Half-time take on your {x} BPM groove"
    if abs(r / 2 - 1) <= 0.06:
        return f"Double-time take on your {x} BPM groove"
    return f"Near your {x} BPM"


GROOVE_SAME = [  # index in groove vector, (text if playlist value high, text if low)
    (3, "matching syncopation", "matching straight feel"),
    (1, "same busy hats", "same sparse drums"),
    (4, "same steady pulse", "same loose pulse"),
    (0, "same punch", "same soft attack"),
]
DIFFS = [  # (vector, idx, positive text, negative text)
    ("tone", 0, "brighter", "darker"),
    ("tone", 2, "heavier low end", "lighter low end"),
    ("groove", 0, "punchier beat", "looser beat"),
    ("groove", 1, "busier drums", "sparser drums"),
    ("groove", 3, "more syncopated", "less syncopated"),
    ("tone", 4, "louder", "softer"),
]


def explain(cand: Norm, cand_bpm_raw: float, vibe: Norm, vibe_bpm: float) -> str:
    clauses = [_tempo_clause(cand_bpm_raw, vibe_bpm)]
    # closest groove trait
    best = min(GROOVE_SAME, key=lambda s: abs(cand.groove[s[0]] - vibe.groove[s[0]]))
    if abs(cand.groove[best[0]] - vibe.groove[best[0]]) < 0.12:
        clauses.append(best[1] if vibe.groove[best[0]] >= 0.5 else best[2])
    # biggest difference
    diffs = []
    for vec, i, pos, neg in DIFFS:
        d = getattr(cand, vec)[i] - getattr(vibe, vec)[i]
        if abs(d) >= 0.12:
            diffs.append((abs(d), pos if d > 0 else neg))
    if diffs:
        clauses.append(max(diffs)[1])
    while len(", ".join(clauses)) > MAX_LEN and len(clauses) > 1:
        clauses.pop()
    return ", ".join(clauses)[:MAX_LEN]


def with_tags(why: str, tags: list[str]) -> str:
    """Append '; shared tags: a, b' when it fits in MAX_LEN, dropping trailing audio clauses
    (never the first) or the second tag to make room; otherwise return `why` unchanged."""
    tags = [t for t in tags if t][:2]
    if not tags:
        return why
    clauses = why.split(", ")
    for k in range(len(clauses), 0, -1):
        for n in range(len(tags), 0, -1):
            s = ", ".join(clauses[:k]) + "; shared tags: " + ", ".join(tags[:n])
            if len(s) <= MAX_LEN:
                return s
    return why
