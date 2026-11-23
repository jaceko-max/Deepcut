"""Distance, match score, and diverse selection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.audio.dsp import octave_distance
from app.scoring.normalize import Norm
from app.scoring.vibes import Vibe, centroid_of

# Hand-built DSP groups (tempo, groove, tone, MFCC timbre) plus a learned "style" group: cosine distance
# between Discogs-EffNet style-activation profiles (app/audio/embed.py). On the accuracy benchmark
# (8 playlists, see README) adding style at 0.6 raised own-vs-foreign-playlist AUC from 0.86 to 0.95 and
# the held-out-artist audio AUC from 0.53 to ~0.65. The DSP groups keep their earlier relative weights
# (0.10/0.20/0.35/0.35, chosen by per-group AUC) and share the remaining 0.4. Without a model (EMBEDDINGS=off
# or a clip without style) distance() rescales the DSP part to the full range, so scores stay comparable.
W_STYLE = 0.60
_DSP = 1.0 - W_STYLE
W_TEMPO, W_GROOVE, W_TONE, W_TIMBRE = 0.10 * _DSP, 0.20 * _DSP, 0.35 * _DSP, 0.35 * _DSP
STYLE_SCALE = 0.60  # cosine distance at which the style group saturates (own tracks ~0.05-0.35, others ~0.35-0.9)
TEMPO_SCALE = 0.30
FAMILY_SHARE = 0.3  # at most ~30% of results may come from one playlist artist's neighbourhood
FAMILY_MIN = 2
HALF_DOUBLE_PENALTY = 0.03  # raw tempo ratio ~2: a 60 BPM ballad is not a 120 BPM banger


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def style_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0..1: cosine distance of two style profiles, saturating at STYLE_SCALE."""
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    cos = float(a @ b) / den if den > 0 else 0.0
    return min(1.0, max(0.0, 1.0 - cos) / STYLE_SCALE)


def distance(a: Norm, b: Norm) -> float:
    tempo = min(1.0, octave_distance(a.bpm_folded, b.bpm_folded) / TEMPO_SCALE)
    ratio = max(a.bpm_raw, b.bpm_raw) / max(1e-6, min(a.bpm_raw, b.bpm_raw))
    octave = HALF_DOUBLE_PENALTY if ratio > 1.6 else 0.0
    dsp = (W_TEMPO * tempo + W_GROOVE * _rms(a.groove, b.groove)
           + W_TONE * _rms(a.tone, b.tone) + W_TIMBRE * _rms(a.timbre, b.timbre))
    if a.style is None or b.style is None:
        return float(dsp / _DSP + octave)
    return float(dsp + W_STYLE * style_distance(a.style, b.style) + octave)


def compute_d_ref(norms: list[Norm], vibes: list[Vibe]) -> float:
    """Median leave-one-out distance of playlist tracks to their own vibe centroid."""
    vals = []
    for v in vibes:
        if len(v.members) < 2:
            continue
        for i in v.members:
            others = [norms[j] for j in v.members if j != i]
            vals.append(distance(norms[i], centroid_of(others)))
    d = float(np.median(vals)) if vals else 0.15
    return float(np.clip(d, 0.06, 0.25))


def match_score(d_c: float, d_ref: float) -> int:
    return int(round(100 * math.exp(-math.log(100 / 85) * (d_c / d_ref) ** 2)))


def candidate_distance(c: Norm, norms: list[Norm], vibes: list[Vibe]) -> tuple[int, float]:
    """Nearest vibe id and blended distance D_c."""
    best_v, best_d = 0, float("inf")
    for v in vibes:
        d = distance(c, v.centroid)
        if d < best_d:
            best_v, best_d = v.id, d
    v = vibes[best_v]
    near = sorted(distance(c, norms[j]) for j in v.members)[:3]
    return best_v, 0.7 * best_d + 0.3 * float(np.mean(near))


@dataclass
class Scored:
    key: object
    artist: str
    vibe_id: int
    score: int
    affinity: float
    family: str | None = None  # playlist artist that led to this candidate

    @property
    def rank_score(self) -> float:
        return self.score * (0.75 + 0.25 * self.affinity)


def select_diverse(scored: list[Scored], limit: int, shares: dict[int, float],
                   family_share: dict[str, float] | None = None) -> list[Scored]:
    """<=2 per artist (<=1 in the top 10); fill across vibes in proportion to share."""
    scored = sorted(scored, key=lambda s: -s.rank_score)
    # largest-remainder quotas
    raw = {v: limit * sh for v, sh in shares.items()}
    quota = {v: int(math.floor(x)) for v, x in raw.items()}
    for v in sorted(raw, key=lambda v: raw[v] - quota[v], reverse=True)[: limit - sum(quota.values())]:
        quota[v] += 1
    per_artist: dict[str, int] = {}
    per_family: dict[str, int] = {}
    top_cap = max(3, math.ceil(FAMILY_SHARE * limit))
    # A family's cap follows how much of the playlist its artist is (an artist with 2 of 100 tracks
    # shouldn't supply a quarter of the results); unknown families get the minimum.
    caps = {f: min(top_cap, max(FAMILY_MIN, math.ceil(3 * sh * limit))) for f, sh in (family_share or {}).items()}
    slack = 0  # added to every cap in the backfill below until the list is full

    def family_cap(f: str) -> int:
        return (caps.get(f, FAMILY_MIN) if family_share is not None else top_cap) + slack
    chosen: list[Scored] = []

    def take(s: Scored) -> bool:
        a = s.artist.lower()
        cap = 1 if len(chosen) < 10 else 2  # <=1 per artist in the top 10, <=2 overall
        if per_artist.get(a, 0) >= cap:
            return False
        if s.family and per_family.get(s.family, 0) >= family_cap(s.family):
            return False
        per_artist[a] = per_artist.get(a, 0) + 1
        if s.family:
            per_family[s.family] = per_family.get(s.family, 0) + 1
        chosen.append(s)
        return True

    taken_ids = set()
    for v, q in quota.items():
        n = 0
        for s in scored:
            if n >= q:
                break
            if s.vibe_id == v and id(s) not in taken_ids and take(s):
                taken_ids.add(id(s))
                n += 1
    while len(chosen) < limit and slack <= limit:  # backfill short vibes, loosening family caps
        for s in scored:  # one step at a time so the remaining slots stay balanced across families
            if len(chosen) >= limit:
                break
            if id(s) not in taken_ids and take(s):
                taken_ids.add(id(s))
        slack += 1
    return order_by_score(chosen)[:limit]


def order_by_score(chosen: list[Scored], top: int = 10) -> list[Scored]:
    """Sort by score, then keep at most one track per artist within the top `top` places."""
    pool = sorted(chosen, key=lambda s: (-s.score, -s.affinity))
    ordered: list[Scored] = []
    while pool:
        seen = {s.artist.lower() for s in ordered} if len(ordered) < top else set()
        pick = next((s for s in pool if s.artist.lower() not in seen), pool[0])
        ordered.append(pick)
        pool.remove(pick)
    return ordered
