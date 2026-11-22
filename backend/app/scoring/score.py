"""Distance, match score, and diverse selection."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.audio.dsp import octave_distance
from app.scoring.normalize import Norm
from app.scoring.vibes import Vibe, centroid_of

# Re-weighted from per-group AUC (own-playlist vs foreign tracks, 12 playlist pairs): timbre 0.68-0.85,
# tone 0.62-0.81, groove 0.46-0.72, tempo 0.47-0.63. The old 0.25/0.30/0.25/0.20 with a 0.15 tempo
# scale saturated the tempo term (~0.2 added to every distance), pushing D_ref to the 0.25 clamp so
# everything scored 80-96. New weights: mean AUC 0.694 -> ~0.81.
W_TEMPO, W_GROOVE, W_TONE, W_TIMBRE = 0.10, 0.20, 0.35, 0.35
TEMPO_SCALE = 0.30
HALF_DOUBLE_PENALTY = 0.03  # raw tempo ratio ~2: a 60 BPM ballad is not a 120 BPM banger


def _rms(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def distance(a: Norm, b: Norm) -> float:
    tempo = min(1.0, octave_distance(a.bpm_folded, b.bpm_folded) / TEMPO_SCALE)
    ratio = max(a.bpm_raw, b.bpm_raw) / max(1e-6, min(a.bpm_raw, b.bpm_raw))
    octave = HALF_DOUBLE_PENALTY if ratio > 1.6 else 0.0
    return float(W_TEMPO * tempo + W_GROOVE * _rms(a.groove, b.groove)
                 + W_TONE * _rms(a.tone, b.tone) + W_TIMBRE * _rms(a.timbre, b.timbre) + octave)


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

    @property
    def rank_score(self) -> float:
        return self.score * (0.75 + 0.25 * self.affinity)


def select_diverse(scored: list[Scored], limit: int, shares: dict[int, float]) -> list[Scored]:
    """<=2 per artist (<=1 in the top 10); fill across vibes in proportion to share."""
    scored = sorted(scored, key=lambda s: -s.rank_score)
    # largest-remainder quotas
    raw = {v: limit * sh for v, sh in shares.items()}
    quota = {v: int(math.floor(x)) for v, x in raw.items()}
    for v in sorted(raw, key=lambda v: raw[v] - quota[v], reverse=True)[: limit - sum(quota.values())]:
        quota[v] += 1
    per_artist: dict[str, int] = {}
    chosen: list[Scored] = []

    def take(s: Scored) -> bool:
        a = s.artist.lower()
        cap = 1 if len(chosen) < 10 else 2  # <=1 per artist in the top 10, <=2 overall
        if per_artist.get(a, 0) >= cap:
            return False
        per_artist[a] = per_artist.get(a, 0) + 1
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
    for s in scored:  # backfill if some vibe had too few candidates
        if len(chosen) >= limit:
            break
        if id(s) not in taken_ids and take(s):
            taken_ids.add(id(s))
    chosen.sort(key=lambda s: (-s.score, -s.affinity))
    # at most one track per artist within the top 10
    ordered: list[Scored] = []
    pool = list(chosen)
    while pool:
        seen = {s.artist.lower() for s in ordered} if len(ordered) < 10 else set()
        pick = next((s for s in pool if s.artist.lower() not in seen), pool[0])
        ordered.append(pick)
        pool.remove(pick)
    return ordered[:limit]
