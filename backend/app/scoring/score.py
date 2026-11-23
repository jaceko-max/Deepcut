"""Distance, match score, and diverse selection."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from app.audio.dsp import octave_distance
from app.scoring.mood import mood_distance
from app.scoring.normalize import Norm
from app.scoring.vibes import Vibe, centroid_of

# Hand-built DSP groups (tempo, groove, tone, MFCC timbre) plus a learned "style" group: cosine distance
# between Discogs-EffNet style-activation profiles (app/audio/embed.py). On the accuracy benchmark
# (8 playlists, see README) adding style at 0.6 raised own-vs-foreign-playlist AUC from 0.86 to 0.95 and
# the held-out-artist audio AUC from 0.50 to 0.59 (same candidates). DSP groups keep their earlier relative weights
# (0.10/0.20/0.35/0.35, chosen by per-group AUC) and share the remaining 0.4. Without a model (EMBEDDINGS=off
# or a clip without style) distance() rescales the DSP part to the full range, so scores stay comparable.
#
# Mood first: with the Essentia mood heads (app/scoring/mood.py) the distance is 0.53 mood + 0.29 style +
# 0.18 DSP. Benchmark (same traced clips re-scored): held-out-artist audio AUC 0.587 -> 0.675, blended-score
# AUC 0.683 -> 0.722, own-vs-foreign AUC 0.949 -> 0.945; mood alone already reaches 0.668 audio AUC.
# Splits from 0.40/0.40/0.20 to 0.70/0.20/0.10 were all within 0.01 of each other.
W_STYLE = 0.60
_DSP = 1.0 - W_STYLE
W_TEMPO, W_GROOVE, W_TONE, W_TIMBRE = 0.10 * _DSP, 0.20 * _DSP, 0.35 * _DSP, 0.35 * _DSP
W_MOOD_M, W_STYLE_M, W_DSP_M = 0.53, 0.29, 0.18  # used when both clips have mood vectors
STYLE_SCALE = 0.60  # cosine distance at which the style group saturates (own tracks ~0.05-0.35, others ~0.35-0.9)
TEMPO_SCALE = 0.30
D_REF_CLIP = (0.06, 0.25)       # bounds on the playlist's typical own distance (score 85 at d_ref)
D_REF_CLIP_MOOD = (0.12, 0.45)  # mood-first distances run ~2x the DSP/style ones (benchmark medians 0.23-0.66)
FAMILY_SHARE = 0.3  # at most ~30% of results may come from one playlist artist's neighbourhood
FAMILY_MIN = 2
TIER_FAMILY_SLACK = 2  # extra per-family slots allowed to fill a popularity tier's quota
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
           + W_TONE * _rms(a.tone, b.tone) + W_TIMBRE * _rms(a.timbre, b.timbre)) / _DSP  # 0..1
    if a.style is None or b.style is None:
        return float(dsp + octave)
    style = style_distance(a.style, b.style)
    if a.mood is None or b.mood is None:
        return float(_DSP * dsp + W_STYLE * style + octave)
    return float(W_MOOD_M * mood_distance(a.mood, b.mood) + W_STYLE_M * style + W_DSP_M * dsp + octave)


def compute_d_ref(norms: list[Norm], vibes: list[Vibe]) -> float:
    """Median leave-one-out distance of playlist tracks to their own vibe centroid."""
    vals = []
    for v in vibes:
        if len(v.members) < 2:
            continue
        for i in v.members:
            others = [norms[j] for j in v.members if j != i]
            vals.append(distance(norms[i], centroid_of(others)))
    lo, hi = D_REF_CLIP_MOOD if norms and all(n.mood is not None for n in norms) else D_REF_CLIP
    d = float(np.median(vals)) if vals else (lo + hi) / 2
    return float(np.clip(d, lo, hi))


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


ENVELOPE_PCT = 80      # a vibe's mood envelope: 80th percentile of its members' leave-one-out mood distance...
ENVELOPE_FACTOR = 1.3  # ...times 1.3. Benchmark: keeps 96% of held-out playlist artists' clips and 91% of other
                       # candidates, rejects 66% of other playlists' tracks (p90 x 1.2: 96% / 92% / 62%).
NEAR_DUP_PCT = 1       # two picks closer than 0.6x the 1st percentile of the playlist's own cross-artist pair
NEAR_DUP_FACTOR = 0.6  # distances are near-duplicates; only the better one is kept. (Candidates sit closer to
                       # each other than playlist tracks do: on the user benchmark playlist the 3rd percentile
                       # would have flagged 10% of candidate pairs, this flags ~0.3%.)


def _mood_gap(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))  # unsaturated, unlike mood_distance()


def mood_envelope(norms: list[Norm], vibes: list[Vibe]) -> dict[int, float] | None:
    """vibe id -> mood radius (ENVELOPE_PCT of members' leave-one-out distance to the vibe's mood centroid,
    times ENVELOPE_FACTOR). None without mood vectors."""
    if not norms or any(n.mood is None for n in norms):
        return None
    out = {}
    for v in vibes:
        ms = np.array([norms[j].mood for j in v.members])
        if len(ms) < 3:
            continue
        tot = ms.sum(0)
        loo = [_mood_gap(m, (tot - m) / (len(ms) - 1)) for m in ms]
        out[v.id] = ENVELOPE_FACTOR * float(np.percentile(loo, ENVELOPE_PCT))
    return out or None


def envelope_ratio(n: Norm, vibes: list[Vibe], radii: dict[int, float] | None) -> float | None:
    """Mood distance to the nearest vibe centroid / that vibe's radius (<= 1: inside). None = no check."""
    if radii is None or n.mood is None:
        return None
    return min((_mood_gap(n.mood, v.centroid.mood) / max(radii[v.id], 1e-6)
                for v in vibes if v.id in radii and v.centroid.mood is not None), default=None)


def near_dup_threshold(norms: list[Norm], artists: list[str]) -> float:
    """NEAR_DUP_FACTOR x the NEAR_DUP_PCT percentile of distances between playlist tracks by different artists."""
    d = [distance(norms[i], norms[j]) for i in range(len(norms)) for j in range(i + 1, len(norms))
         if artists[i] != artists[j]]
    return NEAR_DUP_FACTOR * float(np.percentile(d, NEAR_DUP_PCT)) if d else 0.0


def own_scores(norms: list[Norm], vibes: list[Vibe], d_ref: float) -> list[int]:
    """Leave-one-out audio match score of every playlist track against the rest of its playlist."""
    out = []
    for v in vibes:
        if len(v.members) < 3:
            continue
        for i in v.members:
            rest = [j for j in v.members if j != i]
            cen = centroid_of([norms[j] for j in rest])
            near = sorted(distance(norms[i], norms[j]) for j in rest)[:3]
            out.append(match_score(0.7 * distance(norms[i], cen) + 0.3 * float(np.mean(near)), d_ref))
    return out


@dataclass
class Scored:
    key: object
    artist: str
    vibe_id: int
    score: int
    affinity: float
    family: str | None = None  # playlist artist that led to this candidate
    tier: str | None = None    # "rising" / "underground" / "deep_cut" (popularity mix quotas)

    @property
    def rank_score(self) -> float:
        return self.score * (0.75 + 0.25 * self.affinity)


def select_diverse(scored: list[Scored], limit: int, shares: dict[int, float],
                   family_share: dict[str, float] | None = None, tier_quota: dict[str, int] | None = None,
                   too_close: Callable[[Scored, Scored], bool] | None = None) -> list[Scored]:
    """<=2 per artist (<=1 in the top 10); fill across vibes in proportion to share. With `tier_quota` the
    first pass also caps each popularity tier at its quota (the backfill then tops up from any tier, so a
    short tier never shortens the list). `too_close(a, b)` rejects near-duplicate-sounding picks."""
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

    per_tier: dict[str, int] = {}

    def take(s: Scored, strict_tier: bool = False) -> bool:
        a = s.artist.lower()
        if strict_tier and tier_quota is not None and s.tier in tier_quota \
                and per_tier.get(s.tier, 0) >= tier_quota[s.tier]:
            return False
        if too_close is not None and any(too_close(s, c) for c in chosen):
            return False
        cap = 1 if len(chosen) < 10 else 2  # <=1 per artist in the top 10, <=2 overall
        if per_artist.get(a, 0) >= cap:
            return False
        if s.family and per_family.get(s.family, 0) >= family_cap(s.family):
            return False
        per_artist[a] = per_artist.get(a, 0) + 1
        per_tier[s.tier] = per_tier.get(s.tier, 0) + 1
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
            if s.vibe_id == v and id(s) not in taken_ids and take(s, strict_tier=True):
                taken_ids.add(id(s))
                n += 1
    if tier_quota is not None:
        # Short vibes: fill from other vibes while respecting the tier quotas, loosening family caps by up to
        # TIER_FAMILY_SLACK first (a small tier's few artists often share one family), and only then let
        # other tiers backfill.
        while len(chosen) < limit and slack <= TIER_FAMILY_SLACK:
            for s in scored:
                if len(chosen) >= limit:
                    break
                if id(s) not in taken_ids and take(s, strict_tier=True):
                    taken_ids.add(id(s))
            slack += 1
        slack = 0
    while len(chosen) < limit and slack <= limit:  # backfill short vibes/tiers, loosening family caps
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
