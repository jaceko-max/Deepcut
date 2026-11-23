"""Mood vectors from the Essentia mood heads (app/audio/embed.py), mood distance and mood words.

A clip's mood vector is the z-score of each of the 12 head outputs against a reference population (2,296
benchmark clips across 8 very different playlists: R&B, lo-fi, rap, indie, jazz, metal, house, country),
clipped to +-3 and scaled so the 12 heads together weigh 1, plus two derived axes at weight 0.5:
arousal (aggressive, party, engaging, not relaxed) and valence (happy, not sad). Essentia publishes no
arousal/valence regressor for Discogs-EffNet embeddings, so these are derived from the heads.

Design choices were made on the accuracy benchmark (README "Accuracy"), re-scoring the same traced clips:
  * Euclidean distance on z-scores beat cosine (held-out-artist audio AUC 0.668 vs 0.632 mood-only).
  * Adding the 56 MTG-Jamendo mood/theme activations to the vector hurt (0.639 at weight 0.5, 0.603 at
    1.0): they are noisy (published ROC-AUC 0.76) and are used only for the human-readable mood words.
"""
from __future__ import annotations

import math

import numpy as np

# Order = app.audio.embed.HEADS. Means/stds of the head outputs over the reference population.
MOOD_KEYS = ("relaxed", "sad", "happy", "aggressive", "party", "acoustic", "electronic", "danceable", "voice",
             "bright", "approachable", "engaging")
MOOD_MEAN = np.array([0.5718, 0.3705, 0.2873, 0.2267, 0.3938, 0.3087, 0.4065, 0.6709, 0.7054, 0.4921, 0.6055, 0.696])
MOOD_STD = np.array([0.3557, 0.3205, 0.2521, 0.3342, 0.3745, 0.3415, 0.3824, 0.3582, 0.3569, 0.0357, 0.2587, 0.2492])
THEME_MEAN = np.array([
    0.015, 0.0129, 0.0285, 0.0312, 0.0294, 0.0133, 0.0053, 0.0212, 0.0098, 0.0167, 0.0245, 0.0352, 0.045, 0.0149,
    0.011, 0.0067, 0.0202, 0.0278, 0.1077, 0.0271, 0.0052, 0.0271, 0.0146, 0.0214, 0.0042, 0.007, 0.056, 0.0259,
    0.0082, 0.0119, 0.0259, 0.0855, 0.02, 0.0185, 0.0592, 0.0333, 0.0103, 0.0063, 0.016, 0.019, 0.0065, 0.0374,
    0.0062, 0.032, 0.0301, 0.0063, 0.0129, 0.015, 0.0074, 0.0167, 0.0145, 0.0357, 0.002, 0.0051, 0.0171, 0.0249])
THEME_STD = np.array([
    0.0192, 0.0133, 0.0347, 0.0293, 0.0434, 0.0179, 0.0052, 0.0301, 0.0107, 0.0104, 0.0281, 0.0341, 0.089, 0.015,
    0.0112, 0.006, 0.0193, 0.026, 0.0843, 0.0483, 0.0066, 0.0211, 0.0107, 0.0319, 0.0035, 0.0049, 0.0538, 0.0772,
    0.0077, 0.0134, 0.0256, 0.0774, 0.0311, 0.0179, 0.0456, 0.0395, 0.0071, 0.0072, 0.0217, 0.022, 0.0121, 0.0425,
    0.0107, 0.0277, 0.0276, 0.0066, 0.0148, 0.0171, 0.0112, 0.0226, 0.0177, 0.0788, 0.0025, 0.004, 0.0174, 0.0281])
Z_CLIP = 3.0
W_AXES = 0.5          # weight of the derived arousal/valence pair relative to the 12 heads
MOOD_SCALE = 1.2      # Euclidean mood distance at which the mood group saturates. Benchmark sweep: 0.6-1.2 within
                      # 0.01 AUC; at 0.8 45% of a mixed playlist's own tracks saturated, so mood stopped ranking
_I = {k: i for i, k in enumerate(MOOD_KEYS)}


def mood_z(mood: dict | None) -> np.ndarray | None:
    """Head outputs -> clipped population z-scores (MOOD_KEYS order); None if missing or incomplete."""
    if not mood or any(k not in mood for k in MOOD_KEYS):
        return None
    x = np.array([float(mood[k]) for k in MOOD_KEYS])
    return np.clip((x - MOOD_MEAN) / MOOD_STD, -Z_CLIP, Z_CLIP)


def theme_z(theme) -> np.ndarray | None:
    if theme is None or len(theme) != len(THEME_MEAN):
        return None
    return np.clip((np.asarray(theme, dtype=float) - THEME_MEAN) / THEME_STD, -Z_CLIP, Z_CLIP)


def arousal_valence(z: np.ndarray) -> tuple[float, float]:
    """Derived axes in z units: arousal = mean(aggressive, party, engaging, -relaxed), valence = mean(happy, -sad)."""
    ar = (z[_I["aggressive"]] + z[_I["party"]] + z[_I["engaging"]] - z[_I["relaxed"]]) / 4
    va = (z[_I["happy"]] - z[_I["sad"]]) / 2
    return float(ar), float(va)


def mood_vector(mood: dict | None) -> np.ndarray | None:
    """14-d vector: 12 head z-scores / sqrt(12), then (arousal, valence) * W_AXES / sqrt(2)."""
    z = mood_z(mood)
    if z is None:
        return None
    ar, va = arousal_valence(z)
    return np.concatenate([z / math.sqrt(len(z)), np.array([ar, va]) * W_AXES / math.sqrt(2)])


def mood_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0..1: Euclidean distance of two mood vectors, saturating at MOOD_SCALE."""
    return min(1.0, float(np.linalg.norm(a - b)) / MOOD_SCALE)


def mood_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of two mood vectors (benchmark metric; vectors are population-centred z-scores)."""
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(a @ b) / den if den > 0 else 0.0


# -- mood words ------------------------------------------------------------------------------------
# Each word scores the mean of a few z-scores ("h:" = mood head, "t:" = MTG-Jamendo theme, "-" negates).
# Words in the same group describe the same thing, so a label uses at most one of them.
WORDS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("late-night", "night", ("h:relaxed", "-h:happy", "t:dark", "t:deep", "t:sexy")),
    ("mellow", "calm", ("h:relaxed", "-h:aggressive", "t:soft", "t:calm")),
    ("chill", "calm", ("h:relaxed", "t:relaxing", "t:background", "-h:engaging")),
    ("sensual", "love", ("t:sexy", "t:romantic", "t:love")),
    ("dreamy", "dream", ("t:dream", "t:space", "h:relaxed")),
    ("melancholic", "sad", ("h:sad", "t:melancholic", "t:sad", "-h:happy")),
    ("moody", "night", ("t:dark", "h:sad", "-h:happy", "-h:bright")),
    ("upbeat", "happy", ("h:happy", "t:upbeat", "t:positive", "t:happy")),
    ("uplifting", "happy", ("t:uplifting", "t:hopeful", "t:inspiring")),
    ("party", "party", ("h:party", "t:party", "h:danceable")),
    ("groovy", "party", ("h:danceable", "t:groovy", "t:cool")),
    ("hard-hitting", "hard", ("h:aggressive", "t:heavy", "t:powerful")),
    ("energetic", "hard", ("h:engaging", "t:energetic", "t:fast", "-h:relaxed")),
    ("epic", "hard", ("t:epic", "t:dramatic", "t:trailer")),
    ("acoustic", "texture", ("h:acoustic", "-h:electronic")),
    ("electronic", "texture", ("h:electronic", "-h:acoustic")),
    ("instrumental", "voice", ("-h:voice",)),
)


def _theme_index() -> dict[str, int]:
    from app.audio.embed import THEMES
    return {t: i for i, t in enumerate(THEMES)}


_T = _theme_index()


def word_scores(z: np.ndarray, tz: np.ndarray | None) -> dict[str, float]:
    """Mean z-score of each word's ingredients (theme ingredients are skipped without theme data)."""
    out = {}
    for word, _, parts in WORDS:
        vals = []
        for p in parts:
            sign = -1.0 if p.startswith("-") else 1.0
            kind, name = p.lstrip("-").split(":")
            if kind == "h":
                vals.append(sign * z[_I[name]])
            elif tz is not None:
                vals.append(sign * tz[_T[name]])
        if vals:
            out[word] = float(np.mean(vals))
    return out


def mood_words(z: np.ndarray | None, tz: np.ndarray | None, n: int = 3, ref: dict[str, float] | None = None,
               avoid: set[str] = frozenset(), min_score: float = 0.2) -> list[str]:
    """Up to `n` words, strongest first, one per group. With `ref` (another word-score dict, e.g. the whole
    playlist) half of each score is the difference from it, so sibling vibes read differently; words in
    `avoid` (already used by a sibling) are skipped while alternatives remain."""
    if z is None:
        return []
    sc = word_scores(z, tz)
    if ref:
        sc = {w: 0.5 * s + 0.5 * (s - ref.get(w, 0.0)) for w, s in sc.items()}
    group = {w: g for w, g, _ in WORDS}
    order = sorted(sc, key=lambda w: (w in avoid, -sc[w]))
    out, used = [], set()
    for w in order:
        if sc[w] < min_score or group[w] in used:
            continue
        out.append(w)
        used.add(group[w])
        if len(out) >= n:
            break
    if not out:  # nothing stands out: say the single strongest trait rather than nothing
        out = [max(sc, key=sc.get)] if sc else []
    return out
