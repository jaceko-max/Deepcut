"""Raw features -> normalized 0..1 vectors grouped for distance computation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# (lo, hi) linear bounds, clipped 0..1
BOUNDS = {
    "pulse_clarity": (0.1, 0.8),
    "onset_density": (0.5, 8.0),
    "syncopation": (0.0, 0.6),
    "centroid_log2": (np.log2(400), np.log2(4500)),
    "rolloff_log2": (np.log2(1000), np.log2(10000)),
    "bass_ratio": (0.0, 0.6),
    "sub_ratio": (0.0, 0.3),
    "rms_mean_db": (-35.0, -6.0),
    "dynamic_range_db": (2.0, 30.0),
    "flatness": (0.0, 0.4),
}
MFCC_SCALE_FLOOR = 1.0  # keeps near-identical playlists from exploding the z-score


def lin(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def _n(raw: dict, k: str) -> float:
    lo, hi = BOUNDS[k]
    return lin(raw[k], lo, hi)


@dataclass
class Norm:
    bpm_raw: float
    bpm_folded: float
    groove: np.ndarray  # pulse_clarity, onset_density, regularity, syncopation, pulse_stability
    tone: np.ndarray    # centroid, rolloff, bass, sub, rms, dynamics, flatness
    timbre: np.ndarray  # 24 robust-scaled MFCC stats
    style: np.ndarray | None = None  # L2-normalised sqrt of Discogs-EffNet style activations (None: no model)

    def summary(self, bpm: float | None = None) -> dict:
        g, t = self.groove, self.tone
        r = lambda v: round(float(v), 3)
        return {
            "bpm": round(float(self.bpm_raw if bpm is None else bpm), 1),
            "pulse_clarity": r(g[0]), "onset_density": r(g[1]), "regularity": r(g[2]),
            "syncopation": r(g[3]), "brightness": r(t[0]), "low_end": r(t[2]),
            "energy": r(t[4]), "dynamics": r(t[5]),
        }


class MfccScaler:
    """Robust z-score over the job's reference tracks: clip(0.5 + z/6, 0, 1)."""

    def __init__(self, median: np.ndarray, scale: np.ndarray):
        self.median, self.scale = median, scale

    @classmethod
    def fit(cls, raws: list[dict]) -> "MfccScaler":
        m = np.array([r["mfcc"] for r in raws], dtype=float)
        med = np.median(m, axis=0)
        mad = np.median(np.abs(m - med), axis=0)
        return cls(med, np.maximum(1.4826 * mad + 1e-6, MFCC_SCALE_FLOOR))

    def transform(self, mfcc) -> np.ndarray:
        z = (np.asarray(mfcc, dtype=float) - self.median) / self.scale
        return np.clip(0.5 + z / 6.0, 0.0, 1.0)


def normalize(raw: dict, scaler: MfccScaler) -> Norm:
    groove = np.array([_n(raw, "pulse_clarity"), _n(raw, "onset_density"),
                       float(np.clip(raw["regularity"], 0, 1)), _n(raw, "syncopation"),
                       float(np.clip(raw["pulse_stability"], 0, 1))])
    tone = np.array([_n(raw, "centroid_log2"), _n(raw, "rolloff_log2"), _n(raw, "bass_ratio"),
                     _n(raw, "sub_ratio"), _n(raw, "rms_mean_db"), _n(raw, "dynamic_range_db"),
                     _n(raw, "flatness")])
    return Norm(raw["bpm_raw"], raw["bpm_folded"], groove, tone, scaler.transform(raw["mfcc"]), style_vector(raw.get("style")))


def style_vector(act) -> np.ndarray | None:
    """Mean style activations (400 Discogs styles, sigmoid) -> unit vector of their square roots, which
    lets the many small activations count (a Hellinger-style comparison of two style profiles)."""
    if not act:
        return None
    v = np.sqrt(np.maximum(np.asarray(act, dtype=float), 0.0))
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else None
