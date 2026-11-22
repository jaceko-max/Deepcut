"""Playlist vibe clustering (numpy k-means on a group-weighted embedding)."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from app.scoring.normalize import Norm

W_TEMPO, W_GROOVE, W_TONE, W_TIMBRE = 0.10, 0.20, 0.35, 0.35  # keep in sync with score.py
TEMPO_NORM = 0.908


def tempo_angle(bpm_folded: float) -> float:
    return 2 * math.pi * (math.log2(bpm_folded / 70.0) % 1.0)


def angle_to_bpm(theta: float) -> float:
    return 70.0 * 2 ** ((theta % (2 * math.pi)) / (2 * math.pi))


def embed(norms: list[Norm]) -> np.ndarray:
    rows = []
    for n in norms:
        th = tempo_angle(n.bpm_folded)
        t = np.array([math.cos(th), math.sin(th)]) * math.sqrt(W_TEMPO) / TEMPO_NORM
        rows.append(np.concatenate([
            t,
            n.groove * math.sqrt(W_GROOVE / len(n.groove)),
            n.tone * math.sqrt(W_TONE / len(n.tone)),
            n.timbre * math.sqrt(W_TIMBRE / len(n.timbre)),
        ]))
    return np.array(rows)


def kmeans(X: np.ndarray, k: int, seed: int = 42, n_init: int = 10, iters: int = 100):
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_init):
        centers = [X[rng.integers(len(X))]]
        for _ in range(1, k):
            d2 = np.min(((X[:, None, :] - np.array(centers)[None]) ** 2).sum(-1), axis=1)
            p = d2 / d2.sum() if d2.sum() > 0 else np.full(len(X), 1 / len(X))
            centers.append(X[rng.choice(len(X), p=p)])
        C = np.array(centers)
        labels = np.zeros(len(X), dtype=int)
        for _ in range(iters):
            d = ((X[:, None, :] - C[None]) ** 2).sum(-1)
            new = d.argmin(1)
            for j in range(k):
                if (new == j).any():
                    C[j] = X[new == j].mean(0)
            if (new == labels).all():
                labels = new
                break
            labels = new
        inertia = float(((X - C[labels]) ** 2).sum())
        if best is None or inertia < best[0]:
            best = (inertia, labels.copy(), C.copy())
    return best[1], best[2]


def silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    D = np.sqrt(((X[:, None, :] - X[None]) ** 2).sum(-1))
    s = []
    for i in range(len(X)):
        same = labels == labels[i]
        same[i] = False
        if not same.any():
            s.append(0.0)
            continue
        a = D[i, same].mean()
        b = min(D[i, labels == c].mean() for c in set(labels) if c != labels[i])
        s.append((b - a) / max(a, b, 1e-12))
    return float(np.mean(s))


def choose_labels(X: np.ndarray) -> np.ndarray:
    n = len(X)
    if n < 8:
        return np.zeros(n, dtype=int)
    min_size = max(3, math.ceil(0.15 * n))
    best_s, best_l = -1.0, None
    for k in (2, 3):
        labels, _ = kmeans(X, k)
        sizes = np.bincount(labels, minlength=k)
        if len(set(labels)) < k or sizes.min() < min_size:
            continue
        s = silhouette(X, labels)
        if s >= 0.25 and s > best_s:
            best_s, best_l = s, labels
    return best_l if best_l is not None else np.zeros(n, dtype=int)


@dataclass
class Vibe:
    id: int
    members: list[int]
    share: float
    bpm: float          # median folded BPM of members
    centroid: Norm


def centroid_of(norms: list[Norm]) -> Norm:
    th = np.array([tempo_angle(n.bpm_folded) for n in norms])
    mean_th = math.atan2(np.sin(th).mean(), np.cos(th).mean())
    bpm = angle_to_bpm(mean_th)
    raw = float(np.median([n.bpm_raw for n in norms]))  # unfolded: lets distance() spot half/double time
    return Norm(raw, bpm, np.mean([n.groove for n in norms], 0),
                np.mean([n.tone for n in norms], 0), np.mean([n.timbre for n in norms], 0))


def build_vibes(norms: list[Norm]) -> list[Vibe]:
    labels = choose_labels(embed(norms))
    groups = [np.flatnonzero(labels == c).tolist() for c in sorted(set(labels.tolist()))]
    groups.sort(key=len, reverse=True)
    vibes = []
    for i, g in enumerate(groups):
        ns = [norms[j] for j in g]
        vibes.append(Vibe(i, g, len(g) / len(norms), float(np.median([n.bpm_folded for n in ns])), centroid_of(ns)))
    return vibes
