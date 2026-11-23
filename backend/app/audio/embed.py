"""Learned music "style" profile: Essentia's Discogs-EffNet (ONNX) run with onnxruntime on CPU.

Model: discogs-effnet-bsdynamic-1 by the Music Technology Group, Universitat Pompeu Fabra
(https://essentia.upf.edu/models/), an EfficientNet-B0 trained on Discogs editorial metadata to predict
400 music styles. License: Creative Commons BY-NC-SA 4.0 (non-commercial use only; the LICENSE file in
the model directory is headed "Attribution-NonCommercial-NoDerivatives 4.0" but links the BY-NC-SA 4.0
legal code). The model file is used unmodified and downloaded at Docker build time, never committed.
See README "Attribution".

The model's input is Essentia's TensorflowInputMusiCNN log-mel spectrogram, re-implemented here in numpy
so the app does not need Essentia (checked against Essentia 2.1b6: same frame count, max abs difference
4e-4 on a log scale of 0-6): 16 kHz mono, 512-sample frames (symmetric Hann, |FFT|), 256-sample hop with
the first frame centred on sample 0 (Essentia FrameCutter, startFromZero=false), 96 Slaney-mel bands
0-8000 Hz with linear triangles normalised by the theoretical triangle area ("unit_tri") applied to the
POWER spectrum, then log10(1 + 10000 * x). Up to 6 patches of 128 frames (~2 s) spread over the clip go
through the network; the 400 style activations are averaged over patches.

We keep the style activations rather than the 1280-d penultimate embedding: on the accuracy benchmark
both separate a playlist's own tracks from other playlists' equally well (AUC 0.95) and tie on ranking
held-out playlist artists (style: R@20 0.24, audio AUC 0.56; embedding: 0.21, 0.59, same clips), and the
style profile is a third of the cache size. 6 patches scored as well as 14 at less than half the cost.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import tempfile
import urllib.request
from functools import lru_cache

import numpy as np

log = logging.getLogger("deepcut.embed")

SR = 16000
FRAME = 512
HOP = 256
BANDS = 96
PATCH = 128
PATCH_HOP = 124          # ~2 s between patch starts (Essentia's default 62 overlaps patches by half)
MAX_PATCHES = 6          # spread evenly over the clip
N_STYLES = 400
MODEL_FILE = "discogs-effnet-bsdynamic-1.onnx"
MODEL_SHA256 = "a280825b334797cf677939db8cd5762c0392aedd0ca6415dbc1cd083f045e43c"
MODEL_URL = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/" + MODEL_FILE
DEFAULT_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                  ".cache", "models", MODEL_FILE)  # backend/.cache is git-ignored


def _hz2mel(hz: float) -> float:  # Slaney (Auditory Toolbox), as essentiamath.h hz2melSlaney
    if hz < 1000.0:
        return hz * 3.0 / 200.0
    return 15.0 + math.log(hz / 1000.0) / (math.log(6.4) / 27.0)


def _mel2hz(mel: float) -> float:
    if mel < 15.0:
        return mel * 200.0 / 3.0
    return 1000.0 * math.exp((mel - 15.0) * (math.log(6.4) / 27.0))


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    """(BANDS, FRAME//2+1) weights, as Essentia MelBands(slaneyMel, weighting=linear, normalize=unit_tri)."""
    n_bins = FRAME // 2 + 1
    lo_m, hi_m = _hz2mel(0.0), _hz2mel(SR / 2)
    step = (hi_m - lo_m) / (BANDS + 1)
    edges = [_mel2hz(lo_m + i * step) for i in range(BANDS + 2)]
    scale = (SR / 2) / (n_bins - 1)
    fb = np.zeros((BANDS, n_bins), dtype=np.float64)
    for i in range(BANDS):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        for j in range(math.ceil(lo / scale), math.floor(hi / scale) + 1):
            f = j * scale
            fb[i, j] = (f - lo) / (c - lo) if f < c else (hi - f) / (hi - c)
        fb[i] /= (hi - lo) / 2.0
    return fb.astype(np.float32)


@lru_cache(maxsize=1)
def _hann() -> np.ndarray:
    i = np.arange(FRAME)
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * i / (FRAME - 1))).astype(np.float32)


def melspectrogram(y: np.ndarray) -> np.ndarray:
    """y: mono float32 at 16 kHz -> (frames, 96) float32 log-mel, Essentia TensorflowInputMusiCNN."""
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n == 0:
        return np.zeros((0, BANDS), dtype=np.float32)
    # Essentia FrameCutter(startFromZero=false): frame k is centred on sample k*HOP; the last frame is the
    # first one whose centre is at or past the end of the signal.
    n_frames = -(-n // HOP) + 1
    padded = np.zeros(FRAME // 2 + n_frames * HOP + FRAME, dtype=np.float32)
    padded[FRAME // 2: FRAME // 2 + n] = y
    idx = np.arange(FRAME)[None, :] + HOP * np.arange(n_frames)[:, None]
    frames = padded[idx] * _hann()
    power = np.abs(np.fft.rfft(frames, axis=1)).astype(np.float32) ** 2
    mel = power @ mel_filterbank().T
    return np.log10(1.0 + 10000.0 * mel).astype(np.float32)


def patches(mel: np.ndarray, hop: int = PATCH_HOP, max_patches: int = MAX_PATCHES) -> np.ndarray:
    """(n, 128, 96) model input; clips shorter than one patch are padded by repeating them."""
    if len(mel) == 0:
        return np.zeros((0, PATCH, BANDS), dtype=np.float32)
    if len(mel) < PATCH:
        mel = np.concatenate([mel] * (PATCH // len(mel) + 1))[:PATCH]
    starts = list(range(0, len(mel) - PATCH + 1, max(1, hop)))
    if len(starts) > max_patches:
        starts = [starts[round(i * (len(starts) - 1) / (max_patches - 1))] for i in range(max_patches)]
    return np.stack([mel[s:s + PATCH] for s in starts]).astype(np.float32)


def model_path() -> str:
    return os.environ.get("EMBEDDING_MODEL_PATH") or DEFAULT_MODEL_PATH


def enabled() -> bool:
    """EMBEDDINGS=onnx (default) uses the model when onnxruntime and the model file are present;
    EMBEDDINGS=off disables it (scores then use the hand-built DSP features only)."""
    if os.environ.get("EMBEDDINGS", "onnx").strip().lower() in ("off", "0", "false", "none"):
        return False
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return False
    return os.path.exists(model_path())


_session = None


def _get_session():
    """One onnxruntime session per (worker) process, single-threaded: the process pool is the parallelism."""
    global _session
    if _session is None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.enable_cpu_mem_arena = False  # keeps RSS low on a 512 MB instance
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _session = ort.InferenceSession(model_path(), sess_options=opts, providers=["CPUExecutionProvider"])
    return _session


def style_from_samples(y16k: np.ndarray) -> list[float]:
    """Mean Discogs style activations (400 sigmoid outputs) over up to MAX_PATCHES patches."""
    x = patches(melspectrogram(y16k))
    if len(x) == 0:
        raise ValueError("clip too short for the style model")
    act = _get_session().run(["activations"], {"melspectrogram": x})[0]
    return [round(float(a), 5) for a in np.asarray(act, dtype=np.float64).mean(axis=0)]


def download(dest: str | None = None) -> str:
    """Fetch the pinned model and verify its sha256 (used by the Dockerfile at build time)."""
    dest = dest or model_path()
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    h = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(MODEL_URL, timeout=120) as r:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != MODEL_SHA256:
            raise RuntimeError(f"model checksum mismatch: {h.hexdigest()} != {MODEL_SHA256}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


if __name__ == "__main__":  # python -m app.audio.embed download [dest]
    if len(sys.argv) >= 2 and sys.argv[1] == "download":
        print(download(sys.argv[2] if len(sys.argv) > 2 else None))
    else:
        print("usage: python -m app.audio.embed download [dest]")
        sys.exit(2)
