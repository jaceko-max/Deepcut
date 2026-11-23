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

Mood heads: the same EffNet pass also returns the 1280-d embedding of every patch, which feeds Essentia's
published classification/regression heads for Discogs-EffNet (https://essentia.upf.edu/models/
classification-heads/, same MTG license; each a ~0.5 MB dense network, 1280 -> 100 -> 1-2 outputs, plus
the 56-tag MTG-Jamendo mood/theme head). As in Essentia, heads run on each patch embedding and their
outputs are averaged over patches. All 12 heads add ~4 MB RSS and ~2 ms per clip. Essentia publishes no
arousal/valence regressor for Discogs-EffNet (DEAM/emoMusic/MuSe heads exist only for VGGish and
MusiCNN embeddings), so arousal/valence are derived from the mood heads in app/scoring/mood.py.
"""
from __future__ import annotations

import hashlib
import logging
import math
import os
import sys
import tempfile
import urllib.request
from dataclasses import dataclass
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

HEADS_URL = "https://essentia.upf.edu/models/classification-heads/"


@dataclass(frozen=True)
class Head:
    """One Essentia head on Discogs-EffNet embeddings (input "embeddings" (n, 1280), output "activations")."""
    name: str      # key in the cached "mood" dict
    path: str      # <dir>/<file> under HEADS_URL
    sha256: str
    index: int     # output column kept (the positive class for 2-class softmax heads)

    @property
    def file(self) -> str:
        return self.path.rsplit("/", 1)[1]


# Output index = position of the named class in the head's JSON metadata ("classes").
HEADS = (
    Head("relaxed", "mood_relaxed/mood_relaxed-discogs-effnet-1.onnx",
         "8ba6515a1e5943a72b3b475e3a25fc7a2ff04142c3eaa6aa0716fca371efdfff", 1),  # [non_relaxed, relaxed]
    Head("sad", "mood_sad/mood_sad-discogs-effnet-1.onnx",
         "1a50d11c23181bdfdeabc7d6032a2dad829dfc19c3776d2ee57ef1724cb08805", 1),  # [non_sad, sad]
    Head("happy", "mood_happy/mood_happy-discogs-effnet-1.onnx",
         "0ca322819ef137b4b87e9866bffe7370a630e6f1165184ec106326cef6f81e06", 0),  # [happy, non_happy]
    Head("aggressive", "mood_aggressive/mood_aggressive-discogs-effnet-1.onnx",
         "de36550b5d1660791ad732ed6de6ebfdc3e65dcf50b928b2578ddf103dbfb400", 0),  # [aggressive, not_aggressive]
    Head("party", "mood_party/mood_party-discogs-effnet-1.onnx",
         "c50ac2106ec2f209dd04ad48756582df0e3f3512235310d1a4a3fcc453745f04", 1),  # [non_party, party]
    Head("acoustic", "mood_acoustic/mood_acoustic-discogs-effnet-1.onnx",
         "56b02abf772b9c1cf528e4d6521d6e09515cedccce47895ee5de8de33fbdf848", 0),  # [acoustic, non_acoustic]
    Head("electronic", "mood_electronic/mood_electronic-discogs-effnet-1.onnx",
         "4cc09140d5b078cf39e55975a23cba813e6bd4014092b78e8c415fa32a382149", 0),  # [electronic, non_electronic]
    Head("danceable", "danceability/danceability-discogs-effnet-1.onnx",
         "9ce9b8c44f1dd5df5ffc124e5d41d67acf254232c1b90c7e057e079ab7cead73", 0),  # [danceable, not_danceable]
    Head("voice", "voice_instrumental/voice_instrumental-discogs-effnet-1.onnx",
         "20155e4c439714b0c45c08644b73c8e12d9dccb173bd4ab9934bf1e5aee837ca", 1),  # [instrumental, voice]
    Head("bright", "timbre/timbre-discogs-effnet-1.onnx",
         "3fbeacb0006b6ca3445b8696dcd5d89917a608ab29074f935dfa68816b23b44e", 0),  # [bright, dark]
    Head("approachable", "approachability/approachability_regression-discogs-effnet-1.onnx",
         "f783b17ee994394f30f27d05ccd3fb845e589d107802e55e0b9cf1ea041ec894", 0),  # regression
    Head("engaging", "engagement/engagement_regression-discogs-effnet-1.onnx",
         "9551a6c162c9803aef5ce4e4d29a2e4fae99b5bfdc95dae9dd8fe5335e0fb567", 0),  # regression
)
THEME_HEAD = Head("theme", "mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.onnx",
                  "7d6270acaa5f4bba4b115a0d6849aca05ed6bd153dcb6d9da4f6ab9f99ef10ff", -1)  # 56 sigmoids
THEMES = ("action", "adventure", "advertising", "background", "ballad", "calm", "children", "christmas",
          "commercial", "cool", "corporate", "dark", "deep", "documentary", "drama", "dramatic", "dream",
          "emotional", "energetic", "epic", "fast", "film", "fun", "funny", "game", "groovy", "happy", "heavy",
          "holiday", "hopeful", "inspiring", "love", "meditative", "melancholic", "melodic", "motivational",
          "movie", "nature", "party", "positive", "powerful", "relaxing", "retro", "romantic", "sad", "sexy",
          "slow", "soft", "soundscape", "space", "sport", "summer", "trailer", "travel", "upbeat", "uplifting")
ALL_HEADS = HEADS + (THEME_HEAD,)


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


def heads_dir() -> str:
    """Mood heads live next to the EffNet model unless MOOD_MODELS_DIR says otherwise."""
    return os.environ.get("MOOD_MODELS_DIR") or os.path.dirname(model_path())


def head_path(h: Head) -> str:
    return os.path.join(heads_dir(), h.file)


def moods_enabled() -> bool:
    """Mood heads need the EffNet model plus every head file (a partial set would change the vector)."""
    return enabled() and all(os.path.exists(head_path(h)) for h in ALL_HEADS)


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
_head_sessions: dict[str, object] = {}


def _session_for(path: str):
    """One onnxruntime session per model per (worker) process, single-threaded: the process pool is the
    parallelism."""
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False  # keeps RSS low on a 512 MB instance
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


def _get_session():
    global _session
    if _session is None:
        _session = _session_for(model_path())
    return _session


def _get_head(h: Head):
    if h.name not in _head_sessions:
        _head_sessions[h.name] = _session_for(head_path(h))
    return _head_sessions[h.name]


def run_heads(emb: np.ndarray) -> tuple[dict[str, float], list[float]]:
    """Per-patch embeddings (n, 1280) -> ({head name: mean output}, 56 mean mood/theme activations).
    Heads score each patch and the outputs are averaged, as Essentia's predictors do."""
    emb = np.asarray(emb, dtype=np.float32)
    if emb.ndim != 2 or emb.shape[1] != 1280 or len(emb) == 0:
        raise ValueError(f"expected (n, 1280) embeddings, got {emb.shape}")
    mood = {}
    for h in HEADS:
        out = np.asarray(_get_head(h).run(["activations"], {"embeddings": emb})[0], dtype=np.float64)
        mood[h.name] = round(float(out[:, h.index].mean()), 5)
    theme = np.asarray(_get_head(THEME_HEAD).run(["activations"], {"embeddings": emb})[0], dtype=np.float64)
    return mood, [round(float(v), 5) for v in theme.mean(axis=0)]


def analyze_samples(y16k: np.ndarray, with_moods: bool | None = None) -> dict:
    """One EffNet pass -> {"style": 400 mean style activations, and with the heads available
    "mood": {name: value} and "theme": 56 mood/theme activations}."""
    x = patches(melspectrogram(y16k))
    if len(x) == 0:
        raise ValueError("clip too short for the style model")
    want = moods_enabled() if with_moods is None else with_moods
    names = ["activations", "embeddings"] if want else ["activations"]
    res = _get_session().run(names, {"melspectrogram": x})
    out = {"style": [round(float(a), 5) for a in np.asarray(res[0], dtype=np.float64).mean(axis=0)]}
    if want:
        out["mood"], out["theme"] = run_heads(res[1])
    return out


def style_from_samples(y16k: np.ndarray) -> list[float]:
    """Mean Discogs style activations (400 sigmoid outputs) over up to MAX_PATCHES patches."""
    return analyze_samples(y16k, with_moods=False)["style"]


def _fetch(url: str, sha256: str, dest: str) -> str:
    """Download `url` to `dest` (atomically) and verify its sha256; an existing verified file is kept."""
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    if os.path.exists(dest):
        with open(dest, "rb") as f:
            if hashlib.sha256(f.read()).hexdigest() == sha256:
                return dest
    h = hashlib.sha256()
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest) or ".")
    try:
        with os.fdopen(fd, "wb") as f, urllib.request.urlopen(url, timeout=120) as r:
            while chunk := r.read(1 << 20):
                h.update(chunk)
                f.write(chunk)
        if h.hexdigest() != sha256:
            raise RuntimeError(f"checksum mismatch for {url}: {h.hexdigest()} != {sha256}")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return dest


def download(dest: str | None = None) -> list[str]:
    """Fetch the pinned EffNet model to `dest` and the pinned mood heads next to it, verifying every
    sha256 (used by the Dockerfile at build time and for local setup)."""
    dest = dest or model_path()
    out = [_fetch(MODEL_URL, MODEL_SHA256, dest)]
    hdir = os.environ.get("MOOD_MODELS_DIR") or os.path.dirname(dest)
    for h in ALL_HEADS:
        out.append(_fetch(HEADS_URL + h.path, h.sha256, os.path.join(hdir, h.file)))
    return out


if __name__ == "__main__":  # python -m app.audio.embed download [dest]   (also runs as a plain script)
    if len(sys.argv) >= 2 and sys.argv[1] == "download":
        print("\n".join(download(sys.argv[2] if len(sys.argv) > 2 else None)))
    else:
        print("usage: python -m app.audio.embed download [dest]")
        sys.exit(2)
