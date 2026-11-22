"""numpy/scipy DSP primitives (no librosa)."""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import fft as sfft
from scipy.ndimage import uniform_filter1d

SR = 22050
N_FFT = 2048
HOP = 512
FPS = SR / HOP  # 43.07


def frame_signal(y: np.ndarray, n: int = N_FFT, hop: int = HOP) -> np.ndarray:
    if len(y) < n:
        y = np.pad(y, (0, n - len(y)))
    return sliding_window_view(y, n)[::hop]


def stft_power(y: np.ndarray) -> np.ndarray:
    """Power spectrogram, shape (frames, 1 + N_FFT/2)."""
    win = np.hanning(N_FFT + 1)[:-1].astype(np.float32)
    spec = sfft.rfft(frame_signal(y) * win, axis=1)
    return (spec.real ** 2 + spec.imag ** 2).astype(np.float32)


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(n_mels: int = 64, fmin: float = 30.0, fmax: float = 11025.0) -> np.ndarray:
    """HTK triangular filters, area-normalized. Shape (n_mels, 1 + N_FFT/2)."""
    freqs = np.linspace(0, SR / 2, 1 + N_FFT // 2)
    pts = mel_to_hz(np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2))
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ctr, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / (ctr - lo)
        down = (hi - freqs) / (hi - ctr)
        fb[i] = np.maximum(0, np.minimum(up, down)) * (2.0 / (hi - lo))
    return fb.astype(np.float32)


_FB = None


def logmel_from_power(P: np.ndarray) -> np.ndarray:
    global _FB
    if _FB is None:
        _FB = mel_filterbank()
    mel = P @ _FB.T  # (frames, 64)
    return 10.0 * np.log10(mel + 1e-10)


def mfcc_from_logmel(logmel: np.ndarray, n: int = 12) -> np.ndarray:
    """Orthonormal DCT-II over bands, coefficients 1..n. Shape (frames, n)."""
    return sfft.dct(logmel, type=2, norm="ortho", axis=1)[:, 1 : n + 1]


def onset_envelope(logmel: np.ndarray) -> np.ndarray:
    """Spectral flux on log-mel, detrended by a 0.5 s moving average, scaled to max 1."""
    flux = np.maximum(0.0, np.diff(logmel, axis=0)).mean(axis=1)
    env = np.concatenate([[0.0], flux])
    env = env - uniform_filter1d(env, size=max(3, int(round(0.5 * FPS))), mode="nearest")
    env = np.maximum(env, 0.0)
    m = env.max()
    return env / m if m > 0 else env


def autocorr(x: np.ndarray) -> np.ndarray:
    """Biased autocorrelation via FFT, normalized by ac[0] (zero-mean input)."""
    x = x - x.mean()
    n = len(x)
    nfft = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(x, nfft)
    ac = np.fft.irfft(f * np.conj(f), nfft)[:n]
    return ac / ac[0] if ac[0] > 1e-12 else np.zeros(n)


def _ac_at(ac: np.ndarray, lags: np.ndarray) -> np.ndarray:
    return np.interp(lags, np.arange(len(ac)), ac, right=0.0)


def estimate_tempo(env: np.ndarray, fps: float = FPS):
    """Returns (bpm_raw, pulse_clarity, ac)."""
    ac = autocorr(env)
    bpms = np.arange(40.0, 220.0 + 1e-9, 0.25)
    lags = fps * 60.0 / bpms
    score = _ac_at(ac, lags) + 0.5 * _ac_at(ac, 2 * lags) + 0.5 * _ac_at(ac, lags / 2)
    prior = np.exp(-0.5 * ((np.log2(bpms) - np.log2(120.0)) / 1.0) ** 2)
    s = score * prior
    i = int(np.argmax(s))
    if 0 < i < len(s) - 1:  # parabolic refine on the (log-spaced-ish) grid
        a, b, c = s[i - 1], s[i], s[i + 1]
        den = a - 2 * b + c
        off = 0.5 * (a - c) / den if den != 0 else 0.0
        bpm = bpms[i] + float(np.clip(off, -1, 1)) * 0.25
    else:
        bpm = float(bpms[i])
    lag = fps * 60.0 / bpm
    clarity = float(np.clip(_ac_at(ac, np.array([lag]))[0], 0.0, 1.0))
    return float(bpm), clarity, ac


def fold_bpm(bpm: float, lo: float = 70.0, hi: float = 140.0) -> float:
    while bpm >= hi:
        bpm /= 2.0
    while bpm < lo:
        bpm *= 2.0
    return bpm


def octave_distance(a: float, b: float) -> float:
    """Octave-circular tempo distance in [0, 0.5]; 60 vs 120 -> 0."""
    c = abs(np.log2(a / b)) % 1.0
    return float(min(c, 1.0 - c))


def pulse_stability(env: np.ndarray, lag: float, fps: float = FPS) -> float:
    win, hop = int(4 * fps), int(2 * fps)
    vals = []
    for s in range(0, max(1, len(env) - win + 1), hop):
        seg = env[s : s + win]
        if len(seg) < win:
            break
        ac = autocorr(seg)
        vals.append(float(np.clip(_ac_at(ac, np.array([lag]))[0], -1, 1)))
    return float(np.clip(np.mean(vals), 0, 1)) if vals else 0.0


def beat_track(env: np.ndarray, bpm: float, fps: float = FPS, tightness: float = 100.0) -> np.ndarray:
    """Ellis dynamic-programming beat tracker. Returns beat frame indices."""
    period = fps * 60.0 / bpm
    n = len(env)
    if n < 8 or period < 2:
        return np.array([], dtype=int)
    local = env / (env.std() + 1e-9)
    window = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    txcost = -tightness * (np.log(-window / period)) ** 2
    cum = local.copy()
    back = np.full(n, -1, dtype=int)
    for i in range(n):
        idx = i + window
        ok = idx >= 0
        if not ok.any():
            continue
        cand = cum[idx[ok]] + txcost[ok]
        j = int(np.argmax(cand))
        if cand[j] > 0:
            cum[i] = local[i] + cand[j]
            back[i] = idx[ok][j]
    # pick the end: last local maximum of cum above half the median local-max value
    lm = np.flatnonzero((cum[1:-1] > cum[:-2]) & (cum[1:-1] >= cum[2:])) + 1
    if len(lm) == 0:
        return np.array([], dtype=int)
    thresh = 0.5 * np.median(cum[lm])
    last = lm[cum[lm] >= thresh][-1]
    beats = [int(last)]
    while back[beats[-1]] >= 0:
        beats.append(int(back[beats[-1]]))
    return np.array(beats[::-1], dtype=int)


def pick_onsets(env: np.ndarray) -> np.ndarray:
    """Local maxima within +-3 frames, > mean+0.5*std, at least 3 frames apart."""
    n = len(env)
    if n < 8:
        return np.array([], dtype=int)
    thr = env.mean() + 0.5 * env.std()
    w = sliding_window_view(np.pad(env, 3, mode="edge"), 7)
    cand = np.flatnonzero((env >= w.max(axis=1)) & (env > thr))
    out: list[int] = []
    for c in cand:
        if not out or c - out[-1] >= 3:
            out.append(int(c))
        elif env[c] > env[out[-1]]:
            out[-1] = int(c)
    return np.array(out, dtype=int)
