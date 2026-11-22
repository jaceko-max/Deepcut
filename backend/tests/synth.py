"""Synthetic drum-loop audio for tests (no network, no fixtures)."""
from __future__ import annotations

import io
import wave

import numpy as np
from scipy.signal import butter, sosfilt

SR = 22050


def _kick(freq=55.0, dur=0.25, decay=0.12):
    t = np.arange(int(dur * SR)) / SR
    return np.sin(2 * np.pi * freq * t) * np.exp(-t / (decay / 4))


def _hat(rng, dur=0.04):
    n = int(dur * SR)
    noise = rng.standard_normal(n)
    sos = butter(4, 7000, btype="highpass", fs=SR, output="sos")
    return sosfilt(sos, noise) * np.exp(-np.arange(n) / SR / 0.012)


def _snare(rng, dur=0.12):
    n = int(dur * SR)
    t = np.arange(n) / SR
    return (rng.standard_normal(n) * 0.6 + np.sin(2 * np.pi * 190 * t)) * np.exp(-t / 0.04)


def _place(buf, sig, pos, gain):
    if pos < 0 or pos >= len(buf):
        return
    end = min(len(buf), pos + len(sig))
    buf[pos:end] += gain * sig[: end - pos]


def drum_loop(bpm=120.0, seconds=30.0, kick=1.0, hats="8th", offbeat_snare=False,
              clicks=False, noise_db=-50.0, seed=0, kick_freq=55.0):
    """hats: None | '8th' | '16th'. clicks: replace everything with a metronome click."""
    rng = np.random.default_rng(seed)
    n = int(seconds * SR)
    y = np.zeros(n)
    beat = 60.0 / bpm * SR
    nbeats = int(n / beat) + 1
    if clicks:
        click = np.zeros(80)
        click[:] = np.hanning(80) * np.sin(2 * np.pi * 1500 * np.arange(80) / SR)
        for b in range(nbeats):
            _place(y, click, int(b * beat), 1.0)
    else:
        k = _kick(freq=kick_freq)
        h = _hat(rng)
        s = _snare(rng)
        for b in range(nbeats):
            p = b * beat
            if kick:
                _place(y, k, int(p), kick)
            if hats in ("8th", "16th"):
                _place(y, h, int(p + beat / 2), 0.25)
            if hats == "16th":
                _place(y, h, int(p + beat / 4), 0.18)
                _place(y, h, int(p + 3 * beat / 4), 0.18)
            if hats:
                _place(y, h, int(p), 0.25)
            if offbeat_snare:
                _place(y, s, int(p + beat * 0.75), 0.5)
                _place(y, s, int(p + beat * 0.25 + 2 * beat), 0.4)
    y = y / max(np.abs(y).max(), 1e-9) * 0.8
    y += rng.standard_normal(n) * 10 ** (noise_db / 20)
    return y.astype(np.float32)


def to_wav_bytes(y: np.ndarray, sr: int = SR) -> bytes:
    pcm = (np.clip(y, -1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()
