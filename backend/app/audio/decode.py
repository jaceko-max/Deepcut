"""MP3/WAV decode to mono float32 @ 22050 Hz. miniaudio first, ffmpeg fallback."""
from __future__ import annotations

import subprocess

import numpy as np

SR = 22050


class AudioError(Exception):
    pass


def _decode_miniaudio(data: bytes, sr: int = SR) -> np.ndarray:
    import miniaudio

    d = miniaudio.decode(data, output_format=miniaudio.SampleFormat.FLOAT32, nchannels=1, sample_rate=sr)
    return np.asarray(d.samples, dtype=np.float32)


def _decode_ffmpeg(data: bytes, sr: int = SR) -> np.ndarray:
    p = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(sr), "pipe:1"],
        input=data, capture_output=True, timeout=30,
    )
    if p.returncode != 0 or not p.stdout:
        raise AudioError("ffmpeg decode failed")
    return np.frombuffer(p.stdout, dtype=np.float32).copy()


def decode_audio(data: bytes, sr: int = SR) -> np.ndarray:
    try:
        return _decode_miniaudio(data, sr)
    except Exception as first:  # miniaudio raises DecodeError on unknown formats
        try:
            return _decode_ffmpeg(data, sr)
        except Exception:
            raise AudioError(f"could not decode audio: {first}") from first
