"""Discogs-EffNet style profile: numpy preprocessing (shape, determinism, Essentia conventions),
style distance, and the optional ONNX model (skipped when it has not been downloaded)."""
import math
import os

import numpy as np
import pytest

from app.audio import embed as E
from app.audio.features import extract_features
from app.scoring.normalize import MfccScaler, normalize, style_vector
from app.scoring.score import STYLE_SCALE, W_STYLE, distance, style_distance
from app.scoring.vibes import centroid_of
from tests.synth import drum_loop, to_wav_bytes


def tone_clip(seconds=30.0, f=440.0, sr=E.SR, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * sr)) / sr
    return (0.4 * np.sin(2 * np.pi * f * t) + 0.01 * rng.standard_normal(len(t))).astype(np.float32)


def test_mel_filterbank_matches_essentia_conventions():
    fb = E.mel_filterbank()
    assert fb.shape == (96, 257) and fb.dtype == np.float32
    assert (fb >= 0).all() and (fb.sum(axis=1) > 0).all()
    # unit_tri: each band integrates to ~1 over Hz (narrow bands near 1 kHz are sampled by few bins)
    hz_per_bin = 8000 / 256
    areas = fb.sum(axis=1) * hz_per_bin
    assert np.allclose(areas, 1.0, atol=0.1) and abs(areas.mean() - 1) < 0.01
    assert E._mel2hz(E._hz2mel(2500.0)) == pytest.approx(2500.0)


def test_melspectrogram_shape_frames_and_determinism():
    y = tone_clip(7.3)
    mel = E.melspectrogram(y)
    assert mel.shape == (math.ceil(len(y) / 256) + 1, 96)  # Essentia FrameCutter(startFromZero=false)
    assert mel.dtype == np.float32 and np.isfinite(mel).all() and mel.min() >= 0
    assert np.array_equal(mel, E.melspectrogram(y.copy()))
    # a 440 Hz tone peaks in the band around 440 Hz
    peak = int(np.median(mel[5:-5].argmax(axis=1)))
    centres = [E._mel2hz(E._hz2mel(0) + (i + 1) * (E._hz2mel(8000) / 97)) for i in range(96)]
    assert abs(centres[peak] - 440) < 60
    assert E.melspectrogram(np.zeros(0, np.float32)).shape == (0, 96)


def test_patches_spread_and_pad():
    mel = np.arange(1876 * 96, dtype=np.float32).reshape(1876, 96)  # 30 s at 16 kHz
    p = E.patches(mel)
    assert p.shape == (E.MAX_PATCHES, 128, 96)
    assert p[0, 0, 0] == 0 and p[-1, -1, 0] >= mel[-130, 0]  # first and last patches cover both ends
    short = E.patches(mel[:50])
    assert short.shape == (1, 128, 96)
    assert E.patches(mel[:0]).shape == (0, 128, 96)


def test_style_vector_and_distance():
    a = [0.9, 0.04, 0.0, 0.01]
    b = [0.88, 0.05, 0.0, 0.02]
    c = [0.01, 0.0, 0.9, 0.04]
    va, vb, vc = style_vector(a), style_vector(b), style_vector(c)
    assert np.isclose(np.linalg.norm(va), 1.0)
    assert style_distance(va, va) == pytest.approx(0.0, abs=1e-9)
    assert style_distance(va, vb) < 0.05 < style_distance(va, vc) == 1.0  # saturates at STYLE_SCALE
    assert style_vector(None) is None and style_vector([0, 0]) is None
    assert 0 < STYLE_SCALE <= 1 and 0 < W_STYLE < 1


def _norms():
    raws = [extract_features(to_wav_bytes(drum_loop(bpm, seed=i)), with_style=False) for i, bpm in enumerate((90, 92, 120))]
    sc = MfccScaler.fit(raws)
    return raws, sc


def test_distance_uses_style_and_falls_back_without_it():
    raws, sc = _norms()
    n0, n1 = normalize(raws[0], sc), normalize(raws[1], sc)
    assert n0.style is None
    plain = distance(n0, n1)
    same, other = [0.8, 0.1, 0.0], [0.0, 0.1, 0.8]
    with_same = distance(normalize({**raws[0], "style": same}, sc), normalize({**raws[1], "style": same}, sc))
    with_other = distance(normalize({**raws[0], "style": same}, sc), normalize({**raws[1], "style": other}, sc))
    assert with_same < plain < with_other  # DSP-only distances are rescaled to the full range
    assert with_other - with_same == pytest.approx(W_STYLE, abs=1e-9)
    # one side without style -> DSP-only (rescaled), never a crash
    assert distance(normalize({**raws[0], "style": same}, sc), n1) == pytest.approx(plain)
    cen = centroid_of([normalize({**raws[0], "style": same}, sc), normalize({**raws[1], "style": other}, sc), n0])
    assert cen.style is not None and cen.style.shape == (3,)


def test_embeddings_switch(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDINGS", "off")
    assert not E.enabled()
    monkeypatch.setenv("EMBEDDINGS", "onnx")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(tmp_path / "missing.onnx"))
    assert not E.enabled()  # no model file: DSP only


MODEL = os.environ.get("EMBEDDING_MODEL_PATH") or E.DEFAULT_MODEL_PATH


@pytest.mark.skipif(not os.path.exists(MODEL), reason="style model not downloaded (python -m app.audio.embed download)")
def test_onnx_style_profile_end_to_end(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "onnx")
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", MODEL)
    data = to_wav_bytes(drum_loop(120, seed=3))
    raw = extract_features(data)
    assert len(raw["style"]) == E.N_STYLES and all(0 <= x <= 1 for x in raw["style"])
    assert extract_features(data)["style"] == raw["style"]  # deterministic
