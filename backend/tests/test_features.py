import math

import numpy as np
import pytest

from app.audio import dsp
from app.audio.decode import AudioError, decode_audio
from app.audio.features import extract_features, extract_from_samples
from tests.synth import drum_loop, to_wav_bytes


def near_octave(a, b, tol=0.02):
    return any(abs(a - b * m) / (b * m) <= tol for m in (0.5, 1, 2))


@pytest.mark.parametrize("bpm", [90, 120, 140])
def test_tempo_within_two_percent(bpm):
    f = extract_from_samples(drum_loop(bpm))
    assert near_octave(f["bpm_raw"], bpm), f["bpm_raw"]


def test_tempo_174_folds_to_87():
    f = extract_from_samples(drum_loop(174))
    assert abs(f["bpm_folded"] - 87) <= 2, f["bpm_folded"]
    assert 70 <= f["bpm_folded"] < 140


def test_fold_and_octave_distance():
    assert dsp.fold_bpm(174) == 87 and dsp.fold_bpm(50) == 100
    assert dsp.octave_distance(60, 120) == pytest.approx(0, abs=1e-9)
    assert dsp.octave_distance(100, 100) == 0
    assert dsp.octave_distance(100, 100 * 2 ** 0.5) == pytest.approx(0.5)


def test_bass_vs_hats_spectral_ordering():
    bass = extract_from_samples(drum_loop(120, hats=None))
    hats = extract_from_samples(drum_loop(120, kick=0, hats="8th"))
    assert bass["bass_ratio"] > hats["bass_ratio"] + 0.5
    assert hats["centroid_log2"] > bass["centroid_log2"] + 2
    assert hats["rolloff_log2"] > bass["rolloff_log2"]


def test_syncopated_beats_straight():
    straight = extract_from_samples(drum_loop(120, hats="8th"))
    synco = extract_from_samples(drum_loop(120, hats="8th", offbeat_snare=True))
    assert synco["syncopation"] > straight["syncopation"] + 0.05


def test_16th_denser_than_8th():
    e8 = extract_from_samples(drum_loop(120, hats="8th"))
    e16 = extract_from_samples(drum_loop(120, hats="16th"))
    assert e16["onset_density"] > e8["onset_density"] * 1.3


def test_click_track_is_regular():
    f = extract_from_samples(drum_loop(120, clicks=True))
    assert f["regularity"] > 0.8
    assert f["pulse_clarity"] > 0.3


def test_finite_and_deterministic():
    y = drum_loop(100, seed=3)
    a, b = extract_from_samples(y), extract_from_samples(y.copy())
    assert a == b
    assert len(a["mfcc"]) == 24
    for k, v in a.items():
        if k != "mfcc":
            assert np.isfinite(v), k
    assert np.isfinite(a["mfcc"]).all()


def test_wav_decode_via_miniaudio():
    y = drum_loop(120, seconds=12)
    dec = decode_audio(to_wav_bytes(y))
    assert dec.dtype == np.float32 and abs(len(dec) - len(y)) < 50
    f = extract_features(to_wav_bytes(y))
    assert near_octave(f["bpm_raw"], 120)


def test_rejects_short_and_undecodable():
    with pytest.raises(AudioError):
        extract_from_samples(drum_loop(120, seconds=6))
    with pytest.raises(AudioError):
        decode_audio(b"definitely not audio")


def test_beat_tracker_finds_regular_beats():
    env = dsp.onset_envelope(dsp.logmel_from_power(dsp.stft_power(drum_loop(120, clicks=True) )))
    beats = dsp.beat_track(env, 120.0)
    assert len(beats) > 40
    ibi = np.diff(beats)
    assert abs(np.median(ibi) - dsp.FPS * 0.5) < 1.5


def test_process_pool_executor_roundtrip():
    import asyncio

    from app.audio.worker import analyze_async, make_executor
    ex = make_executor(1)
    try:
        f = asyncio.run(analyze_async(ex, to_wav_bytes(drum_loop(120, seconds=12))))
    finally:
        ex.shutdown()
    assert near_octave(f["bpm_raw"], 120)


# pool recovery -----------------------------------------------------------
def _die_once(data: bytes) -> dict:
    import os
    flag = data.decode()
    if not os.path.exists(flag):
        open(flag, "w").close()
        os._exit(1)  # simulates an OOM-killed worker
    return {"ok": 1}


def _always_die(data: bytes) -> dict:
    import os
    os._exit(1)


def test_pool_recovers_after_worker_death(tmp_path):
    import asyncio
    from app.audio.worker import AnalysisPool
    pool = AnalysisPool(1, fn=_die_once)
    try:
        flag = str(tmp_path / "flag").encode()
        assert asyncio.run(pool.analyze(flag)) == {"ok": 1}  # broke, rebuilt, retried
        assert pool.rebuilds == 1 and pool.failed is False
        assert asyncio.run(pool.analyze(flag)) == {"ok": 1}  # subsequent clips work
    finally:
        pool.shutdown()


def test_pool_marks_failed_when_rebuild_keeps_breaking():
    import asyncio
    import pytest
    from concurrent.futures import BrokenExecutor
    from app.audio.worker import AnalysisPool
    pool = AnalysisPool(1, fn=_always_die)
    try:
        with pytest.raises(BrokenExecutor):
            asyncio.run(pool.analyze(b"x"))
        assert pool.failed is True
    finally:
        pool.shutdown()
