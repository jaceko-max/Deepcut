"""Raw feature extraction: numpy/scipy DSP features plus, when the model is available, the learned
Discogs-EffNet style profile and mood-head outputs (app/audio/embed.py)."""
from __future__ import annotations

import logging

import numpy as np

from app.audio import dsp, embed
from app.audio.decode import SR, AudioError, decode_audio

log = logging.getLogger("deepcut.features")

FEATURE_VERSION = 3  # 2: adds "style" (400 Discogs style activations); 3: adds "mood" + "theme" (mood heads)
MIN_SECONDS = 10.0
SYNC_WEIGHTS = {0: 0.0, 2: 0.5, 1: 1.0, 3: 1.0}


def _syncopation(onsets: np.ndarray, env: np.ndarray, beats: np.ndarray) -> float:
    if len(beats) < 3 or len(onsets) == 0:
        return 0.0
    num = den = 0.0
    for f in onsets:
        k = int(np.searchsorted(beats, f, side="right")) - 1
        if k < 0 or k >= len(beats) - 1:
            continue
        phase = (f - beats[k]) / (beats[k + 1] - beats[k])
        slot = int(round(phase * 4)) % 4
        s = float(env[f])
        num += s * SYNC_WEIGHTS[slot]
        den += s
    return float(num / den) if den > 0 else 0.0


def extract_from_samples(y: np.ndarray) -> dict:
    """y: mono float32 at 22050 Hz. Returns JSON-serializable raw features."""
    total_s = len(y) / SR
    if total_s < MIN_SECONDS:
        raise AudioError(f"clip too short ({total_s:.1f}s)")
    start = int(0.25 * SR)
    y = y[start : start + 30 * SR].astype(np.float32)
    y = y - y.mean()
    if not np.isfinite(y).all() or float(np.abs(y).max()) < 1e-6:
        raise AudioError("silent or invalid clip")
    duration = len(y) / SR

    # loudness on the raw signal
    rms = np.sqrt((dsp.frame_signal(y) ** 2).mean(axis=1))
    rms_db = 20 * np.log10(rms + 1e-9)
    p10, p95 = np.percentile(rms_db, 10), np.percentile(rms_db, 95)
    active = rms_db > p10
    if active.sum() < 3:
        active = np.ones_like(active)
    rms_mean_db = float(rms_db[active].mean())
    dynamic_range_db = float(p95 - p10)

    y = y / (np.abs(y).max() / 0.99)
    P = dsp.stft_power(y)
    logmel = dsp.logmel_from_power(P)

    env = dsp.onset_envelope(logmel)
    bpm_raw, clarity, _ = dsp.estimate_tempo(env)
    bpm_folded = dsp.fold_bpm(bpm_raw)
    lag = dsp.FPS * 60.0 / bpm_raw

    beats = dsp.beat_track(env, bpm_raw)
    if len(beats) >= 4:
        ibi = np.diff(beats).astype(float)
        cv = ibi.std() / max(ibi.mean(), 1e-9)
        regularity = float(np.clip(1 - cv / 0.25, 0, 1))
    else:
        regularity = 0.0
    stability = dsp.pulse_stability(env, lag)

    onsets = dsp.pick_onsets(env)
    onset_density = float(len(onsets) / duration)
    syncopation = _syncopation(onsets, env, beats)

    # spectral, over active frames
    freqs = np.linspace(0, SR / 2, P.shape[1])
    Pa = P[: len(active)][active[: len(P)]] if len(active) >= len(P) else P[active]
    if Pa.shape[0] < 3:
        Pa = P
    tot = Pa.sum(axis=1) + 1e-12
    centroid = (Pa * freqs).sum(axis=1) / tot
    cum = np.cumsum(Pa, axis=1)
    roll = freqs[np.argmax(cum >= 0.85 * cum[:, -1:], axis=1)]
    centroid_log2 = float(np.mean(np.log2(np.maximum(centroid, 20.0))))
    rolloff_log2 = float(np.mean(np.log2(np.maximum(roll, 20.0))))
    bass_ratio = float(Pa[:, freqs < 150].sum() / tot.sum())
    sub_ratio = float(Pa[:, freqs < 60].sum() / tot.sum())
    pe = Pa + 1e-12
    flatness = float(np.mean(np.exp(np.log(pe).mean(axis=1)) / pe.mean(axis=1)))

    mf = dsp.mfcc_from_logmel(logmel)
    mfa = mf[: len(active)][active[: len(mf)]] if len(active) >= len(mf) else mf[active]
    if mfa.shape[0] < 3:
        mfa = mf
    mfcc = np.concatenate([mfa.mean(axis=0), mfa.std(axis=0)])

    out = {
        "version": FEATURE_VERSION,
        "duration_s": float(duration),
        "bpm_raw": float(bpm_raw),
        "bpm_folded": float(bpm_folded),
        "pulse_clarity": float(clarity),
        "pulse_stability": float(stability),
        "regularity": regularity,
        "onset_density": onset_density,
        "syncopation": float(syncopation),
        "centroid_log2": centroid_log2,
        "rolloff_log2": rolloff_log2,
        "bass_ratio": bass_ratio,
        "sub_ratio": sub_ratio,
        "flatness": flatness,
        "rms_mean_db": rms_mean_db,
        "dynamic_range_db": dynamic_range_db,
        "mfcc": [float(v) for v in mfcc],
    }
    for k, v in out.items():
        if isinstance(v, float) and not np.isfinite(v):
            raise AudioError(f"non-finite feature {k}")
    return out


def extract_features(data: bytes, with_style: bool | None = None) -> dict:
    """DSP features, plus "style" (and "mood"/"theme" when the heads are installed) from one EffNet pass."""
    out = extract_from_samples(decode_audio(data))
    if embed.enabled() if with_style is None else with_style:
        try:
            out.update(embed.analyze_samples(decode_audio(data, embed.SR)))
        except Exception:  # noqa: BLE001 - the DSP features alone still score the clip
            log.exception("style model failed; using DSP features only for this clip")
    return out
