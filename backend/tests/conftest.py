from __future__ import annotations

import os

import pytest

# The offline suite is DSP-only and deterministic whether or not the style model is downloaded;
# tests/test_embed.py opts in explicitly.
os.environ.setdefault("EMBEDDINGS", "off")

from app.clients import http as http_mod
from app.config import Settings
from tests import synth


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(http_mod, "BACKOFFS", (0.0, 0.0, 0.0))


@pytest.fixture
def settings(tmp_path):
    return Settings(lastfm_api_key="testkey", cache_dir=str(tmp_path / "cache"), analysis_workers=0,
                    spotify_playlist_source="auto", rate_limit_scale=100.0)


@pytest.fixture(scope="session")
def wavs():
    """Synthetic 30 s previews keyed by name."""
    s = synth
    return {
        "dark90": [s.to_wav_bytes(s.drum_loop(90, hats=None, seed=i, kick_freq=50 + 3 * i)) for i in range(3)],
        "bright120": [s.to_wav_bytes(s.drum_loop(120, kick=0, hats="16th", offbeat_snare=True, seed=10 + i)) for i in range(3)],
        "cand90": s.to_wav_bytes(s.drum_loop(90, hats=None, seed=77, kick_freq=52)),
        "cand120": s.to_wav_bytes(s.drum_loop(120, kick=0, hats="16th", offbeat_snare=True, seed=78)),
        "noise170": s.to_wav_bytes(s.drum_loop(170, kick=0, hats="16th", clicks=False, seed=79, noise_db=-20)),
    }
