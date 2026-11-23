"""Mood heads (preprocessing, metadata, optional ONNX end-to-end), mood vectors/words, mood-first distance,
mood envelope, quality floor, near-duplicates and the popularity-tier quotas."""
from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from app.audio import embed as E
from app.pipeline import (interleave_tiers, min_listeners, popularity_tier, quality_floor, rising_min,
                          tier_quotas, QUALITY_FLOOR, QUALITY_MIN)
from app.scoring import mood as M
from app.scoring.normalize import Norm
from app.scoring.score import (Scored, distance, envelope_ratio, mood_envelope, near_dup_threshold, select_diverse,
                               W_MOOD_M)
from app.scoring.vibes import Vibe, centroid_of, embed as vibe_embed

MODEL = E.model_path()


def mood_of(**over) -> dict:
    """Head outputs at the population mean, with overrides."""
    d = dict(zip(M.MOOD_KEYS, M.MOOD_MEAN.tolist()))
    d.update(over)
    return d


def norm(mood: dict | None = None, style=None, bpm=100.0, tone=0.5) -> Norm:
    st = None if style is None else np.asarray(style, float) / np.linalg.norm(style)
    return Norm(bpm, bpm, np.full(5, 0.5), np.full(7, tone), np.full(24, 0.5), st,
                M.mood_vector(mood), M.mood_z(mood), None)


# -- heads metadata / preprocessing ------------------------------------------------------------------
def test_head_metadata_is_consistent():
    names = [h.name for h in E.HEADS]
    assert tuple(names) == M.MOOD_KEYS, "mood vector order follows the head order"
    files = [h.file for h in E.ALL_HEADS]
    assert len(set(files)) == len(files)
    for h in E.ALL_HEADS:
        assert len(h.sha256) == 64 and int(h.sha256, 16) >= 0
        assert h.file.endswith("-discogs-effnet-1.onnx") and h.path.startswith(h.path.split("/")[0] + "/")
    assert len(E.THEMES) == 56 == len(M.THEME_MEAN) == len(M.THEME_STD)
    assert len(M.MOOD_MEAN) == len(M.MOOD_STD) == len(E.HEADS) and (M.MOOD_STD > 0).all()


def test_heads_live_next_to_the_model_unless_overridden(monkeypatch, tmp_path):
    monkeypatch.setenv("EMBEDDING_MODEL_PATH", str(tmp_path / "m.onnx"))
    monkeypatch.delenv("MOOD_MODELS_DIR", raising=False)
    assert E.head_path(E.HEADS[0]) == str(tmp_path / E.HEADS[0].file)
    monkeypatch.setenv("MOOD_MODELS_DIR", str(tmp_path / "h"))
    assert E.head_path(E.HEADS[0]) == str(tmp_path / "h" / E.HEADS[0].file)
    monkeypatch.setenv("EMBEDDINGS", "onnx")
    assert not E.moods_enabled()  # files missing -> no mood heads (the style model alone still works)


def test_run_heads_rejects_wrong_embedding_shapes():
    with pytest.raises(ValueError):
        E.run_heads(np.zeros((3, 400), np.float32))
    with pytest.raises(ValueError):
        E.run_heads(np.zeros((0, 1280), np.float32))
    with pytest.raises(ValueError):
        E.run_heads(np.zeros(1280, np.float32))


def test_fetch_keeps_a_verified_file_without_network(tmp_path, monkeypatch):
    f = tmp_path / "x.onnx"
    f.write_bytes(b"model bytes")
    sha = hashlib.sha256(b"model bytes").hexdigest()

    def boom(*a, **k):
        raise AssertionError("no download expected")
    monkeypatch.setattr(E.urllib.request, "urlopen", boom)
    assert E._fetch("https://example.invalid/x.onnx", sha, str(f)) == str(f)


@pytest.mark.skipif(not all(os.path.exists(os.path.join(os.path.dirname(MODEL), h.file)) for h in E.ALL_HEADS)
                    or not os.path.exists(MODEL),
                    reason="mood heads not downloaded (python -m app.audio.embed download)")
def test_onnx_mood_heads_end_to_end(monkeypatch):
    monkeypatch.setenv("EMBEDDINGS", "onnx")
    from tests.synth import drum_loop, to_wav_bytes
    from app.audio.features import extract_features
    raw = extract_features(to_wav_bytes(drum_loop(96, seconds=20.0)))
    assert set(raw["mood"]) == set(M.MOOD_KEYS) and len(raw["theme"]) == 56 and len(raw["style"]) == E.N_STYLES
    assert all(-0.5 <= v <= 1.5 for v in raw["mood"].values())  # softmax probabilities / ~0..1 regressions
    assert extract_features(to_wav_bytes(drum_loop(96, seconds=20.0)))["mood"] == raw["mood"]  # deterministic
    assert M.mood_vector(raw["mood"]).shape == (14,)


# -- mood vector / words -----------------------------------------------------------------------------
def test_mood_vector_zscores_and_axes():
    assert M.mood_vector(None) is None and M.mood_vector({"relaxed": 0.5}) is None  # incomplete -> None
    v = M.mood_vector(mood_of())
    assert v.shape == (14,) and np.allclose(v, 0.0)  # population mean -> origin
    z = M.mood_z(mood_of(aggressive=5.0))
    assert z[M.MOOD_KEYS.index("aggressive")] == M.Z_CLIP  # clipped
    ar, va = M.arousal_valence(M.mood_z(mood_of(aggressive=0.95, party=0.95, relaxed=0.05, happy=0.9, sad=0.05)))
    assert ar > 1 and va > 1
    ar, va = M.arousal_valence(M.mood_z(mood_of(aggressive=0.0, party=0.0, relaxed=0.99, happy=0.0, sad=0.95)))
    assert ar < -0.5 and va < -0.5


def test_mood_distance_and_cosine():
    a, b = M.mood_vector(mood_of(relaxed=0.95)), M.mood_vector(mood_of(relaxed=0.05, aggressive=0.95, party=0.9))
    assert M.mood_distance(a, a) == 0.0 and 0 < M.mood_distance(a, b) <= 1.0
    assert M.mood_cosine(a, a) == pytest.approx(1.0) and M.mood_cosine(a, b) < 0


def test_mood_words_describe_and_differ():
    chill = M.mood_z(mood_of(relaxed=0.98, aggressive=0.0, engaging=0.2, voice=0.05, party=0.02))
    words = M.mood_words(chill, None)
    assert "instrumental" in words and ({"chill", "mellow"} & set(words))
    groups = {w: g for w, g, _ in M.WORDS}
    assert len({groups[w] for w in words}) == len(words), "one word per group"
    hard = M.mood_z(mood_of(aggressive=0.99, relaxed=0.0, engaging=0.99))
    assert M.mood_words(hard, None)[0] in ("hard-hitting", "energetic")
    assert M.mood_words(None, None) == []
    # siblings: avoid words an earlier vibe used while alternatives exist
    first = M.mood_words(chill, None)
    assert first[0] not in M.mood_words(chill, None, avoid=set(first[:1]))


# -- mood-first distance and vibes ------------------------------------------------------------------
def test_distance_is_mood_first_and_falls_back():
    st = np.ones(400)
    calm, calm2, hard = (norm(mood_of(relaxed=0.95), st), norm(mood_of(relaxed=0.9), st),
                         norm(mood_of(relaxed=0.05, aggressive=0.95, engaging=0.99), st))
    assert distance(calm, calm2) < distance(calm, hard)
    assert distance(calm, hard) >= W_MOOD_M * M.mood_distance(calm.mood, hard.mood) - 1e-9
    no_mood = norm(None, st)
    assert distance(calm, no_mood) == pytest.approx(distance(norm(None, st), no_mood))  # style + DSP only


def test_vibe_centroids_carry_mood_for_labels():
    st = np.ones(400)
    calm = [norm(mood_of(relaxed=0.95 - 0.01 * i, aggressive=0.02), st) for i in range(4)]
    cen = centroid_of(calm)
    assert cen.mood is not None and cen.mz is not None
    assert vibe_embed(calm).shape[1] == 2 + 5 + 7 + 24  # clustering stays on DSP (see app/scoring/vibes.py)
    assert "mellow" in M.mood_words(cen.mz, None) or "chill" in M.mood_words(cen.mz, None)


# -- envelope / floor / near-duplicates ---------------------------------------------------------------
def _one_vibe(norms):
    return [Vibe(0, list(range(len(norms))), 1.0, 100.0, centroid_of(norms))]


def test_mood_envelope_rejects_outliers_keeps_members():
    st = np.ones(400)
    rng = np.random.default_rng(1)
    pl = [norm(mood_of(relaxed=0.9 + rng.uniform(-0.05, 0.05), sad=0.6 + rng.uniform(-0.1, 0.1)), st) for _ in range(12)]
    vibes = _one_vibe(pl)
    radii = mood_envelope(pl, vibes)
    assert set(radii) == {0} and radii[0] > 0
    inside = norm(mood_of(relaxed=0.9, sad=0.6), st)
    outlier = norm(mood_of(relaxed=0.02, aggressive=0.98, party=0.95, engaging=0.99, sad=0.0), st)
    assert envelope_ratio(inside, vibes, radii) <= 1.0 < envelope_ratio(outlier, vibes, radii)
    assert mood_envelope([norm(None, st)] * 4, _one_vibe([norm(None, st)] * 4)) is None  # no heads: no check
    assert envelope_ratio(norm(None, st), vibes, radii) is None


def test_quality_floor_follows_playlist_coherence():
    assert quality_floor([], 0.8, 0.2) == QUALITY_FLOOR
    assert quality_floor([90] * 20, 0.8, 0.2) == QUALITY_FLOOR       # coherent playlist: the fixed floor
    loose = quality_floor([60] * 20, 0.8, 0.2)                        # loose playlist: its own level
    assert QUALITY_MIN <= loose < QUALITY_FLOOR and loose == round(0.8 * 60 + 0.2 * 100 * 0.75)
    assert quality_floor([5] * 20, 0.8, 0.2) == QUALITY_MIN


def test_near_dup_threshold_is_small_and_cross_artist():
    st = np.ones(400)
    pl = [norm(mood_of(relaxed=0.1 * i), st) for i in range(10)]
    t = near_dup_threshold(pl, [f"a{i}" for i in range(10)])
    assert 0 < t < min(distance(pl[i], pl[i + 1]) for i in range(9))
    assert near_dup_threshold(pl[:2], ["same", "same"]) == 0.0


# -- popularity tiers ----------------------------------------------------------------------------------
def test_tiers_and_quotas():
    assert min_listeners(500_000) == 15_000 and rising_min(500_000) == 100_000
    assert popularity_tier(99_999, 500_000) == "underground" and popularity_tier(100_000, 500_000) == "rising"
    assert rising_min(60_000) == 20_000 and rising_min(10_000) >= min_listeners(10_000)
    assert tier_quotas(13) == {"rising": 7, "underground": 6} and sum(tier_quotas(14).values()) == 14

    class C:
        def __init__(self, n, l):
            self.name, self.listeners = n, l
    order = interleave_tiers([C("r1", 300_000), C("r2", 200_000), C("r3", 150_000), C("u1", 20_000)], 500_000)
    assert [c.name for c in order] == ["r1", "u1", "r2", "r3"]


def sd(artist, score, tier, fam=None):
    return Scored(key=artist, artist=artist, vibe_id=0, score=score, affinity=1.0, family=fam, tier=tier)


def test_select_diverse_fills_tier_quotas_then_backfills():
    pool = [sd(f"R{i}", 95 - i, "rising") for i in range(10)] + [sd(f"U{i}", 80 - i, "underground") for i in range(3)]
    out = select_diverse(pool, 8, {0: 1.0}, tier_quota={"rising": 4, "underground": 4})
    assert len(out) == 8
    assert sum(s.tier == "underground" for s in out) == 3, "all 3 underground picks in, despite lower scores"
    assert sum(s.tier == "rising" for s in out) == 5, "rising backfills the short underground tier"
    plain = select_diverse(pool, 8, {0: 1.0})
    assert all(s.tier == "rising" for s in plain)  # without quotas the score order wins


def test_select_diverse_tier_quota_loosens_family_caps_first():
    pool = ([sd(f"R{i}", 95 - i, "rising", fam=f"f{i}") for i in range(10)]
            + [sd(f"U{i}", 80 - i, "underground", fam="same") for i in range(4)])
    out = select_diverse(pool, 8, {0: 1.0}, {f"f{i}": 0.05 for i in range(10)} | {"same": 0.05},
                         tier_quota={"rising": 4, "underground": 4})
    assert sum(s.tier == "underground" for s in out) == 4


def test_select_diverse_skips_near_duplicates():
    pool = [sd("A", 95, "rising"), sd("B", 94, "rising"), sd("C", 90, "rising"), sd("D", 80, "rising")]
    twins = {frozenset(("A", "B"))}
    out = select_diverse(pool, 3, {0: 1.0}, too_close=lambda a, b: frozenset((a.key, b.key)) in twins)
    assert [s.artist for s in out] == ["A", "C", "D"]


def test_with_mood_lowercases_the_clauses_after_the_lead():
    from app.scoring.explain import with_mood

    why = with_mood("Near your 88 BPM, same sparse drums", ["acoustic"], ["acoustic"])
    assert why == "Same acoustic mood, near your 88 BPM, same sparse drums"
    assert with_mood("Half-time take on your 88 BPM groove", ["melancholic"], []) == \
        "Melancholic mood, half-time take on your 88 BPM groove"
