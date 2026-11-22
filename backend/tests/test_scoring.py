import math

import numpy as np
import pytest

from app.scoring.explain import explain, vibe_label
from app.scoring.normalize import MfccScaler, Norm, lin, normalize
from app.scoring.score import Scored, compute_d_ref, distance, match_score, select_diverse
from app.scoring.vibes import build_vibes, centroid_of


def raw(**kw):
    base = dict(bpm_raw=120.0, bpm_folded=120.0, pulse_clarity=0.45, onset_density=4.0, regularity=0.8,
                syncopation=0.3, pulse_stability=0.6, centroid_log2=math.log2(1500), rolloff_log2=math.log2(4000),
                bass_ratio=0.3, sub_ratio=0.1, rms_mean_db=-20.0, dynamic_range_db=15.0, flatness=0.1,
                mfcc=list(np.linspace(-5, 5, 24)))
    base.update(kw)
    return base


def mk(rng, bpm, tone_c, n, groove_c=0.5):
    out = []
    for _ in range(n):
        out.append(raw(bpm_raw=bpm * rng.uniform(.98, 1.02), bpm_folded=bpm * rng.uniform(.98, 1.02),
                       centroid_log2=math.log2(tone_c) + rng.normal(0, .05), bass_ratio=0.5 - 0.4 * (tone_c > 2000) + rng.normal(0, .02),
                       onset_density=0.5 + 7 * groove_c + rng.normal(0, .8),
                       regularity=float(np.clip(0.7 + rng.normal(0, .15), 0, 1)),
                       syncopation=float(np.clip(0.3 + rng.normal(0, .12), 0, 0.6)),
                       pulse_clarity=float(np.clip(0.45 + rng.normal(0, .15), 0.1, .8)),
                       rms_mean_db=-20 + rng.normal(0, 4), flatness=float(np.clip(0.1 + rng.normal(0, .06), 0, .4)),
                       mfcc=list(np.linspace(-5, 5, 24) + (tone_c / 500) + rng.normal(0, .2, 24))))
    return out


def norms_of(raws):
    sc = MfccScaler.fit(raws)
    return [normalize(r, sc) for r in raws], sc


# normalize ---------------------------------------------------------------
def test_lin_clips():
    assert lin(-1, 0, 1) == 0 and lin(2, 0, 1) == 1 and lin(0.5, 0, 1) == 0.5


def test_normalize_ranges_and_summary():
    n, _ = norms_of([raw(), raw(bpm_raw=90, bpm_folded=90, onset_density=20, mfcc=list(np.zeros(24)))])
    for x in n:
        for v in (x.groove, x.tone, x.timbre):
            assert ((v >= 0) & (v <= 1)).all()
    s = n[0].summary()
    assert set(s) == {"bpm", "pulse_clarity", "onset_density", "regularity", "syncopation", "brightness",
                      "low_end", "energy", "dynamics"}
    assert n[1].groove[1] == 1.0  # onset density saturates


def test_mfcc_robust_scaling_centered():
    rs = [raw(mfcc=list(np.full(24, float(i)))) for i in range(9)]
    sc = MfccScaler.fit(rs)
    assert sc.transform(np.full(24, 4.0)) == pytest.approx(np.full(24, 0.5))


# vibes -------------------------------------------------------------------
def test_two_blobs_give_two_vibes():
    rng = np.random.default_rng(0)
    n, _ = norms_of(mk(rng, 90, 600, 12) + mk(rng, 125, 3500, 12, groove_c=0.8))
    v = build_vibes(n)
    assert len(v) == 2
    assert sorted(len(x.members) for x in v) == [12, 12]
    assert v[0].id == 0 and sum(x.share for x in v) == pytest.approx(1)


def test_one_blob_gives_one_vibe():
    rng = np.random.default_rng(0)  # NB: single-blob silhouette is ~0.16-0.26 across seeds; threshold is 0.25
    n, _ = norms_of(mk(rng, 100, 1200, 20))
    assert len(build_vibes(n)) == 1


def test_small_playlist_is_one_vibe():
    rng = np.random.default_rng(2)
    n, _ = norms_of(mk(rng, 90, 600, 3) + mk(rng, 130, 3500, 4))
    assert len(build_vibes(n)) == 1


def test_circular_centroid_tempo():
    rng = np.random.default_rng(3)
    n, _ = norms_of([raw(bpm_folded=b, bpm_raw=b) for b in (136.0, 72.0, 138.0)])
    c = centroid_of(n)
    assert c.bpm_folded > 130 or c.bpm_folded < 74  # wraps across the 70/140 seam


# score -------------------------------------------------------------------
def test_identical_scores_100_and_monotonic():
    n, _ = norms_of([raw(), raw(onset_density=6.0, bpm_folded=100.0), raw(onset_density=7.5, bpm_folded=85.0, bass_ratio=0.0)])
    d = [distance(n[0], x) for x in n]
    assert d[0] == 0
    assert match_score(0, 0.15) == 100
    s = [match_score(x, 0.15) for x in d]
    assert s[0] > s[1] > s[2]


def test_d_equals_dref_gives_85():
    assert match_score(0.12, 0.12) == 85
    assert match_score(0.24, 0.12) < 60


def test_d_ref_clamped():
    rng = np.random.default_rng(4)
    n, _ = norms_of(mk(rng, 100, 1200, 10))
    assert 0.06 <= compute_d_ref(n, build_vibes(n)) <= 0.25


def test_select_diverse_caps_and_shares():
    items = [Scored(f"a{i}", "Same Artist", 0, 99 - i, 0.5) for i in range(6)]
    items += [Scored(f"b{i}", f"Other {i}", 0, 80 - i, 0.5) for i in range(6)]
    items += [Scored(f"c{i}", f"Third {i}", 1, 60 - i, 0.5) for i in range(6)]
    out = select_diverse(items, 10, {0: 0.6, 1: 0.4})
    arts = [s.artist for s in out]
    assert arts.count("Same Artist") <= 2
    assert len(out) == 10
    assert len(set(arts[:10])) == 10
    assert sum(1 for s in out if s.vibe_id == 1) >= 3


def test_select_diverse_top10_never_repeats_artist_even_if_dup_scores_higher():
    items = [Scored("x1", "Dup", 0, 99, 0.5), Scored("x2", "Dup", 0, 98, 0.5)]
    items += [Scored(f"o{i}", f"Other {i}", 0, 90 - i, 0.5) for i in range(12)]
    out = select_diverse(items, 14, {0: 1.0})
    top = [s.artist for s in out[:10]]
    assert len(set(top)) == 10
    assert [s.score for s in out[:3]] == [99, 90, 89]  # still score-descending apart from the dedupe
    assert [s.artist for s in out].count("Dup") == 2


# explain -----------------------------------------------------------------
def test_explain_half_time_and_length():
    n, _ = norms_of([raw(), raw(bass_ratio=0.0, centroid_log2=math.log2(4400), onset_density=0.6)])
    why = explain(n[1], 61.0, n[0], 122.0)
    assert why.startswith("Half-time take on your 122 BPM groove")
    assert len(why) <= 90
    assert explain(n[0], 121.0, n[0], 120.0).startswith("Same 120 BPM pocket")
    assert explain(n[0], 240.0, n[0], 120.0).startswith("Double-time")
    assert explain(n[0], 100.0, n[0], 120.0).startswith("Near your 120 BPM")


def test_explain_difference_clause():
    n, _ = norms_of([raw(), raw(centroid_log2=math.log2(4400), bass_ratio=0.0)])
    why = explain(n[1], 120.0, n[0], 120.0)
    assert "brighter" in why or "lighter low end" in why


def test_vibe_label():
    n, _ = norms_of([raw(centroid_log2=math.log2(300), bass_ratio=0.5)])
    lab = vibe_label(n[0], 92.0)
    assert lab.startswith("92 BPM · ") and "dark" in lab
