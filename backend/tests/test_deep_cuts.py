"""Deep cuts (lesser-known tracks by the playlist's own core artists) and the new-artist listener floor."""
from collections import Counter

import httpx
import respx

from app.cache import Cache
from app.clients.deezer import DeezerTrack
from app.clients.http import HttpClient
from app.clients.lastfm import LastfmClient
from app.clients.spotify import PlaylistTrack
from app.models import AnalyzeRequest
from app.pipeline import (DEEP_CUT_ARTISTS, DEEP_CUT_CLIPS, DEEP_CUT_LFM_RATIO, DEEP_CUT_PER_ARTIST,
                          LastfmPopularity, deep_cut_artists, deep_cut_slots, merge_deep_cuts, min_listeners,
                          pick_deep_cuts, pick_seeds)
from app.scoring.explain import MAX_LEN, deep_cut_why
from app.scoring.score import Scored

_ids = iter(range(1, 10_000))


def t(title, rank, artist="Don Toliver", album="Album", isrc=None) -> DeezerTrack:
    i = next(_ids)
    return DeezerTrack(id=i, title=title, artist=artist, artist_id=1, album=album, artwork_url=None,
                       preview_url=f"https://p/{i}.mp3", link=f"https://d/{i}", rank=rank, isrc=isrc)


def top_list():
    return [
        t("E85", 816_000), t("Party With The Boys", 817_000), t("No Pole", 810_000), t("Body", 724_000),
        t("TF?", 856_000),                                    # top 5 by position
        t("After Party", 830_000),                            # top 5 by rank, even though it is listed 6th
        t("No Idea", 785_000),                                # well-known: > 60% of the best rank
        t("Let's Rock", 300_000, artist="Kodak Black"),       # someone else's song
        t("Body (Remix)", 400_000),                           # version of a hit
        t("Cardigan", 450_000),                               # in the playlist
        t("Way Bigger (Live at Rolling Loud)", 420_000),      # live take
        t("LOVE IS A DRUG", 495_000),                         # deep cut
        t("Way Bigger", 500_000),                             # deep cut (the live take above must not block it)
        t("Way Bigger - Sped Up", 300_000),                   # sped-up edit
        t("Way Bigger (2020 Remaster)", 480_000),             # second version of an accepted song
        t("Jingle Bells Freestyle", 300_000),                 # holiday
        t("Interlude", 300_000),                              # interlude
        t("Isrc Twin", 300_000, isrc="USX1"),                 # same recording as a playlist track
        t("No Photos", 463_000),                              # deep cut
        t("Long Way To Calabasas", 450_000),                  # beyond DEEP_CUT_CLIPS
    ]


def test_pick_deep_cuts_filters_hits_playlist_versions_and_contamination():
    picks = pick_deep_cuts("Don Toliver", top_list(), {"cardigan"}, {"USX1"})
    assert [p.title for p in picks] == ["LOVE IS A DRUG", "Way Bigger", "No Photos"]
    assert len(picks) == DEEP_CUT_CLIPS


def test_pick_deep_cuts_respects_rank_ratio_and_handles_empty():
    assert pick_deep_cuts("Don Toliver", [], set(), set()) == []
    assert pick_deep_cuts("Nobody", top_list(), set(), set()) == []
    # a flat catalogue (everything as popular as the hits) has no deep cuts
    flat = [t(f"Song {i}", 800_000 - i) for i in range(12)]
    assert pick_deep_cuts("Don Toliver", flat, set(), set()) == []


def test_deep_cut_artists_need_two_tracks_and_skip_non_artists():
    pt = lambda a, i: PlaylistTrack(spotify_id=f"{a}{i}", title=f"T{i}", artists=[a])
    tracks = ([pt("Daniel Caesar", i) for i in range(12)] + [pt("Don Toliver", i) for i in range(8)]
              + [pt("Grand Theft Auto VI Soundtrack", i) for i in range(5)] + [pt("Solo Act", 0)])
    _, counts = pick_seeds(tracks, [], [], [])
    assert deep_cut_artists(counts) == ["Daniel Caesar", "Don Toliver"]
    assert deep_cut_artists(Counter({"A": 1, "B": 1})) == [], "skipped when no artist has 2+ tracks"
    many = Counter({f"A{i}": 2 + i for i in range(12)})
    assert len(deep_cut_artists(many)) == DEEP_CUT_ARTISTS


def sc(artist, score, key=None, aff=0.8, vibe=0):
    return Scored(key=key or f"{artist}-{score}", artist=artist, vibe_id=vibe, score=score, affinity=aff)


def test_deep_cut_share_and_per_artist_caps():
    for limit in (10, 20, 30):
        lo, target, hi = deep_cut_slots(limit)
        assert 0.25 * limit <= lo <= target <= hi <= 0.40 * limit
        assert target == round(limit / 3)  # popularity mix: about a third deep cuts
    discovery = [sc(f"New {i}", 90 - i) for i in range(20)]
    deep = [sc(f"Core {a}", 99 - j, key=f"dc{a}{j}", aff=0.5) for a in range(2) for j in range(3)]
    out = merge_deep_cuts(discovery[:20 - 4], deep, 20)
    dc = [s for s in out if str(s.key).startswith("dc")]
    assert len(out) == 20
    assert len(dc) == 4  # only 2 artists x DEEP_CUT_PER_ARTIST available (< the 6 target)
    assert max(Counter(s.artist for s in dc).values()) <= DEEP_CUT_PER_ARTIST
    top10 = [s.artist for s in out[:10]]
    assert len(top10) == len(set(top10))


def test_deep_cut_target_one_per_artist_first():
    discovery = [sc(f"New {i}", 80 - i) for i in range(14)]
    deep = [sc(f"Core {a}", 95 - j, key=f"dc{a}{j}", aff=0.5) for a in range(8) for j in range(2)]
    out = merge_deep_cuts(discovery, deep, 20)
    dc = [s for s in out if str(s.key).startswith("dc")]
    assert len(dc) == 7 and len({s.artist for s in dc}) == 7, "7 of 20 (a third), spread over 7 artists"
    assert [s.score for s in out] == sorted((s.score for s in out), reverse=True)


def test_deep_cuts_backfill_a_short_discovery_list():
    discovery = [sc(f"New {i}", 80 - i) for i in range(3)]
    deep = [sc(f"Core {a}", 70 - j, key=f"dc{a}{j}", aff=0.5) for a in range(5) for j in range(3)]
    out = merge_deep_cuts(discovery, deep, 20)
    assert len(out) == 13  # 3 discoveries + 5 artists x 2
    assert max(Counter(s.artist for s in out).values()) <= DEEP_CUT_PER_ARTIST


def test_no_deep_cuts_leaves_discovery_untouched():
    discovery = [sc(f"New {i}", 90 - i) for i in range(20)]
    assert [s.key for s in merge_deep_cuts(discovery, [], 20)] == [s.key for s in discovery]


def test_deep_cut_why_prefix_and_length():
    w = deep_cut_why("Don Toliver", "Near your 112 BPM, same sparse drums, heavier low end")
    assert w == "Deep cut from Don Toliver · Near your 112 BPM, same sparse drums, heavier low end"
    long = deep_cut_why("A Very Long Artist Name That Goes On And On And On Forever More",
                        "Near your 112 BPM, same sparse drums, heavier low end")
    assert len(long) <= MAX_LEN and long.startswith("Lesser-known track by an artist you like")
    w = deep_cut_why("Tyler, The Creator", "Same 92 BPM pocket, matching syncopation, darker; shared tags: rap")
    assert len(w) <= MAX_LEN and "shared tags" not in w and w.startswith("Deep cut from Tyler, The Creator · ")


def test_min_listener_floor():
    assert min_listeners(500_000) == 15_000  # the underground tier (15K-100K) exists at the default cap
    assert min_listeners(1_000_000) == 15_000
    assert min_listeners(200_000) == 15_000
    assert min_listeners(40_000) == 10_000   # never above a quarter of the cap
    assert min_listeners(10_000) == 2_500
    for cap in range(10_000, 1_000_001, 10_000):
        assert min_listeners(cap) < cap


def test_default_listener_cap_is_500k():
    assert AnalyzeRequest(playlist_url="x").max_listeners == 500_000


# --- Last.fm per-track popularity (artist.getTopTracks) ---------------------------------------------

LFM_TOP = [("Swim Good", 360_000, 0), ("Thinkin Bout You", 1_000_000, 0), ("Provider", 220_000, 0),
           ("Pretty Sweet", 310_000, 0)] + [(f"Filler {i}", 100_000 - i, 0) for i in range(10)]


def test_lastfm_popularity_ratio_and_missing_titles():
    pop = LastfmPopularity.build(LFM_TOP)
    assert pop.top == 1_000_000
    assert pop.ratio("Swim Good") == 0.36
    assert pop.ratio("Provider (Remastered)") == 0.22          # decorations are stripped before matching
    assert pop.ratio("Never Charted") == pop.floor / pop.top   # rarer than the list's last entry
    assert LastfmPopularity.build(LFM_TOP[:3]) is None          # too few tracks to judge: Deezer fallback


def test_pick_deep_cuts_uses_lastfm_listener_ratio():
    top = [t(x, 800_000 - 10_000 * i, artist="Frank Ocean") for i, x in
           enumerate(["Thinkin Bout You", "Pink", "Nikes", "Ivy", "Lost", "Swim Good", "Pretty Sweet", "Provider",
                      "Filler 1"])]
    pop = LastfmPopularity.build(LFM_TOP)
    picks = [x.title for x in pick_deep_cuts("Frank Ocean", top, set(), set(), lfm=pop)]
    # Swim Good (36% of the top track's listeners) and Pretty Sweet (31%) are too well known even though
    # their Deezer rank is low; Provider (22%) and the filler qualify despite a high Deezer rank.
    assert picks == ["Provider", "Filler 1"]
    assert all(pop.ratio(p) <= DEEP_CUT_LFM_RATIO for p in picks)
    # Without Last.fm data the Deezer rank rule (<= 60% of the best rank) applies: nothing qualifies here.
    assert pick_deep_cuts("Frank Ocean", top, set(), set()) == []


@respx.mock
async def test_lastfm_artist_top_tracks_parses_and_caches(tmp_path):
    route = respx.get("https://ws.audioscrobbler.com/2.0/", params={"method": "artist.getTopTracks"}).mock(
        return_value=httpx.Response(200, json={"toptracks": {"track": [
            {"name": "Thinkin Bout You", "listeners": "1000000", "playcount": "9000000"},
            {"name": "Provider", "listeners": "220000", "playcount": "1500000"},
            {"name": "Broken", "listeners": "n/a"}]}}))
    async with httpx.AsyncClient() as c:
        lf = LastfmClient(HttpClient(c), Cache(str(tmp_path / "c")), "k")
        rows = await lf.artist_top_tracks("Frank Ocean")
        assert rows == [("Thinkin Bout You", 1_000_000, 9_000_000), ("Provider", 220_000, 1_500_000)]
        assert await lf.artist_top_tracks("Frank Ocean") == rows
    assert route.call_count == 1
