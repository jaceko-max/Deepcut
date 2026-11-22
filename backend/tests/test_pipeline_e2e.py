"""Offline end-to-end pipeline: respx-mocked Spotify/Last.fm/Deezer + synthetic WAV previews."""
import re
from collections import Counter

import httpx
import pytest
import respx

from app.jobs import JobStore, Reporter
from app.models import JobResponse
from app.pipeline import PipelineError, build_services, run_analysis
from tests.helpers import PID, embed_html

LFM = "https://ws.audioscrobbler.com/2.0/"

# artist -> (listeners, wav key, deezer id)
CANDS = {
    "Cand Ninety": (30_000, "cand90", 101),
    "Cand Onetwenty": (45_000, "cand120", 102),
    "Noise Merchant": (20_000, "noise170", 103),
    "Big Star": (5_000_000, "cand90", 104),
    "PlA": (10_000, "cand90", 105),   # artist already in the playlist -> excluded
    "Reggaeton Kid": (15_000, "cand90", 106),       # off-genre tags -> rejected before Deezer
    "Gizzard Tribute Band": (8_000, "cand90", 107),  # tribute act -> rejected by name
}
LB_SIM = "https://labs.api.listenbrainz.org/similar-artists/json"
LB_POP = "https://api.listenbrainz.org/1/popularity/artist"
GENRE_TAGS = [{"name": "seen live", "count": 100}, {"name": "trip hop", "count": 80}, {"name": "downtempo", "count": 50}]
TAGS = {"Reggaeton Kid": [{"name": "reggaeton", "count": 100}, {"name": "latin", "count": 60}, {"name": "pop", "count": 20}]}


def mbid(name: str) -> str:
    h = f"{abs(hash(name)) % 10**12:012d}"
    return f"00000000-0000-4000-8000-{h}"


def world(wavs, counters, lb_rows=None, lb_status=200, lb_pop=None, bogus=None, dz_fans=None,
          artists=None, deep_tops=None):
    """artists: optional 12 playlist subtitles; deep_tops: {artist: (deezer id, [(title, rank, wav key, artist)])}
    served by /search/artist + /artist/{id}/top for the deep-cut stage."""
    deep_tops = deep_tops or {}
    deep_ids = {v[0]: (name, v[1]) for name, v in deep_tops.items()}
    deep_wav: dict[int, str] = {}
    tracks = []
    previews = {}
    for i in range(12):
        vibe = "dark90" if i < 6 else "bright120"
        key = f"t{i:02d}"
        previews[key] = wavs[vibe][i % 3]
        tracks.append({"uri": f"spotify:track:{key}{'x' * 19}", "title": f"Track {i}",
                       "subtitle": artists[i] if artists else f"Pl{chr(65 + i)}",
                       "duration": 200000, "audioPreview": {"url": f"https://p.scdn.co/mp3-preview/{key}"}})
    router = respx.mock(assert_all_called=False)
    router.get(f"https://open.spotify.com/embed/playlist/{PID}").mock(
        return_value=httpx.Response(200, text=embed_html(tracks)))

    def sp_preview(request):
        k = request.url.path.rsplit("/", 1)[-1]
        counters["sp"] += 1
        return httpx.Response(200, content=previews[k])
    router.get(url__regex=r"https://p\.scdn\.co/mp3-preview/\w+").mock(side_effect=sp_preview)

    def lfm(request):
        counters["lfm"] += 1
        p = request.url.params
        m = p["method"]
        if m == "artist.getSimilar":
            return httpx.Response(200, json={"similarartists": {"artist": [
                {"name": n, "match": "0.8", "mbid": mbid(n)} for n in CANDS]}})
        if m == "track.getSimilar":
            return httpx.Response(200, json={"similartracks": {"track": []}})
        if m == "artist.getTopTags":
            return httpx.Response(200, json={"toptags": {"tag": TAGS.get(p["artist"], GENRE_TAGS)}})
        if m == "tag.getTopArtists":
            return httpx.Response(200, json={"topartists": {"artist": [{"name": "Cand Ninety"}]}})
        if m == "artist.getInfo":
            a = p["artist"]
            counters["info:" + a] += 1
            listeners = CANDS[a][0] if a in CANDS else 1_000
            if bogus and a in bogus:  # autocorrect lands on a tiny duplicate page
                name, small, real = bogus[a]
                if p["autocorrect"] == "1":
                    return httpx.Response(200, json={"artist": {"name": name, "stats": {"listeners": str(small)}}})
                return httpx.Response(200, json={"artist": {"name": a, "mbid": mbid(a), "stats": {"listeners": str(real)}}})
            return httpx.Response(200, json={"artist": {"mbid": mbid(a), "stats": {"listeners": str(listeners)}}})
        return httpx.Response(200, json={})
    router.get(LFM).mock(side_effect=lfm)

    def lb_sim(request):
        counters["lb"] += 1
        if lb_status != 200:
            return httpx.Response(lb_status, text="down")
        ref = request.url.params["artist_mbids"]
        rows = [{"artist_mbid": ref, "name": "itself", "score": 999, "reference_mbid": ref}]  # LB returns the seed
        rows += [{"artist_mbid": mbid(n), "name": n, "score": sc, "reference_mbid": ref} for n, sc in (lb_rows or [])]
        return httpx.Response(200, json=rows)
    router.get(LB_SIM).mock(side_effect=lb_sim)

    def lb_pop_route(request):
        counters["lbpop"] += 1
        if lb_status != 200:
            return httpx.Response(lb_status, text="down")
        ids = __import__("json").loads(request.content)["artist_mbids"]
        pop = lb_pop or {}
        return httpx.Response(200, json=[{"artist_mbid": i, "total_user_count": pop.get(i, 100), "total_listen_count": 1000}
                                         for i in ids])
    router.post(LB_POP).mock(side_effect=lb_pop_route)

    def dz_artist(request):
        q = request.url.params["q"]
        counters["dzartist:" + q] += 1
        if q in deep_tops:
            return httpx.Response(200, json={"data": [{"id": deep_tops[q][0], "name": q, "nb_fan": 5_000_000}]})
        if q not in CANDS:
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json={"data": [{"id": CANDS[q][2], "name": q, "nb_fan": (dz_fans or {}).get(q, 1000)}]})
    router.get("https://api.deezer.com/search/artist").mock(side_effect=dz_artist)

    def dz_top(request):
        did = int(request.url.path.split("/")[2])
        if did in deep_ids:
            name, rows = deep_ids[did]
            data = []
            for j, (title, rank, wav, artist) in enumerate(rows):
                tid = did * 100 + j
                deep_wav[tid] = wav
                data.append({"id": tid, "title": title, "duration": 200, "rank": rank,
                             "link": f"https://www.deezer.com/track/{tid}",
                             "preview": f"https://cdnt-preview.dzcdn.net/{tid}.mp3?hdnea=x",
                             "artist": {"id": did, "name": artist or name},
                             "album": {"title": "Deep Album", "cover_medium": "https://img/d.jpg"}})
            return httpx.Response(200, json={"data": data})
        name = next(n for n, v in CANDS.items() if v[2] == did)
        track = lambda tid, title, album: {
            "id": tid, "title": title, "duration": 200, "link": f"https://www.deezer.com/track/{tid}",
            "preview": f"https://cdnt-preview.dzcdn.net/{did}.mp3?hdnea=x", "artist": {"id": did, "name": name},
            "album": {"title": album, "cover_medium": "https://img/c.jpg"}}
        # most popular first: a live take and a Christmas song that the contamination filter must drop
        return httpx.Response(200, json={"data": [track(did * 10 + 1, f"{name} Song (Live at Levitation)", "Live"),
                                                  track(did * 10 + 2, "Last Christmas", "Holiday Hits"),
                                                  track(did * 10, f"{name} Song", "Alb")]})
    router.get(url__regex=r"https://api\.deezer\.com/artist/\d+/top").mock(side_effect=dz_top)

    def dz_preview(request):
        counters["dz"] += 1
        did = int(re.search(r"/(\d+)\.mp3", request.url.path).group(1))
        if did in deep_wav:
            return httpx.Response(200, content=wavs[deep_wav[did]])
        key = next(v[1] for v in CANDS.values() if v[2] == did)
        return httpx.Response(200, content=wavs[key])
    router.get(url__regex=r"https://cdnt-preview\.dzcdn\.net/\d+\.mp3").mock(side_effect=dz_preview)
    return router


async def run_once(settings, wavs, counters, limit=10, **world_kw):
    """Cap 200K -> new-artist listener floor min(50K, max(15K, 20K)) = 20K."""
    svc = build_services(settings)
    store = JobStore(settings)
    job = store.create(200_000, limit)
    rep = Reporter(store, job.job_id)
    try:
        with world(wavs, counters, **world_kw):
            await run_analysis(svc, PID, 200_000, limit, rep)
    finally:
        await svc.aclose()
    return store.get(job.job_id)


async def test_full_pipeline_offline(settings, wavs):
    c = Counter()
    job = await run_once(settings, wavs, c)
    assert job.status == "done" and job.stage == "done" and job.progress == 1.0, job.error
    assert job.playlist.source == "spotify_embed"
    assert job.playlist.total_tracks == 12 and job.playlist.analyzed_tracks == 12
    prof = job.playlist_profile
    assert len(prof.vibes) == 2
    assert sorted(round(v.bpm) for v in prof.vibes)[0] in range(86, 94)
    assert abs(sum(v.share for v in prof.vibes) - 1) < 0.01

    by_artist = {r.artist: r for r in job.results}
    assert "Big Star" not in by_artist          # 5M listeners -> filtered_mainstream
    assert "PlA" not in by_artist               # playlist artist excluded
    assert job.stats.filtered_mainstream == 1
    assert job.stats.filtered_in_playlist == 1
    assert job.stats.candidates_analyzed == 3
    assert set(by_artist) == {"Cand Ninety", "Cand Onetwenty", "Noise Merchant"}
    assert by_artist["Cand Onetwenty"].match_score > by_artist["Noise Merchant"].match_score
    assert by_artist["Cand Ninety"].match_score > by_artist["Noise Merchant"].match_score
    assert [r.rank for r in job.results] == [1, 2, 3]
    assert by_artist["Noise Merchant"].rank == 3
    for r in job.results:
        assert r.preview_proxy_url.startswith(f"/api/preview/{r.deezer_id}?sig=")
        assert r.spotify_search_url.startswith("https://open.spotify.com/search/") and " " not in r.spotify_search_url
        assert len(r.why) <= 90 and 0 <= r.match_score <= 100
        assert r.listeners == CANDS[r.artist][0]
        assert r.deezer_id % 10 == 0, "live / Christmas versions must be filtered out"
        assert "trip hop" in r.why or "downtempo" in r.why, r.why
        assert r.source != "deep_cut"
    assert not [k for k in c if k.startswith("dzartist:Pl")], "no artist has 2+ tracks -> no deep-cut lookups"
    assert by_artist["Cand Ninety"].vibe_id != by_artist["Cand Onetwenty"].vibe_id
    assert not [w for w in job.warnings if "ListenBrainz" in w]
    JobResponse.model_validate(job.model_dump(mode="json"))


async def test_listenbrainz_candidates_merge_and_boost(settings, wavs):
    CANDS["LB Only"] = (25_000, "cand120", 108)
    try:
        c = Counter()
        job = await run_once(settings, wavs, c, lb_rows=[("LB Only", 500), ("Cand Ninety", 400), ("Big Star", 300)])
    finally:
        del CANDS["LB Only"]
    assert job.status == "done", job.error
    by_artist = {r.artist: r for r in job.results}
    assert "LB Only" in by_artist and by_artist["LB Only"].source == "similar_artist"
    assert "itself" not in by_artist and "Big Star" not in by_artist
    assert c["lb"] > 0 and c["lbpop"] > 0
    assert not [w for w in job.warnings if "ListenBrainz" in w]


async def test_listenbrainz_popularity_is_a_second_mainstream_signal(settings, wavs):
    c = Counter()
    job = await run_once(settings, wavs, c, lb_pop={mbid("Cand Onetwenty"): 900_000})
    assert job.status == "done", job.error
    assert "Cand Onetwenty" not in {r.artist for r in job.results}
    assert job.stats.filtered_mainstream == 2
    assert c["info:Cand Onetwenty"] == 0, "Last.fm MBID + ListenBrainz prefilter must skip the getInfo lookup"


async def test_listenbrainz_outage_degrades_to_lastfm_only(settings, wavs):
    c = Counter()
    job = await run_once(settings, wavs, c, lb_status=503)
    assert job.status == "done", job.error
    assert {r.artist for r in job.results} == {"Cand Ninety", "Cand Onetwenty", "Noise Merchant"}
    assert any("ListenBrainz" in w for w in job.warnings)


async def test_progress_messages_count_items(settings, wavs):
    from app.jobs import JobStore as JS
    seen = []
    orig = JS.update

    def spy(self, jid, **fields):
        orig(self, jid, **fields)
        j = self.jobs.get(jid)
        if j:
            seen.append((j.progress, j.message or ""))
    JS.update = spy
    try:
        job = await run_once(settings, wavs, Counter())
    finally:
        JS.update = orig
    assert job.status == "done"
    progs = [p for p, _ in seen]
    assert progs == sorted(progs), "progress must be monotonic"
    msgs = " | ".join(m for _, m in seen)
    for label in ("Checking listener counts", "Comparing genre tags", "Looking up previews"):
        assert re.search(label + r" \d+/\d+", msgs), label
    mid = [p for p, m in seen if 0.45 < p < 0.55]
    assert len(set(mid)) >= 5, "progress must advance through the candidate stage"


async def test_second_run_hits_feature_cache(settings, wavs):
    c1, c2 = Counter(), Counter()
    j1 = await run_once(settings, wavs, c1)
    assert c1["sp"] == 12 and c1["dz"] == 3
    j2 = await run_once(settings, wavs, c2)
    assert j2.status == "done"
    assert c2["sp"] == 0 and c2["dz"] == 0, "audio must come from the SQLite feature cache"
    assert c2["lfm"] == 0, "Last.fm responses must come from the http cache"
    assert [r.deezer_id for r in j1.results] == [r.deezer_id for r in j2.results]
    assert [r.match_score for r in j1.results] == [r.match_score for r in j2.results]


async def test_too_few_previews_errors(settings):
    svc = build_services(settings)
    store = JobStore(settings)
    job = store.create(200_000, 10)
    tracks = [{"uri": f"spotify:track:n{i}{'x' * 20}", "title": f"T{i}", "subtitle": f"A{i}", "duration": 1000} for i in range(6)]
    with respx.mock(assert_all_called=False) as r:
        r.get(f"https://open.spotify.com/embed/playlist/{PID}").mock(return_value=httpx.Response(200, text=embed_html(tracks)))
        r.get("https://api.deezer.com/search").mock(return_value=httpx.Response(200, json={"data": []}))
        with pytest.raises(PipelineError) as e:
            await run_analysis(svc, PID, 200_000, 10, Reporter(store, job.job_id))
    await svc.aclose()
    assert e.value.code == "no_previews"


async def test_bogus_autocorrect_and_deezer_fans_are_extra_mainstream_signals(settings, wavs):
    c = Counter()
    job = await run_once(settings, wavs, c, bogus={"Cand Ninety": ("Cand Ninetyy", 2_000, 3_000_000)},
                         dz_fans={"Noise Merchant": 900_000})
    assert job.status == "done", job.error
    assert {r.artist for r in job.results} == {"Cand Onetwenty"}
    assert job.stats.filtered_mainstream == 3  # Big Star (Last.fm), Cand Ninety (exact-name page), Noise Merchant (Deezer)


async def test_listener_floor_drops_too_niche_artists(settings, wavs):
    CANDS["Tiny Act"] = (19_000, "cand90", 109)  # under the 20K floor at a 200K cap
    try:
        job = await run_once(settings, wavs, Counter())
    finally:
        del CANDS["Tiny Act"]
    assert job.status == "done", job.error
    assert "Tiny Act" not in {r.artist for r in job.results}
    assert "Noise Merchant" in {r.artist for r in job.results}  # exactly at the floor (20K) is kept


CORE = "Core Artist"
DEEP_ROWS = [  # Deezer popularity order; ranks are Deezer's log-like popularity
    ("Big Hit", 900_000, "cand90", None),             # top 5 = hits
    ("Second Hit", 880_000, "cand90", None),
    ("Third Hit", 860_000, "cand90", None),
    ("Fourth Hit", 850_000, "cand90", None),
    ("Fifth Hit", 840_000, "cand90", None),
    ("Big Hit (Remix)", 300_000, "cand90", None),     # version of a hit
    ("Track 0", 300_000, "cand90", None),             # already in the playlist
    ("Too Popular", 700_000, "cand90", None),         # > 60% of the best rank
    ("Quiet One (Live at the Roxy)", 300_000, "cand90", None),  # live take
    ("Quiet One", 400_000, "cand90", None),           # deep cut
    ("Quiet One - 2019 Remaster", 380_000, "cand90", None),     # version of the same song
    ("Somebody Else's Song", 300_000, "cand90", "Other Artist"),  # feature on another artist's track
    ("Other Side", 350_000, "cand120", None),          # deep cut
    ("Third Deep", 300_000, "cand90", None),           # deep cut (3rd analyzed)
    ("Fourth Deep", 250_000, "cand90", None),          # beyond DEEP_CUT_CLIPS
]


async def test_deep_cuts_from_core_playlist_artist(settings, wavs):
    c = Counter()
    artists = [CORE, CORE, CORE] + [f"Pl{chr(65 + i)}" for i in range(3, 12)]
    job = await run_once(settings, wavs, c, artists=artists, deep_tops={CORE: (900, DEEP_ROWS)})
    assert job.status == "done", job.error
    deep = [r for r in job.results if r.source == "deep_cut"]
    assert 1 <= len(deep) <= 2, "at most 2 deep cuts per artist"
    titles = {r.title for r in deep}
    assert titles <= {"Quiet One", "Other Side", "Third Deep"}, titles
    for r in deep:
        assert r.artist == CORE
        assert r.why.startswith(f"Deep cut from {CORE} · ") and len(r.why) <= 90, r.why
        assert r.listeners == 1_000  # the artist's Last.fm listeners; the listener cap does not apply
    assert CORE not in {r.artist for r in job.results if r.source != "deep_cut"}
    assert c["dz"] == 3 + 3, "3 discovery clips + 3 deep-cut clips analyzed"
    n_artists = len({r.artist for r in job.results})
    head = [r.artist for r in job.results[:n_artists]]
    assert len(head) == len(set(head)), "every artist appears once before any artist repeats"
    assert [r.rank for r in job.results] == list(range(1, len(job.results) + 1))
    JobResponse.model_validate(job.model_dump(mode="json"))
