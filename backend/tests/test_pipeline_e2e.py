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
}


def world(wavs, counters):
    tracks = []
    previews = {}
    for i in range(12):
        vibe = "dark90" if i < 6 else "bright120"
        key = f"t{i:02d}"
        previews[key] = wavs[vibe][i % 3]
        tracks.append({"uri": f"spotify:track:{key}{'x' * 19}", "title": f"Track {i}", "subtitle": f"Pl{chr(65 + i)}",
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
                {"name": n, "match": "0.8"} for n in CANDS]}})
        if m == "track.getSimilar":
            return httpx.Response(200, json={"similartracks": {"track": []}})
        if m == "artist.getTopTags":
            return httpx.Response(200, json={"toptags": {"tag": [{"name": "seen live", "count": 100}, {"name": "trip hop", "count": 50}]}})
        if m == "tag.getTopArtists":
            return httpx.Response(200, json={"topartists": {"artist": [{"name": "Cand Ninety"}]}})
        if m == "artist.getInfo":
            return httpx.Response(200, json={"artist": {"stats": {"listeners": str(CANDS[p["artist"]][0])}}})
        return httpx.Response(200, json={})
    router.get(LFM).mock(side_effect=lfm)

    def dz_artist(request):
        q = request.url.params["q"]
        return httpx.Response(200, json={"data": [{"id": CANDS[q][2], "name": q}]})
    router.get("https://api.deezer.com/search/artist").mock(side_effect=dz_artist)

    def dz_top(request):
        did = int(request.url.path.split("/")[2])
        name = next(n for n, v in CANDS.items() if v[2] == did)
        return httpx.Response(200, json={"data": [{
            "id": did * 10, "title": f"{name} Song", "duration": 200, "link": f"https://www.deezer.com/track/{did * 10}",
            "preview": f"https://cdnt-preview.dzcdn.net/{did}.mp3?hdnea=x", "artist": {"id": did, "name": name},
            "album": {"title": "Alb", "cover_medium": "https://img/c.jpg"}}]})
    router.get(url__regex=r"https://api\.deezer\.com/artist/\d+/top").mock(side_effect=dz_top)

    def dz_preview(request):
        counters["dz"] += 1
        did = int(re.search(r"/(\d+)\.mp3", request.url.path).group(1))
        key = next(v[1] for v in CANDS.values() if v[2] == did)
        return httpx.Response(200, content=wavs[key])
    router.get(url__regex=r"https://cdnt-preview\.dzcdn\.net/\d+\.mp3").mock(side_effect=dz_preview)
    return router


async def run_once(settings, wavs, counters, limit=10):
    svc = build_services(settings)
    store = JobStore(settings)
    job = store.create(200_000, limit)
    rep = Reporter(store, job.job_id)
    try:
        with world(wavs, counters):
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
        assert r.preview_proxy_url == f"/api/preview/{r.deezer_id}"
        assert r.spotify_search_url.startswith("https://open.spotify.com/search/") and " " not in r.spotify_search_url
        assert len(r.why) <= 90 and 0 <= r.match_score <= 100
        assert r.listeners == CANDS[r.artist][0]
    assert by_artist["Cand Ninety"].vibe_id != by_artist["Cand Onetwenty"].vibe_id
    JobResponse.model_validate(job.model_dump(mode="json"))


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
