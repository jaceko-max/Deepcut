"""Apple Music playlists: link parsing, share-page parsing, API routing."""
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from app import main
from app.clients.apple import AppleMusicClient, apple_page_url, parse_apple_html, parse_apple_playlist
from app.clients.http import HttpClient
from app.clients.spotify import PlaylistNotFound, PlaylistUnavailable
from app.jobs import JobStore, Reporter
from app.pipeline import build_services, run_analysis

FIXTURE = (Path(__file__).parent / "fixtures" / "apple_playlist.html").read_text(encoding="utf-8")
PL = "pl.f4d106fed2bd41149aaacabb233eb5eb"
URL = f"https://music.apple.com/us/playlist/todays-hits/{PL}"


@pytest.mark.parametrize("text", [URL, URL + "?l=en-US", f"music.apple.com/us/playlist/{PL}", f"  {URL}/  ",
                                  "https://music.apple.com/gb/playlist/my-mix/pl.u-AbCd1234-xYz"])
def test_parse_apple_links(text):
    got = parse_apple_playlist(text)
    assert got and got[1].startswith("pl.")


@pytest.mark.parametrize("text", ["", "https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
                                  "https://evil.example/music.apple.com/us/playlist/x/" + PL,
                                  "https://music.apple.com.evil.example/us/playlist/x/" + PL,
                                  "https://music.apple.com/us/album/x/12345", "https://music.apple.com/us/playlist/x/notapl"])
def test_reject_other_links(text):
    assert parse_apple_playlist(text) is None


def test_fetch_url_is_rebuilt_from_validated_parts():
    cc, pid = parse_apple_playlist(URL + "?x=../../etc")
    assert apple_page_url(cc, pid) == f"https://music.apple.com/us/playlist/playlist/{PL}"


def test_parse_page():
    d = parse_apple_html(FIXTURE, PL)
    assert d.source == "apple_music" and d.name == "Today’s Hits" and d.owner == "Apple Music Hits"
    assert d.image_url and d.image_url.startswith("https://") and "{" not in d.image_url
    assert len(d.tracks) == 8
    t = d.tracks[0]
    assert (t.title, t.primary_artist, t.preview_url) == ("stupid song", "Olivia Rodrigo", None)
    assert t.duration_ms and t.spotify_id.startswith("apple")
    assert ["Drake", "Don Toliver"] in [x.artists for x in d.tracks]  # featured credits kept, primary first


@pytest.mark.parametrize("html", ["<html></html>", '<script type="application/json" id="serialized-server-data">{"data":[]}</script>'])
def test_unrecognized_page(html):
    with pytest.raises(PlaylistUnavailable):
        parse_apple_html(html, PL)


async def test_client_statuses():
    async with httpx.AsyncClient() as c:
        cl = AppleMusicClient(HttpClient(c))
        with respx.mock() as r:
            r.get(apple_page_url("us", PL)).mock(return_value=httpx.Response(200, text=FIXTURE))
            assert len((await cl.fetch_playlist("us", PL)).tracks) == 8
        with respx.mock() as r:
            r.get(apple_page_url("us", PL)).mock(return_value=httpx.Response(404))
            with pytest.raises(PlaylistNotFound):
                await cl.fetch_playlist("us", PL)


async def test_private_apple_playlist_reports_not_found(settings):
    svc = build_services(settings)
    store = JobStore(settings)
    job = store.create(200_000, 10)
    try:
        with respx.mock(assert_all_called=False) as r:
            r.get(apple_page_url("us", PL)).mock(return_value=httpx.Response(404))
            await run_analysis(svc, None, 200_000, 10, Reporter(store, job.job_id), apple=("us", PL))
    except Exception as e:  # PipelineError is turned into a job error by the runner
        assert getattr(e, "code", None) == "playlist_not_found"
    else:
        raise AssertionError("expected playlist_not_found")
    finally:
        await svc.aclose()


def test_api_routes_apple_links(settings, monkeypatch):
    seen = {}

    async def fake(svc, pid, max_listeners, limit, rep, tracks=None, apple=None):
        seen.update(pid=pid, apple=apple)

    monkeypatch.setattr(main, "run_analysis", fake)
    with TestClient(main.create_app(settings)) as c:
        r = c.post("/api/analyze", json={"playlist_url": URL})
        assert r.status_code == 202
        r = c.post("/api/analyze", json={"playlist_url": "https://example.com/nope"})
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_playlist_url"
    assert seen == {"pid": None, "apple": ("us", PL)}
