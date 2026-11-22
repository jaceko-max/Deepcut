import json

import httpx
import pytest
import respx

from app.cache import Cache
from app.clients.deezer import DeezerClient, deezer_body_rate_limited
from app.clients.http import HttpClient, UpstreamError
from app.clients.lastfm import LastfmClient
from app.clients.spotify import PlaylistNotFound, SpotifyClient, parse_embed_html
from app.config import Settings
from tests.helpers import FIXTURES, PID, json_resp


def dz_track(i=1, title="Song", artist="Band", preview="https://cdn/p.mp3"):
    return {"id": i, "title": title, "duration": 200, "preview": preview, "link": f"https://www.deezer.com/track/{i}",
            "artist": {"id": 9, "name": artist}, "album": {"title": "Alb", "cover_medium": "https://img"}}


@pytest.fixture
async def http():
    async with httpx.AsyncClient() as c:
        yield c


@pytest.fixture
def cache(tmp_path):
    return Cache(str(tmp_path / "c"))


def dz(http, cache):
    return DeezerClient(HttpClient(http, retry_on_body=deezer_body_rate_limited), cache)


@respx.mock
async def test_deezer_quota_error_is_retried(http, cache):
    route = respx.get("https://api.deezer.com/search").mock(side_effect=[
        json_resp({"error": {"type": "Exception", "message": "Quota limit exceeded", "code": 4}}),
        json_resp({"data": [dz_track(5, "Hello World", "The Band")]}),
    ])
    t = await dz(http, cache).find_track("The Band", "Hello World")
    assert t and t.id == 5
    assert route.call_count == 2


@respx.mock
async def test_deezer_quota_exhausted_raises(http, cache):
    respx.get("https://api.deezer.com/search").mock(
        return_value=json_resp({"error": {"code": 4, "message": "Quota"}}))
    with pytest.raises(UpstreamError):
        await dz(http, cache).find_track("A", "B")


@respx.mock
async def test_deezer_isrc_miss_falls_back_to_search(http, cache):
    respx.get("https://api.deezer.com/track/isrc:USXXX1").mock(
        return_value=json_resp({"error": {"type": "DataException", "message": "no data", "code": 800}}))
    s = respx.get("https://api.deezer.com/search").mock(
        return_value=json_resp({"data": [dz_track(7, "Night Drive", "Some Artist")]}))
    t = await dz(http, cache).find_track("Some Artist", "Night Drive", isrc="USXXX1")
    assert t.id == 7 and s.called


@respx.mock
async def test_deezer_unreadable_isrc_hit_is_a_miss(http, cache):
    respx.get("https://api.deezer.com/track/isrc:BAD").mock(
        return_value=json_resp({"id": 1, "readable": False, "preview": ""}))
    respx.get("https://api.deezer.com/search").mock(return_value=json_resp({"data": []}))
    assert await dz(http, cache).find_track("A", "B", isrc="BAD") is None


@respx.mock
async def test_deezer_rejects_poor_matches_and_duration(http, cache):
    respx.get("https://api.deezer.com/search").mock(return_value=json_resp({"data": [
        dz_track(1, "Totally Different", "Other Person"), dz_track(2, "Night Drive", "Some Artist")]}))
    d = dz(http, cache)
    assert (await d.find_track("Some Artist", "Night Drive")).id == 2
    assert await d.find_track("Some Artist", "Night Drive", duration_ms=300_000) is None  # 200 s vs 300 s


@respx.mock
async def test_deezer_artist_top_and_empty_preview(http, cache):
    respx.get("https://api.deezer.com/search/artist").mock(return_value=json_resp({"data": [
        {"id": 3, "name": "Not Them"}, {"id": 4, "name": "Café Tacvba"}]}))
    respx.get("https://api.deezer.com/artist/4/top").mock(return_value=json_resp({"data": [
        dz_track(1, "A"), dz_track(2, "B", preview="")]}))
    d = dz(http, cache)
    assert await d.find_artist_id("cafe tacvba") == 4
    tops = await d.artist_top(4, 3)
    assert [t.id for t in tops] == [1]


@respx.mock
async def test_lastfm_listeners_string_to_int_and_cache(http, cache):
    route = respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        return_value=json_resp({"artist": {"name": "X", "stats": {"listeners": "48210", "playcount": "9"}}}))
    lf = LastfmClient(HttpClient(http), cache, "k")
    assert await lf.artist_listeners("X") == 48210
    assert await lf.artist_listeners("X") == 48210
    assert route.call_count == 1  # second call served from SQLite


@respx.mock
async def test_lastfm_not_found_and_bad_key(http, cache):
    respx.get("https://ws.audioscrobbler.com/2.0/", params={"method": "artist.getInfo"}).mock(
        return_value=json_resp({"error": 6, "message": "The artist you supplied could not be found"}))
    lf = LastfmClient(HttpClient(http), cache, "k")
    assert await lf.artist_listeners("Nobody") is None
    respx.get("https://ws.audioscrobbler.com/2.0/", params={"method": "artist.getSimilar"}).mock(
        return_value=json_resp({"error": 10, "message": "Invalid API key"}))
    with pytest.raises(UpstreamError):
        await lf.artist_similar("X")


@respx.mock
async def test_lastfm_parsers(http, cache):
    def side(request):
        m = request.url.params["method"]
        return json_resp({
            "artist.getSimilar": {"similarartists": {"artist": [{"name": "A", "match": "0.9"}]}},
            "track.getSimilar": {"similartracks": {"track": [{"name": "T", "match": "0.5", "artist": {"name": "B"}}]}},
            "artist.getTopTags": {"toptags": {"tag": [{"name": "rock", "count": 100}]}},
            "tag.getTopArtists": {"topartists": {"artist": [{"name": "C"}]}},
        }[m])
    respx.get("https://ws.audioscrobbler.com/2.0/").mock(side_effect=side)
    lf = LastfmClient(HttpClient(http), cache, "k")
    assert await lf.artist_similar("x") == [("A", 0.9)]
    assert await lf.track_similar("x", "y") == [("B", "T", 0.5)]
    assert await lf.artist_top_tags("x") == [("rock", 100)]
    assert await lf.tag_top_artists("rock") == ["C"]


def test_parse_embed_fixture():
    html = (FIXTURES / "spotify_embed.html").read_text()
    d = parse_embed_html(html, PID)
    assert d.source == "spotify_embed" and d.name and len(d.tracks) >= 40
    assert all(t.spotify_id and t.title and t.artists for t in d.tracks)
    assert sum(1 for t in d.tracks if t.preview_url and t.preview_url.startswith("https://")) >= 40
    assert all("\xa0" not in a for t in d.tracks for a in t.artists)


@respx.mock
async def test_spotify_api_without_items_falls_back_to_embed(http):
    st = Settings(spotify_client_id="id", spotify_client_secret="sec", spotify_playlist_source="auto")
    respx.post("https://accounts.spotify.com/api/token").mock(return_value=json_resp({"access_token": "tok"}))
    respx.get(f"https://api.spotify.com/v1/playlists/{PID}").mock(
        return_value=json_resp({"name": "Meta only", "owner": {"display_name": "x"}, "images": []}))
    respx.get(f"https://open.spotify.com/embed/playlist/{PID}").mock(
        return_value=httpx.Response(200, text=(FIXTURES / "spotify_embed.html").read_text()))
    d = await SpotifyClient(HttpClient(http), st).fetch_playlist(PID)
    assert d.source == "spotify_embed" and len(d.tracks) >= 40


@respx.mock
async def test_spotify_api_with_items_is_used(http):
    st = Settings(spotify_client_id="id", spotify_client_secret="sec")
    respx.post("https://accounts.spotify.com/api/token").mock(return_value=json_resp({"access_token": "tok"}))
    respx.get(f"https://api.spotify.com/v1/playlists/{PID}").mock(return_value=json_resp({
        "name": "Mine", "owner": {"display_name": "me"}, "images": [{"url": "https://img"}],
        "items": {"total": 1, "items": [{"item": {"type": "track", "id": "t1", "name": "Song", "duration_ms": 1000,
                                                   "artists": [{"name": "Art"}], "external_ids": {"isrc": "ISRC1"}}}]}}))
    d = await SpotifyClient(HttpClient(http), st).fetch_playlist(PID)
    assert d.source == "spotify_api" and d.tracks[0].isrc == "ISRC1" and d.owner == "me"


@respx.mock
async def test_spotify_embed_only_without_credentials_and_404(http):
    st = Settings()
    route = respx.get(f"https://open.spotify.com/embed/playlist/{PID}").mock(return_value=httpx.Response(404))
    with pytest.raises(PlaylistNotFound):
        await SpotifyClient(HttpClient(http), st).fetch_playlist(PID)
    assert route.called


@respx.mock
async def test_lastfm_only_not_found_is_cached(http, cache):
    route = respx.get("https://ws.audioscrobbler.com/2.0/").mock(
        return_value=json_resp({"error": 6, "message": "not found"}))
    lf = LastfmClient(HttpClient(http), cache, "k")
    assert await lf.artist_listeners("Nobody") is None
    assert await lf.artist_listeners("Nobody") is None
    assert route.call_count == 1
    for code in (8, 16, 29, 7):
        n0 = route.call_count
        route.mock(return_value=json_resp({"error": code, "message": "x"}))
        for _ in range(2):
            with pytest.raises(UpstreamError):
                await lf.artist_similar(f"Y{code}")
        assert route.call_count - n0 == 2, code  # not cached
    route.mock(return_value=json_resp({"similarartists": {"artist": [{"name": "A", "match": "0.9"}]}}))
    assert await lf.artist_similar("Y8") == [("A", 0.9)]  # recovers: nothing stale was stored


@respx.mock
async def test_spotify_token_refresh_on_expiry_and_401(http, monkeypatch):
    import app.clients.spotify as sp
    st = Settings(spotify_client_id="id", spotify_client_secret="sec")
    tok = respx.post("https://accounts.spotify.com/api/token").mock(
        return_value=json_resp({"access_token": "tok", "expires_in": 3600}))
    body = {"name": "P", "tracks": {"items": [{"track": {"id": "t1", "name": "S", "type": "track", "artists": [{"name": "A"}]}}]}}
    api = respx.get(f"https://api.spotify.com/v1/playlists/{PID}").mock(return_value=json_resp(body))
    c = SpotifyClient(HttpClient(http), st)
    await c.fetch_playlist(PID)
    await c.fetch_playlist(PID)
    assert tok.call_count == 1  # cached while valid
    now = sp.time.monotonic()
    monkeypatch.setattr(sp.time, "monotonic", lambda: now + 3600 - 30)  # inside the 60 s margin
    await c.fetch_playlist(PID)
    assert tok.call_count == 2
    # 401 once -> refresh and retry
    api.mock(side_effect=[httpx.Response(401), json_resp(body)])
    d = await c.fetch_playlist(PID)
    assert tok.call_count == 3 and d.source == "spotify_api"


@respx.mock
async def test_spotify_auto_mode_falls_back_on_upstream_error_and_bad_json(http):
    st = Settings(spotify_client_id="id", spotify_client_secret="sec", spotify_playlist_source="auto")
    embed = respx.get(f"https://open.spotify.com/embed/playlist/{PID}").mock(
        return_value=httpx.Response(200, text=(FIXTURES / "spotify_embed.html").read_text()))
    # token endpoint failing repeatedly -> UpstreamError from HttpClient
    respx.post("https://accounts.spotify.com/api/token").mock(return_value=httpx.Response(503))
    d = await SpotifyClient(HttpClient(http), st).fetch_playlist(PID)
    assert d.source == "spotify_embed"
    # 200 with a non-JSON body from the playlist endpoint
    respx.post("https://accounts.spotify.com/api/token").mock(return_value=json_resp({"access_token": "t", "expires_in": 3600}))
    respx.get(f"https://api.spotify.com/v1/playlists/{PID}").mock(return_value=httpx.Response(200, text="<html>nope"))
    d = await SpotifyClient(HttpClient(http), st).fetch_playlist(PID)
    assert d.source == "spotify_embed" and embed.call_count == 2
    # token endpoint returning garbage
    respx.post("https://accounts.spotify.com/api/token").mock(return_value=httpx.Response(200, text="garbage"))
    d = await SpotifyClient(HttpClient(http), st).fetch_playlist(PID)
    assert d.source == "spotify_embed"
