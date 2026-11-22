"""Opt-in network tests: `pytest -m live`."""
import httpx
import pytest

from app.clients.spotify import SpotifyClient, parse_playlist_id
from app.clients.http import HttpClient
from app.config import Settings

pytestmark = pytest.mark.live


async def test_live_embed_playlist():
    async with httpx.AsyncClient(timeout=15) as c:
        d = await SpotifyClient(HttpClient(c), Settings()).fetch_playlist("37i9dQZF1DXcBWIGoYBM5M")
    assert d.tracks and d.name
