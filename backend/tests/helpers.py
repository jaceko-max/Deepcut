"""Shared fixtures for building fake Spotify/Last.fm/Deezer worlds with respx."""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import respx

FIXTURES = Path(__file__).parent / "fixtures"
PID = "37i9dQZF1DXcBWIGoYBM5M"


def embed_html(tracks: list[dict], name="Test Mix", owner="tester") -> str:
    entity = {"name": name, "subtitle": owner, "coverArt": {"sources": [{"url": "https://i.scdn.co/image/x", "width": 640, "height": 640}]},
              "trackList": tracks}
    data = {"props": {"pageProps": {"state": {"data": {"entity": entity}}}}}
    return f'<html><script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script></html>'


def json_resp(d, status=200):
    return httpx.Response(status, json=d)
