"""Pasted track lists (request validation, pipeline input), the reserve pool and its contract."""
import json
from collections import Counter
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.config import Settings
from app.main import create_app
from app.models import JobResponse, Stats
from tests.test_api import URL, poll
from tests.test_pipeline_e2e import run_once

CONTRACT = Path(__file__).resolve().parents[2] / "contract"
TRACKS = [{"artist": "Low Hum Collective", "title": "Undertow"}, {"artist": "Second Artist", "title": "Another One"}]


@pytest.fixture
def client(settings, monkeypatch):
    seen = {}

    async def pipeline(svc, pid, max_listeners, limit, rep, tracks=None):
        seen.update(pid=pid, tracks=tracks)
        rep.done([], Stats())

    monkeypatch.setattr(main, "run_analysis", pipeline)
    with TestClient(create_app(settings)) as c:
        c.seen = seen
        yield c


def test_tracks_input_accepted_and_runs_without_playlist_url(client):
    r = client.post("/api/analyze", json={"tracks": TRACKS})
    assert r.status_code == 202
    j, _ = poll(client, r.json()["job_id"])
    assert j["status"] == "done" and j["reserve"] == []
    assert client.seen == {"pid": None, "tracks": TRACKS}


def test_exactly_one_of_url_or_tracks_else_400(client):
    bad = [
        {},
        {"playlist_url": URL, "tracks": TRACKS},
        {"tracks": []},
        {"tracks": [{"artist": "", "title": "x"}]},
        {"tracks": [{"artist": "a"}]},
        {"tracks": [{"artist": "a" * 201, "title": "x"}]},
        {"tracks": [{"artist": "a", "title": "t" * 301}]},
        {"tracks": [{"artist": "a", "title": f"t{i}"} for i in range(201)]},
    ]
    for body in bad:
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request", body


def test_200_pasted_tracks_fit_the_body_guard(client):
    tracks = [{"artist": f"Artist Number {i}" + "x" * 60, "title": f"Some Rather Long Song Title {i}" + "y" * 80}
              for i in range(200)]
    assert len(json.dumps({"tracks": tracks})) > 16 * 1024  # would have failed the old 16 KB guard
    assert client.post("/api/analyze", json={"tracks": tracks}).status_code == 202


def test_pasted_playlist_builds_pipeline_input():
    from app.pasted import pasted_playlist
    pl = pasted_playlist([{"artist": "A; B", "title": " Song "}, {"artist": "Tyler, The Creator", "title": "X"}])
    assert pl.source == "pasted" and pl.name == "Pasted playlist" and pl.id.startswith("pasted-")
    assert [t.artists for t in pl.tracks] == [["A", "B"], ["Tyler, The Creator"]]
    assert all(t.preview_url is None for t in pl.tracks) and pl.tracks[0].title == "Song"


def test_pick_reserve_one_per_artist_and_excludes_chosen():
    from app.pasted import pick_reserve
    from app.scoring.score import Scored
    mk = lambda k, a, s: Scored(key=k, artist=a, vibe_id=0, score=s, affinity=0.5)
    pool = [mk(1, "A", 90), mk(2, "B", 80), mk(3, "B", 79), mk(4, "C", 70), mk(5, "D", 60)]
    got = pick_reserve(pool, {1}, {"a"}, n=3)
    assert [s.key for s in got] == [2, 4, 5]  # one per artist, chosen artist A excluded
    got = pick_reserve(pool, {1}, {"a"}, n=10)
    assert [s.key for s in got] == [2, 4, 5, 3]  # then extra tracks fill the remainder


async def test_reserve_pool_continues_ranks_and_excludes_results(settings, wavs):
    job = await run_once(settings, wavs, Counter(), limit=2)
    assert job.status == "done", job.error
    assert len(job.results) == 2 and len(job.reserve) == 1
    assert [r.rank for r in job.reserve] == [3]
    assert not {r.deezer_id for r in job.reserve} & {r.deezer_id for r in job.results}
    for r in job.reserve:  # same shape and filters as the list
        assert r.deezer_id % 10 == 0 and r.listeners <= 200_000 and len(r.why) <= 90
        assert r.preview_proxy_url.startswith(f"/api/preview/{r.deezer_id}?sig=")
    JobResponse.model_validate(job.model_dump(mode="json"))


async def test_reserve_empty_when_everything_fits(settings, wavs):
    job = await run_once(settings, wavs, Counter(), limit=10)
    assert job.reserve == []


def test_contract_example_has_reserve_with_continuing_ranks():
    job = JobResponse.model_validate_json((CONTRACT / "job_done.example.json").read_text())
    assert len(job.reserve) >= 2
    assert [r.rank for r in job.reserve] == list(range(len(job.results) + 1, len(job.results) + len(job.reserve) + 1))
    assert not {r.deezer_id for r in job.reserve} & {r.deezer_id for r in job.results}


def test_reserve_defaults_to_empty_for_older_payloads():
    raw = json.loads((CONTRACT / "job_running.example.json").read_text())
    raw.pop("reserve", None)
    assert JobResponse.model_validate(raw).reserve == []
