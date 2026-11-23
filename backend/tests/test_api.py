import time

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import app.main as main
from app.main import create_app
from app.config import Settings
from app.models import Result, Stats
from tests.helpers import PID

URL = f"https://open.spotify.com/playlist/{PID}?si=zz"


async def fake_pipeline(svc, pid, max_listeners, limit, rep):
    import asyncio
    assert pid == PID
    rep.stage("analyzing_playlist", 0.3, "working")
    await asyncio.sleep(0.05)
    rep.stage("ranking", 0.2, "lower progress must not regress")
    rep.done([], Stats())


@pytest.fixture
def client(settings, monkeypatch):
    monkeypatch.setattr(main, "run_analysis", fake_pipeline)
    with TestClient(create_app(settings)) as c:
        yield c


def poll(client, jid, timeout=5):
    end = time.time() + timeout
    seen = []
    while time.time() < end:
        j = client.get(f"/api/jobs/{jid}").json()
        seen.append(j["progress"])
        if j["status"] in ("done", "error"):
            return j, seen
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok", "version": "0.1.0"}


def test_analyze_and_poll_to_done(client):
    r = client.post("/api/analyze", json={"playlist_url": URL})
    assert r.status_code == 202
    body = r.json()
    assert body["status"] == "queued" and body["poll_url"] == f"/api/jobs/{body['job_id']}"
    j, seen = poll(client, body["job_id"])
    assert j["status"] == "done" and j["stage"] == "done" and j["progress"] == 1.0
    assert j["params"] == {"max_listeners": 500000, "limit": 20}
    assert seen == sorted(seen)
    assert j["results"] == [] and j["error"] is None and j["warnings"] == []


def test_unknown_job_404_shape(client):
    r = client.get("/api/jobs/nope")
    assert r.status_code == 404
    assert r.json() == {"error": {"code": "job_not_found", "message": r.json()["error"]["message"]}}


def test_bad_bodies_are_400(client):
    for body in [{}, {"playlist_url": URL, "limit": 2}, {"playlist_url": URL, "max_listeners": 5},
                 {"playlist_url": URL, "limit": "x"}]:
        r = client.post("/api/analyze", json=body)
        assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request", body
    r = client.post("/api/analyze", content="not json", headers={"content-type": "application/json"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"
    r = client.post("/api/analyze", json={"playlist_url": "https://example.com/x"})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_playlist_url"


def test_fourth_post_is_429(client):
    for _ in range(3):
        assert client.post("/api/analyze", json={"playlist_url": PID}).status_code == 202
    r = client.post("/api/analyze", json={"playlist_url": PID})
    assert r.status_code == 429
    assert r.json()["error"]["code"] == "too_many_jobs"
    assert int(r.headers["retry-after"]) > 0
    # a different client IP is unaffected
    assert client.post("/api/analyze", json={"playlist_url": PID}, headers={"x-forwarded-for": "9.9.9.9"}).status_code == 202


def test_server_busy_503(settings, monkeypatch):
    import asyncio

    async def never(svc, pid, m, l, rep):
        await asyncio.sleep(30)
    monkeypatch.setattr(main, "run_analysis", never)
    settings.max_concurrent_jobs = 1
    settings.max_queued_jobs = 2
    settings.ip_job_limit = 100
    with TestClient(create_app(settings)) as c:
        codes = [c.post("/api/analyze", json={"playlist_url": PID}).status_code for _ in range(5)]
    # 1 running + 2 queued accepted, then busy
    assert codes[:3] == [202, 202, 202] and 503 in codes


def test_cors_header(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_unknown_route_uses_error_shape(client):
    r = client.get("/api/nothing")
    assert r.status_code == 404 and "error" in r.json()


def _hold_ids(client, *ids):
    """Put results with these Deezer ids into a job the store currently holds."""
    from types import SimpleNamespace
    store = client.app.state.store
    job = store.create(20, 20)
    store.update(job.job_id, results=[SimpleNamespace(deezer_id=i) for i in ids])


def test_preview_redirect_and_404(client):
    _hold_ids(client, 42, 43)
    with respx.mock:
        respx.get("https://api.deezer.com/track/42").mock(return_value=httpx.Response(200, json={
            "id": 42, "title": "T", "preview": "https://cdnt-preview.dzcdn.net/fresh.mp3?hdnea=new", "artist": {"id": 1, "name": "A"}}))
        respx.get("https://api.deezer.com/track/43").mock(return_value=httpx.Response(200, json={"id": 43, "readable": False, "preview": ""}))
        r = client.get("/api/preview/42", follow_redirects=False)
        assert r.status_code == 307 and r.headers["location"].startswith("https://cdnt-preview.dzcdn.net/fresh.mp3")
        r = client.get("/api/preview/43", follow_redirects=False)
        assert r.status_code == 404 and r.json()["error"]["code"] == "preview_not_found"


def test_preview_unknown_id_404_without_upstream_call(client):
    with respx.mock(assert_all_called=False) as m:
        route = m.get(url__regex=r"https://api\.deezer\.com/.*").mock(return_value=httpx.Response(200, json={}))
        r = client.get("/api/preview/999", follow_redirects=False)
    assert r.status_code == 404 and r.json()["error"]["code"] == "preview_not_found"
    assert route.call_count == 0


def test_preview_rate_limited(settings, monkeypatch):
    monkeypatch.setattr(main, "run_analysis", fake_pipeline)
    settings.preview_ip_limit = 3
    with TestClient(create_app(settings)) as c:
        codes = [c.get("/api/preview/1", follow_redirects=False).status_code for _ in range(5)]
        r = c.get("/api/preview/1", follow_redirects=False)
    assert codes == [404, 404, 404, 429, 429]
    assert r.json()["error"]["code"] == "too_many_requests" and int(r.headers["retry-after"]) > 0


def test_xff_spoofing_left_entry_does_not_bypass_limit(client):
    # proxy appends the real peer last; clients can only forge entries to its left
    for i in range(3):
        r = client.post("/api/analyze", json={"playlist_url": PID}, headers={"x-forwarded-for": f"1.1.1.{i}, 5.5.5.5"})
        assert r.status_code == 202
    r = client.post("/api/analyze", json={"playlist_url": PID}, headers={"x-forwarded-for": "1.1.1.99, 5.5.5.5"})
    assert r.status_code == 429


def test_global_job_limit_429_with_retry_after(settings, monkeypatch):
    monkeypatch.setattr(main, "run_analysis", fake_pipeline)
    settings.global_job_limit = 4
    with TestClient(create_app(settings)) as c:
        codes = [c.post("/api/analyze", json={"playlist_url": PID}, headers={"x-forwarded-for": f"7.7.7.{i}"}).status_code
                 for i in range(6)]
        r = c.post("/api/analyze", json={"playlist_url": PID}, headers={"x-forwarded-for": "8.8.8.8"})
    assert codes == [202] * 4 + [429, 429]
    assert r.json()["error"]["code"] == "too_many_jobs" and int(r.headers["retry-after"]) > 0


def test_server_busy_does_not_consume_ip_slot(settings, monkeypatch):
    import asyncio

    async def never(svc, pid, m, l, rep):
        await asyncio.sleep(30)
    monkeypatch.setattr(main, "run_analysis", never)
    settings.max_concurrent_jobs = 1
    settings.max_queued_jobs = 1
    with TestClient(create_app(settings)) as c:
        c.app.state.limiter.limit = 100
        c.post("/api/analyze", json={"playlist_url": PID})
        c.post("/api/analyze", json={"playlist_url": PID})
        before = len(c.app.state.limiter.hits["testclient"])
        gbefore = len(c.app.state.global_limiter.hits["*"])
        r = c.post("/api/analyze", json={"playlist_url": PID})
        assert r.status_code == 503
        assert len(c.app.state.limiter.hits["testclient"]) == before
        assert len(c.app.state.global_limiter.hits["*"]) == gbefore


def test_cors_exposes_retry_after(client):
    r = client.get("/api/health", headers={"Origin": "http://localhost:3000"})
    assert r.headers["access-control-expose-headers"] == "Retry-After"


def test_oversized_content_length_413(client):
    r = client.post("/api/analyze", content=b"x" * 20000, headers={"content-type": "application/json", "origin": "http://localhost:3000"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "invalid_request"
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_oversized_chunked_body_413(client):
    def gen():
        for _ in range(40):
            yield b"x" * 1000
    r = client.post("/api/analyze", content=gen(), headers={"content-type": "application/json"})
    assert r.status_code == 413 and r.json()["error"]["code"] == "invalid_request"


def test_long_playlist_url_400(client):
    r = client.post("/api/analyze", json={"playlist_url": "https://open.spotify.com/" + "a" * 600})
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


def test_unhandled_500_has_cors_headers(settings, monkeypatch):
    async def boom(*a, **k):
        raise RuntimeError("secret detail")
    monkeypatch.setattr(main, "parse_playlist_id", lambda s: (_ for _ in ()).throw(RuntimeError("secret detail")))
    with TestClient(create_app(settings), raise_server_exceptions=False) as c:
        r = c.post("/api/analyze", json={"playlist_url": PID}, headers={"origin": "http://localhost:3000"})
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "internal_error", "message": "Internal server error"}}
    assert r.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_docs_disabled_by_default_and_enabled_by_flag(settings):
    with TestClient(create_app(settings)) as c:
        for path in ("/docs", "/redoc", "/openapi.json"):
            assert c.get(path).status_code == 404
    settings.enable_docs = True
    with TestClient(create_app(settings)) as c:
        assert c.get("/openapi.json").status_code == 200


def test_health_503_when_pool_failed(settings):
    from types import SimpleNamespace
    svc = SimpleNamespace(executor=SimpleNamespace(failed=True), cache=None)
    with TestClient(create_app(settings, services=svc)) as c:
        r = c.get("/api/health")
        assert r.status_code == 503
        svc.executor.failed = False
        assert c.get("/api/health").status_code == 200


def test_sweeper_purges_expired_http_cache(settings, monkeypatch):
    import asyncio
    from app.cache import Cache
    cache = Cache(settings.cache_dir)
    cache._put_http("old", "x", -10)
    cache._put_http("new", "y", 1000)
    n = asyncio.run(cache.purge_expired())
    assert n == 1 and cache._get_http("new") == "y"


def test_job_error_message_is_generic(settings):
    import asyncio
    from app.jobs import JobRunner, JobStore
    from app.pipeline import PipelineError
    store = JobStore(settings)
    job = store.create(20, 20)

    async def work(rep):
        raise PipelineError("upstream_error", "Last.fm is not responding. Please try again in a minute.",
                            "Last.fm API key rejected: Invalid API key")

    async def go():
        r = JobRunner(store, settings)
        r.submit(job.job_id, work)
        await asyncio.gather(*r.tasks)
    asyncio.run(go())
    j = store.get(job.job_id)
    assert j.error.code == "upstream_error" and "key" not in j.error.message.lower()


def test_cors_origins_trailing_slash_stripped(monkeypatch):
    monkeypatch.setattr("app.config.load_env_file", lambda *a, **k: None)
    monkeypatch.setenv("CORS_ORIGINS", "https://a.app/, https://b.app//,")
    assert Settings.from_env().cors_origins == ["https://a.app", "https://b.app"]


def test_signed_preview_works_without_job_in_store(client, settings):
    from app.signing import preview_path

    with respx.mock:
        respx.get("https://api.deezer.com/track/77").mock(return_value=httpx.Response(200, json={
            "id": 77, "title": "T", "preview": "https://cdnt-preview.dzcdn.net/s.mp3?hdnea=x", "artist": {"id": 1, "name": "A"}}))
        r = client.get(preview_path(settings.preview_secret, 77), follow_redirects=False)
    assert r.status_code == 307 and r.headers["location"].startswith("https://cdnt-preview.dzcdn.net/s.mp3")


def test_forged_preview_signature_rejected_without_upstream_call(client, settings):
    from app.signing import preview_sig

    with respx.mock(assert_all_called=False) as m:
        route = m.get(url__regex=r"https://api\.deezer\.com/.*").mock(return_value=httpx.Response(200, json={}))
        bad = client.get("/api/preview/78?sig=" + "0" * 20, follow_redirects=False)
        other = client.get(f"/api/preview/79?sig={preview_sig(settings.preview_secret, 78)}", follow_redirects=False)
    assert bad.status_code == 404 and other.status_code == 404
    assert route.call_count == 0


def test_cloudflare_client_ip_separates_visitors_behind_render(client):
    # On Render every request carries Render's proxy as the last X-Forwarded-For entry; Cloudflare's
    # CF-Connecting-IP is what tells visitors apart.
    proxy = {"x-forwarded-for": "9.9.9.9, 10.0.0.1"}
    for i in range(3):
        r = client.post("/api/analyze", json={"playlist_url": PID}, headers={**proxy, "cf-connecting-ip": "1.1.1.1"})
        assert r.status_code == 202
    assert client.post("/api/analyze", json={"playlist_url": PID},
                       headers={**proxy, "cf-connecting-ip": "1.1.1.1"}).status_code == 429
    assert client.post("/api/analyze", json={"playlist_url": PID},
                       headers={**proxy, "cf-connecting-ip": "2.2.2.2"}).status_code == 202


def test_true_client_ip_used_when_cf_header_missing():
    from starlette.requests import Request

    scope = {"type": "http", "headers": [(b"true-client-ip", b"3.3.3.3"), (b"x-forwarded-for", b"10.0.0.1")],
             "client": ("127.0.0.1", 1)}
    assert main.client_ip(Request(scope)) == "3.3.3.3"


def test_default_ip_job_limit_is_six():
    from app.config import Settings

    assert Settings().ip_job_limit == 6
