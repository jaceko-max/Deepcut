"""FastAPI app: /api/health, /api/analyze, /api/jobs/{id}, /api/preview/{id}."""
from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.clients.http import UpstreamError
from app.clients.spotify import parse_playlist_id
from app.config import VERSION, Settings
from app.jobs import JobRunner, JobStore, ServerBusy
from app.models import AnalyzeRequest, AnalyzeResponse, JobResponse
from app.pipeline import Services, build_services, run_analysis
from app.ratelimit import IPJobLimiter
from app.signing import valid_preview_sig

log = logging.getLogger("deepcut")


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, headers: dict | None = None):
        self.status, self.code, self.message, self.headers = status, code, message, headers or {}


def error_response(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message}}, status_code=status, headers=headers)


def client_ip(request: Request) -> str:
    """The visitor's address. Render sits behind Cloudflare, which overwrites CF-Connecting-IP/True-Client-IP
    with the real client address. Render's own proxy appends *its* address to X-Forwarded-For, so the
    right-most entry there is only a fallback for other hosts (and every left entry is client-spoofable)."""
    for header in ("cf-connecting-ip", "true-client-ip"):
        value = (request.headers.get(header) or "").strip()
        if value:
            return value
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        last = fwd.split(",")[-1].strip()
        if last:
            return last
    return request.client.host if request.client else "unknown"


class GuardMiddleware:
    """Pure ASGI, installed inside CORSMiddleware: rejects oversized bodies (413) and turns unhandled exceptions
    into a 500 JSON response that still passes through CORS (Starlette's own 500 handler sits outside CORS)."""

    def __init__(self, app, max_body_bytes: int):
        self.app, self.max_body = app, max_body_bytes

    async def _send_error(self, send, status: int, code: str, message: str) -> None:
        body = json.dumps({"error": {"code": code, "message": message}}).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode())]})
        await send({"type": "http.response.body", "body": body})

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)
        too_large = "Request body too large."
        for k, v in scope.get("headers", []):
            if k == b"content-length":
                try:
                    if int(v) > self.max_body:
                        return await self._send_error(send, 413, "invalid_request", too_large)
                except ValueError:
                    pass
        received = 0

        async def limited_receive():
            nonlocal received
            msg = await receive()
            if msg["type"] == "http.request":
                received += len(msg.get("body", b""))
                if received > self.max_body:
                    raise HTTPException(413, too_large)
            return msg

        started = False

        async def tracking_send(msg):
            nonlocal started
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, limited_receive, tracking_send)
        except HTTPException as e:
            if started:
                raise
            await self._send_error(send, e.status_code, "invalid_request", str(e.detail))
        except Exception:
            log.exception("unhandled error")
            if started:
                raise
            await self._send_error(send, 500, "internal_error", "Internal server error")


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    logging.basicConfig(level=settings.log_level.upper())
    logging.getLogger("httpx").setLevel(logging.WARNING)  # request URLs contain the Last.fm key

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        svc = services or build_services(settings)
        app.state.svc = svc
        app.state.store = JobStore(settings)
        app.state.runner = JobRunner(app.state.store, settings)
        app.state.limiter = IPJobLimiter(settings.ip_job_limit, settings.ip_job_window_seconds)
        app.state.global_limiter = IPJobLimiter(settings.global_job_limit, settings.ip_job_window_seconds)
        app.state.preview_limiter = IPJobLimiter(settings.preview_ip_limit, settings.preview_window_seconds)
        if not settings.lastfm_api_key:
            log.warning("LASTFM_API_KEY is not set; analysis jobs will fail")

        async def sweeper():
            while True:
                await asyncio.sleep(300)
                app.state.store.sweep()
                app.state.limiter.sweep()
                app.state.global_limiter.sweep()
                app.state.preview_limiter.sweep()
                cache = getattr(svc, "cache", None)
                if cache is not None:
                    try:
                        await cache.purge_expired()
                    except Exception:
                        log.exception("http_cache purge failed")

        task = asyncio.create_task(sweeper())
        try:
            yield
        finally:
            task.cancel()
            if services is None:
                await svc.aclose()

    docs = {} if settings.enable_docs else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    app = FastAPI(title="Deepcut API", version=VERSION, lifespan=lifespan, **docs)
    # Added first = innermost, so its 413/500 responses still receive CORS headers.
    app.add_middleware(GuardMiddleware, max_body_bytes=settings.max_body_bytes)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origins, allow_origin_regex=settings.cors_origin_regex,
        allow_methods=["GET", "POST", "OPTIONS"], allow_headers=["*"], expose_headers=["Retry-After"],
    )

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, e: ApiError):
        return error_response(e.status, e.code, e.message, e.headers)

    @app.exception_handler(RequestValidationError)
    async def _validation(_: Request, e: RequestValidationError):
        first = e.errors()[0] if e.errors() else {}
        loc = ".".join(str(x) for x in first.get("loc", []) if x != "body")
        return error_response(400, "invalid_request", f"Invalid request{': ' + loc if loc else ''}: {first.get('msg', '')}".strip())

    @app.exception_handler(StarletteHTTPException)
    async def _http(_: Request, e: StarletteHTTPException):
        code = "not_found" if e.status_code == 404 else "invalid_request" if e.status_code < 500 else "internal_error"
        return error_response(e.status_code, code, str(e.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_: Request, e: Exception):
        log.exception("unhandled error")
        return error_response(500, "internal_error", "Internal server error")

    @app.get("/api/health")
    async def health(request: Request):
        svc = getattr(request.app.state, "svc", None)
        if getattr(getattr(svc, "executor", None), "failed", False):
            # analysis pool could not be recovered: fail the health check so the host restarts the service
            return error_response(503, "service_unavailable", "Analysis workers are down.")
        return {"status": "ok", "version": VERSION}

    @app.post("/api/analyze", status_code=202, response_model=AnalyzeResponse)
    async def analyze(body: AnalyzeRequest, request: Request):
        pid = parse_playlist_id(body.playlist_url)
        if not pid:
            raise ApiError(400, "invalid_playlist_url", "That doesn't look like a Spotify playlist link.")
        store: JobStore = request.app.state.store
        if store.queued_count() >= settings.max_queued_jobs:
            raise ApiError(503, "server_busy", "The server is busy right now. Please try again in a minute.", {"Retry-After": "30"})
        ip, limiter, glimiter = client_ip(request), request.app.state.limiter, request.app.state.global_limiter
        wait = glimiter.check("*")
        if wait is not None:
            raise ApiError(429, "too_many_jobs", "The service is handling a lot of analyses right now. Please try again in a few minutes.", {"Retry-After": str(wait)})
        wait = limiter.check(ip)
        if wait is not None:
            glimiter.release("*")
            raise ApiError(429, "too_many_jobs", "You've started too many analyses recently. Please wait a few minutes.", {"Retry-After": str(wait)})
        try:
            job = store.create(body.max_listeners, body.limit)
        except ServerBusy:
            limiter.release(ip)  # a rejected request must not use up the client's quota
            glimiter.release("*")
            raise ApiError(503, "server_busy", "The server is busy right now. Please try again in a minute.", {"Retry-After": "30"})
        svc = request.app.state.svc
        request.app.state.runner.submit(
            job.job_id, lambda rep: run_analysis(svc, pid, body.max_listeners, body.limit, rep))
        return AnalyzeResponse(job_id=job.job_id, poll_url=f"/api/jobs/{job.job_id}")

    @app.get("/api/jobs/{job_id}", response_model=JobResponse)
    async def get_job(job_id: str, request: Request):
        job = request.app.state.store.get(job_id)
        if not job:
            raise ApiError(404, "job_not_found", "Job not found or expired. Run the analysis again.")
        return job

    @app.get("/api/preview/{deezer_id}")
    async def preview(deezer_id: int, request: Request, sig: str | None = None):
        wait = request.app.state.preview_limiter.check(client_ip(request))
        if wait is not None:
            raise ApiError(429, "too_many_requests", "Too many preview requests. Please slow down.", {"Retry-After": str(wait)})
        signed = valid_preview_sig(request.app.state.svc.settings.preview_secret, deezer_id, sig)
        if not signed and not request.app.state.store.has_result_id(deezer_id):
            raise ApiError(404, "preview_not_found", "No preview available for this track.")
        svc = request.app.state.svc
        try:
            t = await (getattr(svc, "preview_deezer", None) or svc.deezer).get_track(deezer_id)
        except UpstreamError:
            raise ApiError(502, "upstream_error", "Deezer is not responding.")
        if not t:
            raise ApiError(404, "preview_not_found", "No preview available for this track.")
        return RedirectResponse(t.preview_url, status_code=307)

    return app


app = create_app()
