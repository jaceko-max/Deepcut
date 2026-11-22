"""Environment-driven settings."""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field

VERSION = "0.1.0"


ENV_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")


def load_env_file(path: str = ENV_FILE) -> None:
    """Load KEY=VALUE lines into os.environ without overriding variables that are already set."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.removeprefix("export ").strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


@dataclass
class Settings:
    lastfm_api_key: str = ""
    spotify_client_id: str = ""
    spotify_client_secret: str = ""
    spotify_playlist_source: str = "auto"  # auto | api | embed
    cors_origins: list[str] = field(default_factory=lambda: ["http://localhost:3000"])
    cors_origin_regex: str | None = None
    cache_dir: str = "./.cache"
    max_concurrent_jobs: int = 2
    analysis_workers: int = 2  # 0 = run analysis in a thread (tests)
    job_ttl_seconds: int = 3600
    http_user_agent: str = "Deepcut/0.1"
    log_level: str = "info"
    job_timeout_seconds: int = 420
    max_queued_jobs: int = 10
    ip_job_limit: int = 3
    ip_job_window_seconds: int = 600
    global_job_limit: int = 30  # max job creations across all clients per ip_job_window_seconds
    preview_ip_limit: int = 60  # /api/preview requests per IP per preview_window_seconds
    preview_window_seconds: int = 60
    max_body_bytes: int = 16 * 1024
    enable_docs: bool = False
    preview_secret: str = field(default_factory=lambda: secrets.token_hex(16))  # set PREVIEW_SECRET so links survive restarts
    rate_limit_scale: float = 1.0  # multiplies upstream token-bucket capacity (tests raise it)

    @property
    def has_spotify_credentials(self) -> bool:
        return bool(self.spotify_client_id and self.spotify_client_secret)

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file()
        env = os.environ.get
        origins = [o.strip().rstrip("/") for o in env("CORS_ORIGINS", "http://localhost:3000").split(",") if o.strip().rstrip("/")]
        source = env("SPOTIFY_PLAYLIST_SOURCE", "auto").strip().lower()
        if source not in ("auto", "api", "embed"):
            source = "auto"
        return cls(
            lastfm_api_key=env("LASTFM_API_KEY", "").strip(),
            spotify_client_id=env("SPOTIFY_CLIENT_ID", "").strip(),
            spotify_client_secret=env("SPOTIFY_CLIENT_SECRET", "").strip(),
            spotify_playlist_source=source,
            cors_origins=origins,
            cors_origin_regex=(env("CORS_ORIGIN_REGEX") or None),
            cache_dir=env("CACHE_DIR", "./.cache"),
            max_concurrent_jobs=_int("MAX_CONCURRENT_JOBS", 2),
            analysis_workers=_int("ANALYSIS_WORKERS", 2),
            job_ttl_seconds=_int("JOB_TTL_SECONDS", 3600),
            job_timeout_seconds=_int("JOB_TIMEOUT_SECONDS", 420),
            http_user_agent=env("HTTP_USER_AGENT", "Deepcut/0.1"),
            log_level=env("LOG_LEVEL", "info"),
            global_job_limit=_int("GLOBAL_JOB_LIMIT", 30),
            enable_docs=env("ENABLE_DOCS", "").strip().lower() in ("1", "true", "yes"),
            preview_secret=env("PREVIEW_SECRET", "").strip() or secrets.token_hex(16),
        )
