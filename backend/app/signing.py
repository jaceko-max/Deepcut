"""HMAC-signed preview links, so /api/preview keeps working after the in-memory job store is lost."""
from __future__ import annotations

import hashlib
import hmac


def preview_sig(secret: str, deezer_id: int) -> str:
    return hmac.new(secret.encode(), f"preview:{deezer_id}".encode(), hashlib.sha256).hexdigest()[:20]


def preview_path(secret: str, deezer_id: int) -> str:
    return f"/api/preview/{deezer_id}?sig={preview_sig(secret, deezer_id)}"


def valid_preview_sig(secret: str, deezer_id: int, sig: str | None) -> bool:
    return bool(sig) and hmac.compare_digest(preview_sig(secret, deezer_id), sig)
