"""SQLite cache (WAL): raw audio features and HTTP responses."""
from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time


class Cache:
    def __init__(self, cache_dir: str):
        os.makedirs(cache_dir, exist_ok=True)
        self.path = os.path.join(cache_dir, "deepcut.sqlite")
        with self._conn() as c:
            c.execute("PRAGMA journal_mode=WAL")
            c.execute("CREATE TABLE IF NOT EXISTS features (key TEXT PRIMARY KEY, data TEXT NOT NULL, created REAL NOT NULL)")
            c.execute("CREATE TABLE IF NOT EXISTS http_cache (key TEXT PRIMARY KEY, body TEXT NOT NULL, expires REAL NOT NULL)")
        self._purge_expired()

    def _purge_expired(self) -> int:
        c = self._conn()
        try:
            with c:
                return c.execute("DELETE FROM http_cache WHERE expires < ?", (time.time(),)).rowcount
        finally:
            c.close()

    async def purge_expired(self) -> int:
        return await asyncio.to_thread(self._purge_expired)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, timeout=10)
        c.execute("PRAGMA busy_timeout=10000")
        return c

    # sync primitives -------------------------------------------------
    def _get_features(self, key: str):
        c = self._conn()
        try:
            row = c.execute("SELECT data FROM features WHERE key=?", (key,)).fetchone()
        finally:
            c.close()
        return json.loads(row[0]) if row else None

    def _put_features(self, key: str, data: dict) -> None:
        c = self._conn()
        try:
            with c:
                c.execute("INSERT OR REPLACE INTO features(key,data,created) VALUES(?,?,?)", (key, json.dumps(data), time.time()))
        finally:
            c.close()

    def _get_http(self, key: str):
        c = self._conn()
        try:
            row = c.execute("SELECT body FROM http_cache WHERE key=? AND expires>?", (key, time.time())).fetchone()
        finally:
            c.close()
        return row[0] if row else None

    def _put_http(self, key: str, body: str, ttl: float) -> None:
        c = self._conn()
        try:
            with c:
                c.execute("INSERT OR REPLACE INTO http_cache(key,body,expires) VALUES(?,?,?)", (key, body, time.time() + ttl))
        finally:
            c.close()

    # async API -------------------------------------------------------
    async def get_features(self, key: str):
        return await asyncio.to_thread(self._get_features, key)

    async def put_features(self, key: str, data: dict) -> None:
        await asyncio.to_thread(self._put_features, key, data)

    async def get_http(self, key: str):
        return await asyncio.to_thread(self._get_http, key)

    async def put_http(self, key: str, body: str, ttl: float) -> None:
        await asyncio.to_thread(self._put_http, key, body, ttl)
