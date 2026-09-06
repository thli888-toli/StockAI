"""SQLite JSON cache shared by the US-stock agents."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from framework.config import US_STOCK_CACHE_DB
from plugins.stock_common import json_dumps, json_loads


class UsJsonCache:
    """Small (namespace, key) -> payload cache stored in the US cache DB."""

    def __init__(self, db_path: str | Path = US_STOCK_CACHE_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS us_json_cache (
                    namespace TEXT NOT NULL,
                    cache_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(namespace, cache_key)
                )
                """
            )

    def get(self, namespace: str, cache_key: str, max_age_seconds: int) -> Any | None:
        if max_age_seconds <= 0:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT payload, updated_at FROM us_json_cache
                WHERE namespace=? AND cache_key=?
                """,
                (namespace, cache_key),
            ).fetchone()
        if not row:
            return None
        payload, updated_at = row
        if time.time() - updated_at > max_age_seconds:
            return None
        return json_loads(payload, None)

    def put(self, namespace: str, cache_key: str, payload: Any) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO us_json_cache(namespace, cache_key, payload, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(namespace, cache_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (namespace, cache_key, json_dumps(payload), time.time()),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


US_CACHE = UsJsonCache()
