"""Persistent SQLite cache for fundamental-analysis tool outputs."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from framework.config import STOCK_CACHE_DB


class FundamentalCacheStore:
    """Stores serialized tool payloads keyed by (symbol, tool, period_key)."""

    def __init__(self, db_path: str | Path = STOCK_CACHE_DB) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS fundamental_cache (
                    symbol TEXT NOT NULL,
                    tool TEXT NOT NULL,
                    period_key TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(symbol, tool, period_key)
                )
                """
            )

    def get(self, symbol: str, tool: str, period_key: str, max_age_seconds: int) -> str | None:
        if max_age_seconds <= 0:
            return None
        with self._lock:
            row = self._conn.execute(
                """
                SELECT payload, updated_at
                FROM fundamental_cache
                WHERE symbol=? AND tool=? AND period_key=?
                """,
                (symbol, tool, period_key),
            ).fetchone()
        if not row:
            return None
        payload, updated_at = row
        if time.time() - updated_at > max_age_seconds:
            return None
        return payload

    def put(self, symbol: str, tool: str, period_key: str, payload: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO fundamental_cache(symbol, tool, period_key, payload, updated_at)
                VALUES(?, ?, ?, ?, ?)
                ON CONFLICT(symbol, tool, period_key) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (symbol, tool, period_key, payload, time.time()),
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()
