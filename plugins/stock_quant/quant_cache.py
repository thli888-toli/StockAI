"""Persistent SQLite cache for quant prediction payloads."""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from framework.config import STOCK_CACHE_DB


class QuantCacheStore:
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
                CREATE TABLE IF NOT EXISTS quant_cache (
                    symbol TEXT NOT NULL,
                    feature_hash TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(symbol, feature_hash)
                )
                """
            )

    def get(self, symbol: str, feature_hash: str) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT payload FROM quant_cache WHERE symbol=? AND feature_hash=?",
                (symbol, feature_hash),
            ).fetchone()
        return row[0] if row else None

    def put(self, symbol: str, feature_hash: str, payload: str) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                """
                INSERT INTO quant_cache(symbol, feature_hash, payload, updated_at)
                VALUES(?, ?, ?, ?)
                ON CONFLICT(symbol, feature_hash) DO UPDATE SET
                    payload=excluded.payload,
                    updated_at=excluded.updated_at
                """,
                (symbol, feature_hash, payload, time.time()),
            )
