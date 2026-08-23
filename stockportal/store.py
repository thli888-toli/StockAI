"""SQLite-backed watchlist storage for the stock analysis portal."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class WatchlistStore:
    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self.lock, self.conn:
            self.conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    openid TEXT NOT NULL UNIQUE,
                    nickname TEXT NOT NULL,
                    avatar TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT
                );
                CREATE TABLE IF NOT EXISTS watchlist (
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    company_name TEXT NOT NULL DEFAULT '',
                    industry TEXT NOT NULL DEFAULT '',
                    run_id TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    error TEXT,
                    outputs TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, symbol)
                );
                """
            )
            columns = {row["name"] for row in self.conn.execute("PRAGMA table_info(watchlist)")}
            if "user_id" not in columns:
                self.conn.execute("ALTER TABLE watchlist RENAME TO watchlist_old")
                self.conn.execute(
                    """
                    CREATE TABLE watchlist (
                        user_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        company_name TEXT NOT NULL DEFAULT '',
                        industry TEXT NOT NULL DEFAULT '',
                        run_id TEXT,
                        status TEXT NOT NULL DEFAULT 'running',
                        error TEXT,
                        outputs TEXT NOT NULL DEFAULT '{}',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY(user_id, symbol)
                    )
                    """
                )
                self.conn.execute(
                    """
                    INSERT INTO watchlist(
                        user_id, symbol, company_name, industry, run_id, status, error,
                        outputs, created_at, updated_at
                    )
                    SELECT
                        'default', symbol, company_name, industry, run_id, status, error,
                        outputs, created_at, updated_at
                    FROM watchlist_old
                    """
                )
                self.conn.execute("DROP TABLE watchlist_old")
            self.conn.execute(
                """
                INSERT OR IGNORE INTO users(id, openid, nickname, avatar, created_at)
                VALUES('default', 'default', '默认用户', '', ?)
                """,
                (_now(),),
            )

    @staticmethod
    def _item_from_row(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["outputs"] = json.loads(item.get("outputs") or "{}")
        return item

    def upsert(
        self,
        user_id: str,
        symbol: str,
        *,
        run_id: str | None = None,
        status: str = "running",
        error: str | None = None,
        outputs: dict[str, Any] | None = None,
        company_name: str = "",
        industry: str = "",
    ) -> dict[str, Any]:
        outputs_json = json.dumps(outputs or {}, default=str)
        now = _now()
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO watchlist(
                    user_id, symbol, company_name, industry, run_id, status, error, outputs,
                    created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, symbol) DO UPDATE SET
                    company_name=excluded.company_name,
                    industry=excluded.industry,
                    run_id=excluded.run_id,
                    status=excluded.status,
                    error=excluded.error,
                    outputs=excluded.outputs,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    symbol,
                    company_name,
                    industry,
                    run_id,
                    status,
                    error,
                    outputs_json,
                    now,
                    now,
                ),
            )
        return self.get(user_id, symbol) or {
            "user_id": user_id,
            "symbol": symbol,
            "company_name": company_name,
            "industry": industry,
            "run_id": run_id,
            "status": status,
            "error": error,
            "outputs": outputs or {},
            "created_at": now,
            "updated_at": now,
        }

    def get(self, user_id: str, symbol: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                "SELECT * FROM watchlist WHERE user_id=? AND symbol=?",
                (user_id, symbol),
            ).fetchone()
        return self._item_from_row(row) if row else None

    def all_items(self, user_id: str) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM watchlist WHERE user_id=? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        return [self._item_from_row(row) for row in rows]

    def delete(self, user_id: str, symbol: str) -> bool:
        with self.lock, self.conn:
            cursor = self.conn.execute(
                "DELETE FROM watchlist WHERE user_id=? AND symbol=?",
                (user_id, symbol),
            )
            return cursor.rowcount > 0
