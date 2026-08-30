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
                    tags TEXT NOT NULL DEFAULT '[]',
                    run_id TEXT,
                    status TEXT NOT NULL DEFAULT 'running',
                    error TEXT,
                    outputs TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(user_id, symbol)
                );
                CREATE TABLE IF NOT EXISTS chart_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    period TEXT NOT NULL,
                    label TEXT NOT NULL DEFAULT '',
                    payload TEXT NOT NULL,
                    saved_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_chart_snapshots_user_symbol
                    ON chart_snapshots(user_id, symbol, saved_at DESC);
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
                        tags TEXT NOT NULL DEFAULT '[]',
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
            columns = {
                row["name"]
                for row in self.conn.execute("PRAGMA table_info(watchlist)")
            }
            if "tags" not in columns:
                self.conn.execute(
                    "ALTER TABLE watchlist ADD COLUMN tags TEXT NOT NULL DEFAULT '[]'"
                )
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
        try:
            item["tags"] = json.loads(item.get("tags") or "[]")
        except (TypeError, ValueError):
            item["tags"] = []
        return item

    @staticmethod
    def _normalize_tags(tags: list[str] | str | None) -> list[str]:
        """Return a unique, trimmed list of non-empty tags."""
        if tags is None:
            return []
        if isinstance(tags, str):
            raw = [part.strip() for part in tags.replace("，", ",").replace("、", ",").split(",")]
        else:
            raw = [str(part).strip() for part in tags]
        seen: list[str] = []
        for tag in raw:
            if tag and tag not in seen:
                seen.append(tag)
        return seen

    @staticmethod
    def _tags_json(tags: list[str] | str | None) -> str:
        return json.dumps(WatchlistStore._normalize_tags(tags), ensure_ascii=False)

    def update_tags(
        self,
        user_id: str,
        symbol: str,
        tags: list[str] | str | None,
    ) -> dict[str, Any] | None:
        """Replace the tag list of one watchlist row and return the item."""
        normalized = self._normalize_tags(tags)
        with self.lock, self.conn:
            self.conn.execute(
                "UPDATE watchlist SET tags=?, updated_at=? WHERE user_id=? AND symbol=?",
                (json.dumps(normalized, ensure_ascii=False), _now(), user_id, symbol),
            )
        return self.get(user_id, symbol)

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
        tags: list[str] | str | None = None,
    ) -> dict[str, Any]:
        outputs_json = json.dumps(outputs or {}, default=str)
        if tags is None:
            existing = self.get(user_id, symbol)
            tags = (existing or {}).get("tags", [])
        tags_json = self._tags_json(tags)
        now = _now()
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO watchlist(
                    user_id, symbol, company_name, industry, tags, run_id, status, error,
                    outputs, created_at, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id, symbol) DO UPDATE SET
                    company_name=excluded.company_name,
                    industry=excluded.industry,
                    run_id=excluded.run_id,
                    status=excluded.status,
                    error=excluded.error,
                    outputs=excluded.outputs,
                    tags=excluded.tags,
                    updated_at=excluded.updated_at
                """,
                (
                    user_id,
                    symbol,
                    company_name,
                    industry,
                    tags_json,
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
            "tags": self._normalize_tags(tags),
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

    def all_by_symbol(self, symbol: str) -> list[dict[str, Any]]:
        """Return every watchlist row for a symbol across all users."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM watchlist WHERE symbol=? ORDER BY created_at DESC",
                (symbol,),
            ).fetchall()
        return [self._item_from_row(row) for row in rows]

    def delete(self, user_id: str, symbol: str) -> bool:
        with self.lock, self.conn:
            cursor = self.conn.execute(
                "DELETE FROM watchlist WHERE user_id=? AND symbol=?",
                (user_id, symbol),
            )
            return cursor.rowcount > 0

    def save_chart_snapshot(
        self,
        user_id: str,
        symbol: str,
        period: str,
        payload: dict[str, Any],
        label: str = "",
    ) -> dict[str, Any]:
        now = _now()
        with self.lock, self.conn:
            cursor = self.conn.execute(
                """
                INSERT INTO chart_snapshots(user_id, symbol, period, label, payload, saved_at)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                (
                    user_id,
                    symbol,
                    period,
                    label or now,
                    json.dumps(payload, default=str),
                    now,
                ),
            )
            snapshot_id = int(cursor.lastrowid)
            # Keep only the most recent 20 snapshots per user + symbol.
            self.conn.execute(
                """
                DELETE FROM chart_snapshots
                WHERE user_id=? AND symbol=? AND id NOT IN (
                    SELECT id FROM chart_snapshots
                    WHERE user_id=? AND symbol=?
                    ORDER BY saved_at DESC, id DESC
                    LIMIT 20
                )
                """,
                (user_id, symbol, user_id, symbol),
            )
        return {
            "id": snapshot_id,
            "user_id": user_id,
            "symbol": symbol,
            "period": period,
            "label": label or now,
            "saved_at": now,
        }

    def list_chart_snapshots(
        self, user_id: str, symbol: str
    ) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT id, period, label, saved_at
                FROM chart_snapshots
                WHERE user_id=? AND symbol=?
                ORDER BY saved_at DESC, id DESC
                """,
                (user_id, symbol),
            ).fetchall()
        return [dict(row) for row in rows]

    def get_chart_snapshot(
        self, user_id: str, symbol: str, snapshot_id: int
    ) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute(
                """
                SELECT * FROM chart_snapshots
                WHERE user_id=? AND symbol=? AND id=?
                """,
                (user_id, symbol, snapshot_id),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        item["payload"] = json.loads(item.get("payload") or "{}")
        return item

    def delete_chart_snapshot(
        self, user_id: str, symbol: str, snapshot_id: int
    ) -> bool:
        with self.lock, self.conn:
            cursor = self.conn.execute(
                """
                DELETE FROM chart_snapshots
                WHERE user_id=? AND symbol=? AND id=?
                """,
                (user_id, symbol, snapshot_id),
            )
            return cursor.rowcount > 0
