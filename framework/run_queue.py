"""SQLite-backed FIFO queue for pending orchestrator runs."""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SQLiteRunQueue:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_queue (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL UNIQUE,
                    query TEXT NOT NULL,
                    context TEXT,
                    graph_config TEXT,
                    created_at TEXT NOT NULL
                )
                """
            )

    def enqueue(
        self,
        run_id: str,
        query: str,
        context: str | None,
        graph_config: str,
    ) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO run_queue(run_id, query, context, graph_config, created_at)
                VALUES(?, ?, ?, ?, ?)
                """,
                (run_id, query, context, graph_config, _now()),
            )

    def dequeue(self) -> dict[str, Any] | None:
        with self.lock, self.conn:
            row = self.conn.execute(
                "SELECT * FROM run_queue ORDER BY id LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            self.conn.execute("DELETE FROM run_queue WHERE id=?", (row["id"],))
            return dict(row)

    def remove(self, run_id: str) -> bool:
        with self.lock, self.conn:
            cursor = self.conn.execute(
                "DELETE FROM run_queue WHERE run_id=?", (run_id,)
            )
            return cursor.rowcount > 0

    def list_items(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT * FROM run_queue ORDER BY id"
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        with self.lock:
            self.conn.close()
