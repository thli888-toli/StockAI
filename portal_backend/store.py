"""SQLite storage for portal metrics, logs, agents, graph, and runs."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class PortalStore:
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
                CREATE TABLE IF NOT EXISTS agents (
                    name TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    last_heartbeat TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    description TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS metrics_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    service TEXT NOT NULL,
                    ts TEXT NOT NULL,
                    data TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_metrics_service_ts
                    ON metrics_snapshots(service, ts);
                CREATE TABLE IF NOT EXISTS logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    service TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    run_id TEXT,
                    node_id TEXT,
                    event TEXT,
                    message TEXT NOT NULL,
                    extra TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_logs_agent_ts
                    ON logs(agent, ts);
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    graph_config TEXT,
                    status TEXT NOT NULL,
                    query TEXT NOT NULL,
                    outputs TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    events TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    data TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS graph_configs (
                    name TEXT PRIMARY KEY,
                    active INTEGER NOT NULL,
                    updated_at TEXT NOT NULL
                );
                """
            )
        self._ensure_column("runs", "graph_config", "TEXT")

    def _ensure_column(self, table: str, column: str, column_type: str) -> None:
        with self.lock, self.conn:
            columns = {row["name"] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            if column not in columns:
                self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {column_type}")

    def upsert_agent(self, view: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO agents(name, status, last_heartbeat, endpoint, description, updated_at)
                VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    status=excluded.status,
                    last_heartbeat=excluded.last_heartbeat,
                    endpoint=excluded.endpoint,
                    description=excluded.description,
                    updated_at=excluded.updated_at
                """,
                (
                    view["name"],
                    view["status"],
                    view["last_heartbeat"],
                    view["card"]["endpoint"],
                    view["card"].get("description", ""),
                    _now(),
                ),
            )

    def insert_metric(self, service: str, data: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                "INSERT INTO metrics_snapshots(service, ts, data) VALUES(?, ?, ?)",
                (service, _now(), json.dumps(data, default=str)),
            )

    def insert_logs(self, records: list[dict[str, Any]]) -> None:
        rows = []
        for record in records:
            extra = dict(record.get("extra") or {})
            if record.get("exception"):
                extra["exception"] = record["exception"]
            rows.append(
                (
                    record.get("timestamp") or _now(),
                    record.get("level") or "INFO",
                    record.get("service") or record.get("agent") or "unknown",
                    record.get("agent") or record.get("service") or "unknown",
                    record.get("run_id"),
                    record.get("node_id"),
                    record.get("event"),
                    record.get("message", ""),
                    json.dumps(extra, default=str),
                )
            )
        with self.lock, self.conn:
            self.conn.executemany(
                """
                INSERT INTO logs(ts, level, service, agent, run_id, node_id, event, message, extra)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

    def upsert_run(self, run: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO runs(run_id, graph_config, status, query, outputs, error, created_at, updated_at, events)
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    graph_config=excluded.graph_config,
                    status=excluded.status,
                    outputs=excluded.outputs,
                    error=excluded.error,
                    updated_at=excluded.updated_at,
                    events=excluded.events
                """,
                (
                    run["run_id"],
                    run.get("graph_config"),
                    run["status"],
                    run["query"],
                    json.dumps(run.get("outputs", {}), default=str),
                    run.get("error"),
                    run.get("created_at") or _now(),
                    run.get("updated_at") or _now(),
                    json.dumps(run.get("events", []), default=str),
                ),
            )

    def mark_orphaned_running_runs_failed(self, known_run_ids: set[str]) -> None:
        """Mark a stored 'running' run failed only when it disappeared from the orchestrator."""
        with self.lock:
            rows = self.conn.execute(
                "SELECT run_id FROM runs WHERE status='running'"
            ).fetchall()
            for row in rows:
                if row["run_id"] not in known_run_ids:
                    self.conn.execute(
                        "UPDATE runs SET status='failed', error='run no longer present in orchestrator' WHERE run_id=?",
                        (row["run_id"],),
                    )

    def set_graph(self, data: dict[str, Any]) -> None:
        with self.lock, self.conn:
            self.conn.execute(
                """
                INSERT INTO graph(id, data, updated_at) VALUES(1, ?, ?)
                ON CONFLICT(id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
                """,
                (json.dumps(data, default=str), _now()),
            )

    def set_graph_configs(self, configs: list[dict[str, Any]]) -> None:
        with self.lock, self.conn:
            self.conn.execute("DELETE FROM graph_configs")
            self.conn.executemany(
                "INSERT INTO graph_configs(name, active, updated_at) VALUES(?, ?, ?)",
                [
                    (
                        item["name"],
                        1 if item.get("active") else 0,
                        _now(),
                    )
                    for item in configs
                ],
            )

    def get_agents(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute("SELECT * FROM agents ORDER BY name").fetchall()
        return [dict(row) for row in rows]

    def get_graph(self) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT data FROM graph WHERE id=1").fetchone()
        return json.loads(row["data"]) if row else None

    def get_graph_configs(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                "SELECT name, active FROM graph_configs ORDER BY name"
            ).fetchall()
        return [{"name": row["name"], "active": bool(row["active"])} for row in rows]

    def list_runs(self, limit: int = 50, graph_config: str | None = None) -> list[dict[str, Any]]:
        where = ""
        params: list[Any] = []
        if graph_config:
            where = " WHERE graph_config=?"
            params.append(graph_config)
        params.append(limit)
        with self.lock:
            rows = self.conn.execute(
                f"SELECT * FROM runs{where} ORDER BY created_at DESC LIMIT ?", params
            ).fetchall()
        runs = []
        for row in rows:
            item = dict(row)
            item["outputs"] = json.loads(item.pop("outputs") or "{}")
            item["events"] = json.loads(item.pop("events") or "[]")
            runs.append(item)
        return runs

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        with self.lock:
            row = self.conn.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if not row:
            return None
        item = dict(row)
        item["outputs"] = json.loads(item.pop("outputs") or "{}")
        item["events"] = json.loads(item.pop("events") or "[]")
        return item

    def query_metrics(self, service: str, limit: int = 200) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.conn.execute(
                """
                SELECT ts, data FROM metrics_snapshots
                WHERE service=?
                ORDER BY ts DESC
                LIMIT ?
                """,
                (service, limit),
            ).fetchall()
        result = []
        for row in reversed(rows):
            result.append({"ts": row["ts"], **json.loads(row["data"] or "{}")})
        return result

    def query_logs(
        self,
        agent: str | None = None,
        level: str | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        event: str | None = None,
        q: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = []
        params: list[Any] = []
        if agent:
            clauses.append("agent=?")
            params.append(agent)
        if level:
            clauses.append("level=?")
            params.append(level.upper())
        if run_id:
            clauses.append("run_id=?")
            params.append(run_id)
        if node_id:
            clauses.append("node_id=?")
            params.append(node_id)
        if event:
            clauses.append("event=?")
            params.append(event)
        if q:
            clauses.append("message LIKE ?")
            params.append(f"%{q}%")
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        params.append(limit)
        with self.lock:
            rows = self.conn.execute(
                f"SELECT * FROM logs {where} ORDER BY ts DESC LIMIT ?", params
            ).fetchall()
        return [dict(row) for row in rows]
