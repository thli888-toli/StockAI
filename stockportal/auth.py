"""Mock WeChat login and session management for the stock portal."""

from __future__ import annotations

import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from framework.config import STOCK_PORTAL_DB


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AuthStore:
    def __init__(self, db_path: str | Path = STOCK_PORTAL_DB) -> None:
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
                """
            )

    def login(
        self,
        openid: str | None = None,
        nickname: str | None = None,
    ) -> dict[str, Any]:
        nickname = (nickname or "").strip()
        openid = (openid or "").strip()
        if not openid:
            if nickname in ("default", "默认用户"):
                openid = "default"
            else:
                with self.lock:
                    row = self.conn.execute(
                        "SELECT openid FROM users WHERE nickname=? ORDER BY created_at LIMIT 1",
                        (nickname,),
                    ).fetchone()
                openid = row["openid"] if row else f"mock_{uuid.uuid4().hex}"
        nickname = nickname or f"用户_{openid[:6]}"
        now = _now()
        with self.lock, self.conn:
            row = self.conn.execute(
                "SELECT id, openid, nickname, avatar FROM users WHERE openid=?",
                (openid,),
            ).fetchone()
            if row:
                user_id = row["id"]
                avatar = row["avatar"]
            else:
                user_id = uuid.uuid4().hex
                avatar = ""
                self.conn.execute(
                    """
                    INSERT INTO users(id, openid, nickname, avatar, created_at)
                    VALUES(?, ?, ?, ?, ?)
                    """,
                    (user_id, openid, nickname, avatar, now),
                )
            token = secrets.token_hex(32)
            self.conn.execute(
                """
                INSERT INTO sessions(token, user_id, created_at, expires_at)
                VALUES(?, ?, ?, NULL)
                """,
                (token, user_id, now),
            )
        return {
            "token": token,
            "user_id": user_id,
            "openid": openid,
            "nickname": nickname,
            "avatar": avatar,
        }

    def get_user_by_token(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        with self.lock:
            row = self.conn.execute(
                """
                SELECT u.id AS user_id, u.openid, u.nickname, u.avatar
                FROM sessions s
                JOIN users u ON u.id = s.user_id
                WHERE s.token=? AND (s.expires_at IS NULL OR s.expires_at > ?)
                """,
                (token, _now()),
            ).fetchone()
        if not row:
            return None
        return {
            "user_id": row["user_id"],
            "openid": row["openid"],
            "nickname": row["nickname"],
            "avatar": row["avatar"],
        }
