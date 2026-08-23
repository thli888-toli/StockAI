"""Async SQLite checkpoint helpers for LangGraph."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path


def _patch_aiosqlite() -> None:
    import aiosqlite

    if not hasattr(aiosqlite.Connection, "is_alive"):
        def is_alive(self) -> bool:
            return getattr(self, "_connection", None) is not None

        aiosqlite.Connection.is_alive = is_alive


@asynccontextmanager
async def open_checkpointer(db_path: str | Path):
    """Open a short-lived AsyncSqliteSaver for the lifetime of one run."""
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    _patch_aiosqlite()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(path.resolve())) as saver:
        await saver.setup()
        yield saver


async def create_shared_checkpointer(db_path: str | Path):
    """Open one long-lived AsyncSqliteSaver shared by all concurrent runs.

    The saver owns an asyncio lock around checkpoint writes, which prevents the
    ``database is locked`` errors that happen when every run opens its own
    connection and writes to the same SQLite file simultaneously.
    """
    import aiosqlite
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    _patch_aiosqlite()
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = await aiosqlite.connect(str(path.resolve()), timeout=30.0)
    await conn.execute("PRAGMA journal_mode=WAL")
    await conn.execute("PRAGMA synchronous=NORMAL")
    await conn.execute("PRAGMA busy_timeout=30000")
    saver = AsyncSqliteSaver(conn)
    await saver.setup()
    return saver
