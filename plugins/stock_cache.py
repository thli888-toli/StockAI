"""Persistent incremental SQLite store for AkShare stock history."""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd


COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "turnover",
    "pct_change",
]


class StockHistoryStore:
    """Stores daily bars once and appends only newly fetched data."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._lock = threading.Lock()
        self._create_tables()

    def _create_tables(self) -> None:
        with self._lock, self._conn:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS stock_history_bars (
                    symbol TEXT NOT NULL,
                    adjust TEXT NOT NULL,
                    trade_date TEXT NOT NULL,
                    open REAL NOT NULL,
                    high REAL NOT NULL,
                    low REAL NOT NULL,
                    close REAL NOT NULL,
                    volume REAL NOT NULL,
                    amount REAL NOT NULL,
                    turnover REAL NOT NULL,
                    pct_change REAL NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(symbol, adjust, trade_date)
                );
                CREATE INDEX IF NOT EXISTS idx_stock_history_bars_symbol_date
                    ON stock_history_bars(symbol, adjust, trade_date);
                CREATE TABLE IF NOT EXISTS stock_history_meta (
                    symbol TEXT NOT NULL,
                    adjust TEXT NOT NULL,
                    start_date TEXT NOT NULL,
                    end_date TEXT NOT NULL,
                    row_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(symbol, adjust)
                );
                """
            )

    def get_meta(self, symbol: str, adjust: str = "qfq") -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT start_date, end_date, row_count, updated_at
                FROM stock_history_meta
                WHERE symbol=? AND adjust=?
                """,
                (symbol, adjust),
            ).fetchone()
        if not row:
            return None
        return {
            "symbol": symbol,
            "adjust": adjust,
            "start_date": row[0],
            "end_date": row[1],
            "row_count": int(row[2]),
            "updated_at": row[3],
        }

    def missing_ranges(
        self,
        symbol: str,
        adjust: str,
        start_date: str,
        end_date: str,
    ) -> list[tuple[str, str]]:
        meta = self.get_meta(symbol, adjust)
        if not meta:
            return [(start_date, end_date)]

        requested_start = date.fromisoformat(start_date)
        requested_end = date.fromisoformat(end_date)
        stored_start = date.fromisoformat(meta["start_date"])
        stored_end = date.fromisoformat(meta["end_date"])
        ranges: list[tuple[str, str]] = []

        if requested_start < stored_start:
            ranges.append((requested_start.isoformat(), (stored_start - timedelta(days=1)).isoformat()))
        if stored_end < requested_end:
            ranges.append(((stored_end + timedelta(days=1)).isoformat(), requested_end.isoformat()))
        return ranges

    def load(
        self,
        symbol: str,
        adjust: str,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        where = ["symbol=?", "adjust=?"]
        params: list[Any] = [symbol, adjust]
        if start_date:
            where.append("trade_date>=?")
            params.append(start_date)
        if end_date:
            where.append("trade_date<=?")
            params.append(end_date)

        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT trade_date, open, high, low, close, volume, amount, turnover, pct_change
                FROM stock_history_bars
                WHERE {' AND '.join(where)}
                ORDER BY trade_date
                """,
                params,
            ).fetchall()

        frame = pd.DataFrame(rows, columns=COLUMNS)
        if frame.empty:
            return frame
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        numeric_columns = COLUMNS[1:]
        frame[numeric_columns] = frame[numeric_columns].apply(pd.to_numeric, errors="coerce")
        return frame.reset_index(drop=True)

    def merge(self, symbol: str, adjust: str, frame: pd.DataFrame) -> None:
        if frame is None or frame.empty:
            return
        missing_columns = [column for column in COLUMNS if column not in frame.columns]
        if missing_columns:
            raise ValueError(f"missing columns in history merge: {missing_columns}")

        now = time.time()
        rows = []
        for item in frame[COLUMNS].itertuples(index=False, name=None):
            trade_date = pd.to_datetime(item[0]).strftime("%Y-%m-%d")
            rows.append(
                (
                    symbol,
                    adjust,
                    trade_date,
                    float(item[1]),
                    float(item[2]),
                    float(item[3]),
                    float(item[4]),
                    float(item[5]),
                    float(item[6]),
                    float(item[7]),
                    float(item[8]),
                    now,
                )
            )

        with self._lock, self._conn:
            self._conn.executemany(
                """
                INSERT INTO stock_history_bars(
                    symbol, adjust, trade_date, open, high, low, close,
                    volume, amount, turnover, pct_change, updated_at
                )
                VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, adjust, trade_date) DO UPDATE SET
                    open=excluded.open,
                    high=excluded.high,
                    low=excluded.low,
                    close=excluded.close,
                    volume=excluded.volume,
                    amount=excluded.amount,
                    turnover=excluded.turnover,
                    pct_change=excluded.pct_change,
                    updated_at=excluded.updated_at
                """,
                rows,
            )
            self._update_meta(symbol, adjust, now)

    def _update_meta(self, symbol: str, adjust: str, now: float) -> None:
        row = self._conn.execute(
            """
            SELECT MIN(trade_date), MAX(trade_date), COUNT(*)
            FROM stock_history_bars
            WHERE symbol=? AND adjust=?
            """,
            (symbol, adjust),
        ).fetchone()
        if not row or row[0] is None:
            return
        self._conn.execute(
            """
            INSERT INTO stock_history_meta(symbol, adjust, start_date, end_date, row_count, updated_at)
            VALUES(?, ?, ?, ?, ?, ?)
            ON CONFLICT(symbol, adjust) DO UPDATE SET
                start_date=excluded.start_date,
                end_date=excluded.end_date,
                row_count=excluded.row_count,
                updated_at=excluded.updated_at
            """,
            (symbol, adjust, row[0], row[1], int(row[2]), now),
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


StockHistoryCache = StockHistoryStore
