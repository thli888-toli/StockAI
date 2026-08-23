from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.stock_cache import StockHistoryStore  # noqa: E402


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-20", "2026-08-21"]),
            "open": [10.0, 11.0],
            "high": [12.0, 13.0],
            "low": [9.0, 10.0],
            "close": [11.0, 12.0],
            "volume": [1000, 1200],
            "amount": [20000, 22000],
            "turnover": [1.0, 1.1],
            "pct_change": [0.0, 9.09],
        }
    )


def _missing_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-22"]),
            "open": [12.0],
            "high": [13.0],
            "low": [11.0],
            "close": [13.0],
            "volume": [1300],
            "amount": [24000],
            "turnover": [1.2],
            "pct_change": [8.33],
        }
    )


def test_stock_history_store_merges_and_persists(tmp_path):
    db_path = tmp_path / "stock_cache.db"
    first = StockHistoryStore(db_path)
    first.merge("600519", "qfq", _frame())

    assert first.missing_ranges(
        "600519",
        "qfq",
        "2026-08-20",
        "2026-08-22",
    ) == [("2026-08-22", "2026-08-22")]

    first.merge("600519", "qfq", _missing_frame())
    first.close()

    second = StockHistoryStore(db_path)
    loaded = second.load("600519", "qfq", "2026-08-20", "2026-08-22")
    meta = second.get_meta("600519", "qfq")
    second.close()

    assert len(loaded) == 3
    assert loaded["date"].min().strftime("%Y-%m-%d") == "2026-08-20"
    assert loaded["date"].max().strftime("%Y-%m-%d") == "2026-08-22"
    assert meta["start_date"] == "2026-08-20"
    assert meta["end_date"] == "2026-08-22"
    assert meta["row_count"] == 3


def test_stock_history_store_returns_full_range_when_empty(tmp_path):
    store = StockHistoryStore(tmp_path / "stock_cache.db")
    assert store.missing_ranges("600519", "qfq", "2024-01-01", "2026-08-22") == [
        ("2024-01-01", "2026-08-22")
    ]
    assert store.load("600519", "qfq").empty
    store.close()


def test_stock_history_store_supports_backfill_earlier_range(tmp_path):
    store = StockHistoryStore(tmp_path / "stock_cache.db")
    store.merge("600519", "qfq", _frame())
    assert store.missing_ranges("600519", "qfq", "2026-08-01", "2026-08-21") == [
        ("2026-08-01", "2026-08-19")
    ]
    store.close()
