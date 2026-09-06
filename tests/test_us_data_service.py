from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import TaskRequest  # noqa: E402
from plugins.stock_common import normalize_akshare_frame, resample_ohlcv  # noqa: E402
from plugins.us_common import validate_us_symbol  # noqa: E402
from plugins.us_data import service as us_data_service  # noqa: E402


def _raw_frame(rows: int = 900) -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype="float64")
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": close - 0.5,
            "收盘": close,
            "最高": close + 1.0,
            "最低": close - 1.5,
            "成交量": 1_000_000.0,
            "成交额": close * 1_000_000.0,
            "换手率": 0.5,
        }
    )


def test_validate_us_symbol_accepts_letters_and_rejects_cn_code():
    assert validate_us_symbol("aapl") == "AAPL"
    assert validate_us_symbol("BRK.B") == "BRK.B"
    with pytest.raises(ValueError):
        validate_us_symbol("600519")


def test_us_data_payload_uses_usd_and_normalized_features(monkeypatch, tmp_path):
    daily_raw = _raw_frame()
    daily = normalize_akshare_frame(daily_raw)
    monthly = resample_ohlcv(daily, "ME")
    monthly["pct_change"] = monthly["close"].pct_change().fillna(0.0)

    monkeypatch.setattr(
        us_data_service,
        "US_STOCK_CACHE_DB",
        str(tmp_path / "us_cache.db"),
    )
    async def fake_fetch(symbol, start, end):
        return daily

    monkeypatch.setattr(us_data_service, "fetch_us_history", fake_fetch)
    monkeypatch.setattr(
        us_data_service,
        "yf_ticker_history",
        lambda symbol, period="10y", interval="1mo": monthly,
    )
    monkeypatch.setattr(
        us_data_service,
        "yf_ticker_info",
        lambda ticker: {
            "longName": "Apple Inc.",
            "sector": "Technology",
            "industry": "Consumer Electronics",
            "exchange": "NMS",
        },
    )

    handler = us_data_service.UsDataHandler()
    payload = json.loads(
        asyncio.run(handler.run(TaskRequest(query="AAPL", inputs={})))
    )
    assert payload["currency"] == "USD"
    assert payload["market"] == "us"
    assert payload["company_name"] == "Apple Inc."
    assert payload["industry"] == "Technology / Consumer Electronics"
    assert len(payload["daily_features"]) >= 400
    assert payload["monthly_history"]
    required = {"date", "open", "high", "low", "close", "volume"}
    assert required.issubset(payload["daily_features"][0])
    assert payload["latest"]["close"] > 0
