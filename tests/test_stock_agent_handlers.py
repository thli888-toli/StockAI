from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import TaskRequest  # noqa: E402
from plugins.stock_analyst.service import StockAnalystHandler  # noqa: E402
from plugins.stock_data import service as stock_data_service  # noqa: E402
from plugins.stock_data.service import StockDataHandler  # noqa: E402
from plugins.stock_news.service import StockNewsHandler  # noqa: E402
from plugins.stock_quant import service as stock_quant_service  # noqa: E402
from plugins.stock_quant.service import StockQuantHandler  # noqa: E402
from plugins.stock_common import normalize_akshare_frame  # noqa: E402


def _raw_history(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype="float64")
    return pd.DataFrame(
        {
            "日期": dates,
            "开盘": close - 1.0,
            "收盘": close,
            "最高": close + 1.0,
            "最低": close - 2.0,
            "成交量": 1000000.0,
            "成交额": 20000000.0,
            "换手率": 1.0,
        }
    )


class _FakeHistoryStore:
    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def missing_ranges(self, symbol, adjust, start_date, end_date):
        return [(start_date, end_date)]

    def merge(self, symbol, adjust, frame):
        return None

    def load(self, symbol, adjust, start_date, end_date):
        return self.frame

    def get_meta(self, symbol, adjust):
        return {
            "symbol": symbol,
            "adjust": adjust,
            "start_date": start_date if "start_date" in locals() else "2024-01-01",
            "end_date": end_date if "end_date" in locals() else "2026-08-22",
            "row_count": len(self.frame),
            "updated_at": 1,
        }


@pytest.mark.asyncio
async def test_stock_data_handler_complete_run(monkeypatch):
    fake_store = _FakeHistoryStore(normalize_akshare_frame(_raw_history()))
    monkeypatch.setattr(stock_data_service, "StockHistoryStore", lambda path: fake_store)

    handler = StockDataHandler()
    monkeypatch.setattr(
        handler,
        "_fetch_daily",
        lambda symbol, start_date, end_date: _raw_history(),
    )

    async def fake_company_info(symbol):
        return {"company_name": "贵州茅台", "industry": "白酒"}

    monkeypatch.setattr(handler, "_fetch_company_info_optional", fake_company_info)

    result = await handler.run(TaskRequest(query="600519"))
    payload = json.loads(result)

    assert payload["symbol"] == "600519"
    assert payload["company_name"] == "贵州茅台"
    assert payload["macd"]["daily"]["trend"] == "bullish"
    assert payload["daily_features"]
    assert payload["history_cache"]["row_count"] == 180


@pytest.mark.asyncio
async def test_stock_news_handler_complete_run(monkeypatch):
    async def fake_eastmoney(self, symbol, industry, warnings):
        return [
            {
                "title": "关于公司发布重大合同的公告",
                "published_at": "2026-08-20 09:00:00",
                "url": "https://eastmoney.example/1",
                "source": "东方财富",
                "summary": "company news",
                "source_type": "eastmoney_news",
            }
        ]

    async def fake_cninfo(self, symbol, warnings):
        return [
            {
                "title": "公司发布重大合同的公告",
                "published_at": "2026-08-21 10:00:00",
                "url": "https://cninfo.example/1",
                "source": "巨潮资讯",
                "summary": "disclosure",
                "source_type": "cninfo_disclosure",
            }
        ]

    monkeypatch.setattr(StockNewsHandler, "_fetch_eastmoney", fake_eastmoney)
    monkeypatch.setattr(StockNewsHandler, "_fetch_cninfo", fake_cninfo)

    result = await StockNewsHandler().run(
        TaskRequest(
            query="600519",
            inputs={
                "market_data": json.dumps(
                    {"symbol": "600519", "company_name": "贵州茅台", "industry": "白酒"}
                )
            },
        )
    )
    payload = json.loads(result)

    assert payload["symbol"] == "600519"
    assert len(payload["eastmoney_news"]) == 1
    assert len(payload["cninfo_disclosures"]) == 1
    assert len(payload["cross_validated"]) == 1
    assert payload["cross_validated"][0]["confidence"] == "high"


@pytest.mark.asyncio
async def test_stock_quant_handler_complete_run(monkeypatch):
    expected = {
        "symbol": "600519",
        "as_of": "2026-08-22",
        "model": "LightGBM",
        "horizons": {
            "1d": {"up_probability": 0.58, "direction": "up", "confidence": 0.16},
            "5d": {"up_probability": 0.55, "direction": "up", "confidence": 0.10},
            "20d": {"up_probability": 0.49, "direction": "flat", "confidence": 0.02},
        },
        "backtest": {"walk_forward_auc": 0.61, "sample_count": 720},
        "warnings": [],
    }

    def fake_build(symbol, records, as_of):
        return json.dumps(expected)

    monkeypatch.setattr(stock_quant_service, "_build_quant_payload", fake_build)

    result = await StockQuantHandler().run(
        TaskRequest(
            query="600519",
            inputs={
                "market_data": json.dumps(
                    {"symbol": "600519", "as_of": "2026-08-22", "daily_features": []}
                )
            },
        )
    )
    assert json.loads(result) == expected


@pytest.mark.asyncio
async def test_stock_analyst_handler_llm_configured_path(monkeypatch):
    async def fake_llm(system, user, max_tokens=2000):
        return (
            "```json\n"
            '{"signal":{"1d":{"direction":"up"},"5d":{"direction":"up"},'
            '"20d":{"direction":"up"}},"report":"# LLM report"}\n'
            "```"
        )

    monkeypatch.setattr("plugins.stock_analyst.service.llm_configured", lambda: True)
    monkeypatch.setattr("plugins.stock_analyst.service.llm_reply", fake_llm)

    result = await StockAnalystHandler().run(
        TaskRequest(
            query="600519",
            inputs={
                "market_data": json.dumps(
                    {
                        "symbol": "600519",
                        "company_name": "贵州茅台",
                        "industry": "白酒",
                        "macd": {"daily": {"trend": "bullish"}},
                    }
                ),
                "news": json.dumps({"source_counts": {}, "warnings": []}),
                "quant": json.dumps(
                    {
                        "horizons": {
                            "1d": {"direction": "up", "confidence": 0.7},
                            "5d": {"direction": "up", "confidence": 0.7},
                            "20d": {"direction": "up", "confidence": 0.7},
                        },
                        "backtest": {},
                    }
                ),
            },
        )
    )

    assert "# LLM report" in result
    assert "免责声明" in result
