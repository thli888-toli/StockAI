from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import TaskRequest  # noqa: E402
from plugins.stock_fundamental import service as fundamental_service  # noqa: E402
from plugins.stock_fundamental.service import StockFundamentalHandler  # noqa: E402


def _statements() -> dict:
    return {
        "symbol": "600519",
        "periods": ["2026-06-30", "2026-03-31", "2025-12-31", "2025-09-30", "2025-06-30"],
        "balance_sheet": [
            {"report_date": "2026-06-30", "parent_equity": 2.0e11},
            {"report_date": "2025-12-31", "parent_equity": 1.9e11},
        ],
        "income_statement": [
            {"report_date": "2026-06-30", "operate_income": 100.0e9, "parent_net_profit": 45.0e9},
            {"report_date": "2026-03-31", "operate_income": 55.0e9, "parent_net_profit": 27.0e9},
            {"report_date": "2025-12-31", "operate_income": 170.0e9, "parent_net_profit": 82.0e9},
            {"report_date": "2025-09-30", "operate_income": 128.0e9, "parent_net_profit": 64.0e9},
            {"report_date": "2025-06-30", "operate_income": 82.0e9, "parent_net_profit": 41.0e9},
        ],
        "cash_flow": [
            {"report_date": "2026-06-30", "netcash_operate": 70.0e9, "construct_long_asset": 0.8e9, "assign_dividend": 35.0e9},
            {"report_date": "2026-03-31", "netcash_operate": 27.0e9, "construct_long_asset": 0.6e9, "assign_dividend": None},
            {"report_date": "2025-12-31", "netcash_operate": 61.0e9, "construct_long_asset": 3.1e9, "assign_dividend": 67.0e9},
            {"report_date": "2025-09-30", "netcash_operate": 38.0e9, "construct_long_asset": 2.3e9, "assign_dividend": 37.0e9},
            {"report_date": "2025-06-30", "netcash_operate": 30.0e9, "construct_long_asset": 1.5e9, "assign_dividend": 35.0e9},
        ],
    }


def _indicators() -> dict:
    return {
        "symbol": "600519",
        "em": [
            {
                "report_date": "2026-06-30",
                "eps": 35.57,
                "bps": 155.0,
                "roe": 12.0,
                "gross_margin": 91.0,
                "net_margin": 48.0,
                "debt_ratio": 20.0,
                "revenue_yoy": 12.0,
                "net_profit_yoy": 15.0,
            }
        ],
        "sina": [],
        "abstract": [],
        "latest": {
            "report_date": "2026-06-30",
            "eps": 35.57,
            "bps": 155.0,
            "roe": 12.0,
            "gross_margin": 91.0,
            "debt_ratio": 20.0,
            "revenue_yoy": 12.0,
            "net_profit_yoy": 15.0,
        },
    }


def _snapshot() -> dict:
    return {
        "symbol": "600519",
        "data_date": "2026-08-24",
        "close": 1304.66,
        "pe_ttm": 20.03,
        "pe_static": 19.81,
        "pb": 6.49,
        "ps": 9.41,
        "pcf": 13.69,
        "total_shares": 1_250_081_601,
        "dividend_yield": 4.1,
    }


def _fake_run_tool(all_fail: bool = False):
    async def fake(name, symbol, market_data, warnings, metrics=None):
        if all_fail and name in fundamental_service.CORE_TOOLS:
            raise RuntimeError(f"{name} unavailable")
        if name == "get_company_profile":
            return {"symbol": symbol, "company_name": "贵州茅台", "industry": "白酒"}
        if name == "get_financial_statements":
            return _statements()
        if name == "get_financial_indicators":
            return _indicators()
        if name == "get_financial_abstract":
            return {"symbol": symbol, "periods": [], "business_composition": []}
        if name == "get_valuation_snapshot":
            return _snapshot()
        if name == "get_historical_valuation_percentile":
            return {
                "symbol": symbol,
                "window": "近三年",
                "metrics": {
                    "pe_ttm": {"p50": 25.0, "percentile": 30.0},
                    "pb": {"p50": 8.0, "percentile": 25.0},
                },
            }
        if name == "get_industry_valuation_comparison":
            return {"symbol": symbol, "source": "cninfo", "pe": {"median": 22.0, "mean": 30.0}}
        if name == "get_earnings_forecast":
            return {"symbol": symbol, "research_reports": [], "earnings_guidance": []}
        if name == "estimate_fair_value":
            return {
                "fair_value_range": {"low": 1350.0, "mid": 1500.0, "high": 1650.0},
                "per_method": {
                    "relative": {"available": True, "price": 1500.0},
                    "dcf": {"available": True, "price": 1450.0},
                    "ddm": {"available": True, "price": 1550.0},
                },
                "verdict": {
                    "label": "低估",
                    "current_price": 1304.66,
                    "margin": -0.1302,
                },
                "available_methods": ["relative", "dcf", "ddm"],
                "assumptions": {"discount_rate": 0.1, "terminal_growth": 0.02},
                "disclaimer": "估算区间不构成投资建议。",
            }
        raise ValueError(f"unexpected tool: {name}")

    return fake


@pytest.mark.asyncio
async def test_handler_produces_analysis_report_section_and_summary(monkeypatch):
    monkeypatch.setattr(
        "plugins.stock_fundamental.service.run_tool",
        _fake_run_tool(),
    )
    monkeypatch.setattr("plugins.stock_fundamental.service.llm_configured", lambda: False)
    result = await StockFundamentalHandler().run(
        TaskRequest(
            query="600519",
            inputs={"market_data": json.dumps({"latest": {"close": 1304.66}})},
        )
    )
    payload = json.loads(result)
    assert set(payload) == {"analysis", "report_section", "summary"}
    assert payload["analysis"]["symbol"] == "600519"
    assert payload["analysis"]["industry"] == "白酒"
    assert payload["analysis"]["metrics"]["industry"]["pe"]["median"] == 22.0
    assert payload["analysis"]["valuation"]["fair_value_range"]["mid"] == 1500.0
    assert "## 基本面与估值" in payload["report_section"]
    assert "低估" in payload["report_section"]
    assert payload["summary"]["valuation_verdict"] == "低估"
    assert payload["summary"]["fair_value_range"]["high"] == 1650.0
    assert payload["summary"]["as_of"] == "2026-08-24"


@pytest.mark.asyncio
async def test_handler_optional_tool_failure_is_tolerated(monkeypatch):
    async def fake(name, symbol, market_data, warnings, metrics=None):
        if name == "get_industry_valuation_comparison":
            raise RuntimeError("industry API down")
        return await _fake_run_tool()(name, symbol, market_data, warnings, metrics)

    monkeypatch.setattr("plugins.stock_fundamental.service.run_tool", fake)
    monkeypatch.setattr("plugins.stock_fundamental.service.llm_configured", lambda: False)
    result = await StockFundamentalHandler().run(TaskRequest(query="600519"))
    payload = json.loads(result)
    assert any("get_industry_valuation_comparison" in warning for warning in payload["analysis"]["warnings"])
    assert "## 基本面与估值" in payload["report_section"]


@pytest.mark.asyncio
async def test_handler_fails_when_all_core_tools_fail(monkeypatch):
    monkeypatch.setattr(
        "plugins.stock_fundamental.service.run_tool",
        _fake_run_tool(all_fail=True),
    )
    with pytest.raises(RuntimeError, match="核心基本面工具"):
        await StockFundamentalHandler().run(TaskRequest(query="600519"))


def test_ttm_computes_trailing_twelve_months():
    records = [
        {"report_date": "2025-06-30", "value": 30.0},
        {"report_date": "2025-09-30", "value": 38.0},
        {"report_date": "2025-12-31", "value": 61.0},
        {"report_date": "2026-03-31", "value": 27.0},
        {"report_date": "2026-06-30", "value": 70.0},
    ]
    assert fundamental_service._ttm(records, "value") == pytest.approx(70.0 + 61.0 - 30.0)


def test_ttm_annual_report_returns_cumulative_value():
    records = [
        {"report_date": "2025-12-31", "value": 61.0},
        {"report_date": "2026-03-31", "value": 27.0},
    ]
    # latest is 2026-03-31 -> TTM = 27 + 61 - prior-year Q1 (missing) -> fallback needs 5 rows
    assert fundamental_service._ttm(records, "value") is None
    annual = [{"report_date": "2025-12-31", "value": 61.0}]
    assert fundamental_service._ttm(annual, "value") == 61.0
