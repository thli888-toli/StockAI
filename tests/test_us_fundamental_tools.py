from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import TaskRequest  # noqa: E402
from plugins.us_fundamental import edgar as edgar_module  # noqa: E402
from plugins.us_cache import UsJsonCache  # noqa: E402
from plugins.us_fundamental import service as fundamental_service  # noqa: E402
from plugins.us_fundamental import tools  # noqa: E402
from plugins.us_fundamental.us_config import clear_us_config_cache  # noqa: E402


def _monthly_history(rows: int = 40) -> pd.DataFrame:
    dates = pd.date_range("2023-06-30", periods=rows, freq="ME")
    close = pd.Series(range(100, 100 + rows), dtype="float64")
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": close - 1.0,
            "high": close + 2.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1_000_000.0,
            "amount": close * 1_000_000.0,
            "turnover": 0.0,
            "pct_change": 0.0,
        }
    )
    return frame


@pytest.fixture(autouse=True)
def _no_llm_and_clean_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(tools, "llm_configured", lambda: False)
    monkeypatch.setattr(
        tools,
        "US_CACHE",
        UsJsonCache(tmp_path / "us_tool_cache.db"),
    )
    clear_us_config_cache()
    yield
    clear_us_config_cache()


def _base_results() -> dict:
    statements = {
        "revenue_ttm": 100_000_000_000.0,
        "net_income_ttm": 20_000_000_000.0,
        "ocf_ttm": 30_000_000_000.0,
        "capex_ttm": 5_000_000_000.0,
        "dividends_ttm": 2_000_000_000.0,
        "latest_equity": 50_000_000_000.0,
        "latest_shares": 1_000_000_000.0,
        "latest_report_date": "2025-09-30",
        "source": "sec_edgar",
    }
    snapshot = {
        "data_date": date.today().isoformat(),
        "close": 100.0,
        "total_market_cap": 100_000_000_000.0,
        "total_shares": 1_000_000_000.0,
        "pe_ttm": 5.0,
        "pb": 2.0,
        "ps": 1.0,
        "dividend_yield": 2.0,
        "trailing_eps": 20.0,
        "forward_eps": 25.0,
        "gross_margin": 0.4,
        "earnings_growth": 0.1,
        "revenue_growth": 0.08,
    }
    hist = {
        "metrics": {
            "pe_ttm": {"latest": 30.0, "p25": 20.0, "p50": 25.0, "p75": 32.0, "percentile": 60.0, "samples": 36},
            "pb": {"latest": 5.0, "p25": 4.0, "p50": 5.0, "p75": 6.0, "percentile": 50.0, "samples": 36},
            "ps": {"latest": 6.0, "p25": 4.0, "p50": 5.0, "p75": 7.0, "percentile": 55.0, "samples": 36},
        }
    }
    peers = {
        "peers": {
            "source": "us_config",
            "pe": {"median": 25.0, "mean": 26.0},
            "pb": {"median": 4.0, "mean": 4.2},
            "ps": {"median": 5.0, "mean": 5.2},
            "peer_list": [{"code": "MSFT", "name": "Microsoft", "pe_ttm": 25.0, "pb": 4.0, "ps": 5.0}],
        },
        "source": "us_config",
        "basis": "peers",
        "peer_count": 1,
    }
    return {
        "get_us_profile": {
            "symbol": "TEST",
            "company_name": "Test Corp",
            "industry": "Technology",
            "source": "yfinance",
        },
        "get_us_financial_statements": {"symbol": "TEST", **statements},
        "get_us_valuation_snapshot": {"symbol": "TEST", **snapshot},
        "get_us_historical_percentile": hist,
        "get_us_peer_comparison": peers,
        "get_us_earnings_forecast": {
            "consensus_growth": 0.1,
            "research_reports": [{"year": date.today().year, "eps_avg": 25.0}],
            "target_mean_price": 120.0,
        },
    }


def test_peer_comparison_uses_config_peers_and_medians(monkeypatch):
    monkeypatch.setattr(
        tools,
        "_load_peer_config",
        lambda ticker: [{"code": "MSFT", "name": "Microsoft"}],
    )
    monkeypatch.setattr(
        tools,
        "yf_ticker_info",
        lambda ticker: {
            "currentPrice": 400.0,
            "trailingPE": 30.0,
            "priceToBook": 10.0,
            "priceToSalesTrailing12Months": 12.0,
        },
    )
    result = asyncio.run(tools.run_us_tool("get_us_peer_comparison", "TEST", {}))
    assert result["source"] == "us_config"
    assert result["peers"]["pe"]["median"] == 30.0
    assert result["peer_count"] == 1


def test_valuation_snapshot_computes_missing_multiples_from_sec(monkeypatch):
    monkeypatch.setattr(
        tools,
        "yf_ticker_info",
        lambda ticker: {
            "currentPrice": 100.0,
            "marketCap": 100_000_000_000.0,
            "source": "yahoo_chart_meta",
        },
    )
    monkeypatch.setattr(
        tools,
        "sec_fundamentals",
        lambda ticker: {
            "latest_shares": 1_000_000_000.0,
            "latest_equity": 50_000_000_000.0,
            "revenue_ttm": 100_000_000_000.0,
            "net_income_ttm": 20_000_000_000.0,
            "dividends_ttm": 2_000_000_000.0,
        },
    )
    result = asyncio.run(tools.run_us_tool("get_us_valuation_snapshot", "TEST", {}))
    assert result["pe_ttm"] == 5.0
    assert result["pb"] == 2.0
    assert result["ps"] == 1.0
    assert result["dividend_yield"] == 2.0


def test_historical_percentile_produces_stats(monkeypatch):
    monkeypatch.setattr(
        tools,
        "sec_fundamentals",
        lambda ticker: {"latest_shares": 1_000_000_000.0},
    )
    monkeypatch.setattr(edgar_module, "company_facts", lambda cik: {})
    monkeypatch.setattr(edgar_module, "ticker_to_cik", lambda ticker: 1)
    monkeypatch.setattr(tools, "yf_ticker_history", lambda *a, **k: _monthly_history())
    monkeypatch.setattr(
        tools,
        "_rolling_ttm",
        lambda records, as_of, field: {"net_income": 20_000_000_000.0, "revenue": 100_000_000_000.0}[field],
    )
    monkeypatch.setattr(tools, "_latest_equity", lambda records, as_of: 50_000_000_000.0)
    result = asyncio.run(tools.run_us_tool("get_us_historical_percentile", "TEST", {}))
    assert set(result["metrics"]) == {"pe_ttm", "pb", "ps"}
    assert result["metrics"]["pe_ttm"]["samples"] >= 20
    assert result["metrics"]["pe_ttm"]["p50"] > 0


def test_handler_metrics_keys_match_valuation_engine(monkeypatch):
    results = _base_results()
    cached_results = dict(results)

    async def fake_tool(name, ticker, market_data, metrics=None):
        if name == "estimate_fair_value":
            return tools._estimate_fair_value_tool(metrics or {}, ticker)
        return cached_results[name]

    monkeypatch.setattr(fundamental_service, "run_us_tool", fake_tool)
    output = asyncio.run(
        fundamental_service.UsFundamentalHandler().run(
            TaskRequest(query="TEST", inputs={"market_data": json.dumps({"latest": {"close": 100.0}})})
        )
    )
    payload = json.loads(output)
    analysis = payload["analysis"]
    metrics = analysis["metrics"]
    for key in (
        "symbol",
        "current_price",
        "total_shares",
        "eps_ttm",
        "bps",
        "sps_ttm",
        "fcf",
        "dps",
        "roe",
        "valuation",
        "historical",
        "industry_peers",
    ):
        assert key in metrics
    assert metrics["currency"] == "USD"
    assert metrics["fcf"] == 25_000_000_000.0
    assert "美元" in payload["report_section"]
    assert payload["summary"]["valuation_verdict"] in ("低估", "合理", "高估", "数据不足")


def test_handler_fails_when_core_tools_all_fail(monkeypatch):
    async def failing_tool(name, ticker, market_data, metrics=None):
        raise RuntimeError("boom")

    monkeypatch.setattr(fundamental_service, "run_us_tool", failing_tool)
    with pytest.raises(RuntimeError, match="核心美股工具"):
        asyncio.run(
            fundamental_service.UsFundamentalHandler().run(
                TaskRequest(query="TEST", inputs={})
            )
        )


def test_us_metrics_fill_sec_revenue_cagr_when_consensus_missing():
    results = _base_results()
    results["get_us_financial_statements"]["annual"] = [
        {"end": "2023-09-30", "revenue": 100_000_000_000.0},
        {"end": "2024-09-28", "revenue": 110_000_000_000.0},
        {"end": "2025-09-27", "revenue": 121_000_000_000.0},
    ]
    forecast = dict(results["get_us_earnings_forecast"])
    forecast["consensus_growth"] = None
    forecast["research_reports"] = []
    results["get_us_earnings_forecast"] = forecast
    metrics = fundamental_service._build_metrics(
        "TEST",
        {"latest": {"close": 100.0}},
        results,
    )
    assert metrics["revenue_growth_cagr"] == pytest.approx(0.10, abs=1e-9)
    assert metrics["growth_source"] == "sec_3y_revenue_cagr"
