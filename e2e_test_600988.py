"""Standalone end-to-end integration test for A-share code 600988.

This script exercises the real stock_data -> stock_news -> stock_quant ->
stock_analyst pipeline using live AkShare endpoints. LLM analysis is forced to
its deterministic fallback to keep the integration test fast and offline-safe.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import TaskRequest  # noqa: E402
from plugins.stock_analyst import service as analyst_service  # noqa: E402
from plugins.stock_analyst.service import StockAnalystHandler  # noqa: E402
from plugins.stock_data.service import StockDataHandler  # noqa: E402
from plugins.stock_news.service import StockNewsHandler  # noqa: E402
from plugins.stock_quant.service import StockQuantHandler  # noqa: E402


SYMBOL = "600988"


def print_section(title: str) -> None:
    print(f"\n===== {title} =====")


async def main() -> int:
    print_section("stock_data")
    try:
        market_data_text = await StockDataHandler().run(TaskRequest(query=SYMBOL))
        market_data = json.loads(market_data_text)
        print(
            "OK",
            market_data.get("company_name"),
            "rows in feature matrix:",
            len(market_data.get("daily_features", [])),
            "history:",
            market_data.get("history_cache"),
        )
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED stock_data: {exc}")
        return 1

    print_section("stock_news")
    try:
        news_text = await StockNewsHandler().run(
            TaskRequest(query=SYMBOL, inputs={"market_data": market_data_text})
        )
        news = json.loads(news_text)
        print("OK", "source_counts:", news.get("source_counts"), "warnings:", news.get("warnings"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED stock_news: {exc}")
        return 1

    print_section("stock_quant")
    try:
        quant_text = await StockQuantHandler().run(
            TaskRequest(query=SYMBOL, inputs={"market_data": market_data_text})
        )
        quant = json.loads(quant_text)
        print("OK", "horizons:", quant.get("horizons"), "backtest:", quant.get("backtest"))
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED stock_quant: {exc}")
        return 1

    print_section("stock_analyst (deterministic fallback)")
    original_llm_configured = analyst_service.llm_configured
    analyst_service.llm_configured = lambda: False
    try:
        report = await StockAnalystHandler().run(
            TaskRequest(
                query=SYMBOL,
                inputs={
                    "market_data": market_data_text,
                    "news": news_text,
                    "quant": quant_text,
                },
            )
        )
        analyst_payload = json.loads(report)
        analyst_report = analyst_payload.get("report") or ""
        if SYMBOL not in analyst_report or "免责声明" not in analyst_report:
            raise RuntimeError("final report is missing symbol or disclaimer")
        print("OK report length:", len(analyst_report), "summary:", analyst_payload.get("summary"))
        print(analyst_report[:500])
    except Exception as exc:  # noqa: BLE001
        print(f"FAILED stock_analyst: {exc}")
        return 1
    finally:
        analyst_service.llm_configured = original_llm_configured

    print_section("RESULT")
    print(f"E2E PASS for {SYMBOL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
