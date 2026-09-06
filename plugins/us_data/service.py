"""US market-data agent: yfinance/Stooq OHLCV + company profile."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from framework.config import US_STOCK_CACHE_DB
from framework.schemas import TaskRequest
from plugins.stock_cache import StockHistoryStore
from plugins.stock_common import (
    TTLCache,
    compute_macd,
    json_dumps,
    prepare_daily_features,
    resample_ohlcv,
    run_blocking,
)
from plugins.us_common import (
    fetch_us_history,
    validate_us_symbol,
    yf_ticker_history,
    yf_ticker_info,
)


LOOKBACK_YEARS = 3
MONTHLY_LOOKBACK_YEARS = 10
DATA_CACHE = TTLCache(ttl_seconds=900)


def _trend(histogram: float) -> str:
    if histogram > 0:
        return "bullish"
    if histogram < 0:
        return "bearish"
    return "flat"


def _macd_summary(close: pd.Series) -> dict[str, Any]:
    macd = compute_macd(close)
    if macd.empty:
        return {"macd": 0.0, "signal": 0.0, "histogram": 0.0, "trend": "flat"}
    last = macd.iloc[-1]
    return {
        "macd": round(float(last["macd"]), 6),
        "signal": round(float(last["signal"]), 6),
        "histogram": round(float(last["histogram"]), 6),
        "trend": _trend(float(last["histogram"])),
    }


FEATURE_COLUMNS = [
    "date",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "pct_change",
    "turnover",
    "macd",
    "macd_signal",
    "macd_histogram",
    "rsi14",
    "ma20",
    "ma66",
    "ma154",
    "ma250",
    "volatility20",
    "volume_ratio",
    "return_1d",
    "return_5d",
    "return_20d",
    "close_ma20_ratio",
    "close_ma66_ratio",
    "close_ma154_ratio",
    "close_ma250_ratio",
    "atr14",
    "bollinger_bandwidth",
    "bollinger_pctb",
    "close_high20_ratio",
    "close_low20_ratio",
    "amount_ratio",
]


class UsDataHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_us_symbol(request.query)
        cache_key = f"us_data:{symbol}:{date.today().isoformat()}"
        cached = DATA_CACHE.get(cache_key)
        if cached:
            return cached

        end_date = date.today()
        start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)
        history_store = StockHistoryStore(
            US_STOCK_CACHE_DB,
            table="us_stock_history",
            meta_table="us_stock_history_meta",
        )
        missing_ranges = await run_blocking(
            history_store.missing_ranges,
            symbol,
            "raw",
            start_date.isoformat(),
            end_date.isoformat(),
            timeout=5.0,
            retries=0,
        )
        for fetch_start, fetch_end in missing_ranges:
            raw = await fetch_us_history(symbol, fetch_start, fetch_end)
            if raw is not None and not raw.empty:
                await run_blocking(
                    history_store.merge,
                    symbol,
                    "raw",
                    raw,
                    timeout=10.0,
                    retries=0,
                )

        frame = await run_blocking(
            history_store.load,
            symbol,
            "raw",
            start_date.isoformat(),
            end_date.isoformat(),
            timeout=10.0,
            retries=0,
        )
        if len(frame) < 60:
            raise ValueError(
                "insufficient US history: need at least 60 trading rows "
                f"(got {len(frame)} for {symbol})"
            )

        history_meta = await run_blocking(
            history_store.get_meta,
            symbol,
            "raw",
            timeout=5.0,
            retries=0,
        )
        info = await run_blocking(yf_ticker_info, symbol, timeout=20.0, retries=0)
        daily = prepare_daily_features(frame)
        weekly = resample_ohlcv(frame, "W-FRI")
        monthly = resample_ohlcv(frame, "ME")
        latest_row = frame.iloc[-1]
        close = frame["close"]
        recent_daily = frame.tail(10).copy()
        recent_daily["date"] = recent_daily["date"].dt.strftime("%Y-%m-%d")

        features = daily[FEATURE_COLUMNS].copy()
        features["date"] = features["date"].dt.strftime("%Y-%m-%d")

        monthly_store = StockHistoryStore(
            US_STOCK_CACHE_DB,
            table="us_monthly_history",
            meta_table="us_monthly_history_meta",
        )
        monthly_history: list[dict[str, Any]] = []
        monthly_start = end_date - timedelta(days=365 * MONTHLY_LOOKBACK_YEARS)
        monthly_raw = await run_blocking(
            yf_ticker_history,
            symbol,
            period=f"{MONTHLY_LOOKBACK_YEARS}y",
            interval="1mo",
            timeout=40.0,
            retries=1,
        )
        if monthly_raw is not None and not monthly_raw.empty:
            await run_blocking(
                monthly_store.merge,
                symbol,
                "raw",
                monthly_raw,
                timeout=10.0,
                retries=0,
            )
        monthly_frame = await run_blocking(
            monthly_store.load,
            symbol,
            "raw",
            monthly_start.isoformat(),
            end_date.isoformat(),
            timeout=10.0,
            retries=0,
        )
        if monthly_frame.empty and not monthly.empty:
            await run_blocking(
                monthly_store.merge,
                symbol,
                "raw",
                monthly,
                timeout=10.0,
                retries=0,
            )
            monthly_frame = monthly
        if not monthly_frame.empty:
            view = monthly_frame.tail(180).copy()
            view["date"] = view["date"].dt.strftime("%Y-%m-%d")
            monthly_history = view.to_dict(orient="records")

        company_name = (
            str(info.get("longName") or info.get("shortName") or symbol).strip()
        )
        if company_name == symbol:
            try:
                from plugins.us_fundamental.edgar import sec_company_title

                company_name = (
                    await run_blocking(
                        sec_company_title,
                        symbol,
                        timeout=15.0,
                        retries=0,
                    )
                ) or symbol
            except Exception:
                pass
        industry = str(info.get("industry") or "").strip()
        sector = str(info.get("sector") or "").strip()
        if industry and sector:
            industry_text = f"{sector} / {industry}"
        else:
            industry_text = sector or industry

        payload = {
            "symbol": symbol,
            "company_name": company_name,
            "industry": industry_text,
            "exchange": str(info.get("exchange") or info.get("fullExchangeName") or ""),
            "currency": "USD",
            "market": "us",
            "as_of": latest_row["date"].strftime("%Y-%m-%d"),
            "latest": {
                "date": latest_row["date"].strftime("%Y-%m-%d"),
                "open": round(float(latest_row["open"]), 4),
                "high": round(float(latest_row["high"]), 4),
                "low": round(float(latest_row["low"]), 4),
                "close": round(float(latest_row["close"]), 4),
                "volume": float(latest_row["volume"]),
                "amount": float(latest_row.get("amount", 0.0)),
                "pct_change": round(float(latest_row.get("pct_change", 0.0)), 4),
            },
            "recent_daily": recent_daily[
                ["date", "open", "high", "low", "close", "volume", "amount", "pct_change"]
            ].to_dict(orient="records"),
            "stats": {
                "pct_change_20d": round(
                    float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
                    if len(close) > 21
                    else 0.0,
                    4,
                ),
                "pct_change_60d": round(
                    float((close.iloc[-1] / close.iloc[-61] - 1) * 100)
                    if len(close) > 61
                    else 0.0,
                    4,
                ),
            },
            "macd": {
                "daily": _macd_summary(daily["close"]),
                "weekly": (
                    _macd_summary(weekly["close"])
                    if not weekly.empty
                    else _macd_summary(daily["close"])
                ),
                "monthly": (
                    _macd_summary(monthly["close"])
                    if not monthly.empty
                    else _macd_summary(daily["close"])
                ),
            },
            "daily_features": features.to_dict(orient="records"),
            "monthly_history": monthly_history,
            "history_cache": history_meta or {},
        }
        result = json_dumps(payload)
        DATA_CACHE.set(cache_key, result)
        return result


def build_agent() -> UsDataHandler:
    return UsDataHandler()
