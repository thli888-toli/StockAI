"""Shared helpers for the US-stock agents (yfinance / Stooq / SEC sources)."""

from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

from plugins.stock_common import json_dumps, json_loads, run_blocking


US_SYMBOL_RE = re.compile(
    r"^[A-Za-z][A-Za-z0-9]{0,9}([.\-][A-Za-z0-9]{1,2})?$"
)


def validate_us_symbol(symbol: str) -> str:
    """Validate and canonicalize a US ticker (uppercase; keeps . / - forms)."""
    text = (symbol or "").strip().upper()
    if not US_SYMBOL_RE.fullmatch(text):
        raise ValueError(
            "query must be a US ticker such as AAPL/MSFT/BRK.B (letters, digits, "
            "optional . or - class suffix)"
        )
    return text


def yahoo_symbol(symbol: str) -> str:
    """Yahoo Finance uses dashes for class shares (BRK.B -> BRK-B)."""
    return (symbol or "").strip().replace(".", "-")


def display_symbol(symbol: str) -> str:
    """Canonical display form used by reports and the portal."""
    return (symbol or "").strip().upper()


def _normalize_us_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize yfinance/Stooq OHLCV into the canonical feature frame."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    working = frame.copy()
    rename: dict[str, str] = {}
    for column in working.columns:
        key = str(column).strip().lower()
        aliases = {
            "date": "date",
            "open": "open",
            "high": "high",
            "low": "low",
            "close": "close",
            "volume": "volume",
        }
        if key in aliases:
            rename[column] = aliases[key]
    working = working.rename(columns=rename)
    required = ["date", "open", "high", "low", "close", "volume"]
    missing = [column for column in required if column not in working.columns]
    if missing:
        raise ValueError(f"missing required market data columns: {missing}")
    working = working[
        ["date", "open", "high", "low", "close", "volume"]
    ].copy()
    for column in ("open", "high", "low", "close", "volume"):
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working["date"] = pd.to_datetime(working["date"], errors="coerce")
    working = working.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    working = working.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    if working.empty:
        return working
    working["amount"] = working["close"] * working["volume"]
    working["turnover"] = 0.0
    working["pct_change"] = working["close"].pct_change() * 100.0
    working[["amount", "turnover", "pct_change"]] = working[
        ["amount", "turnover", "pct_change"]
    ].fillna(0.0)
    return working


def yf_ticker_info(ticker: str) -> dict[str, Any]:
    """Best-effort quote/info dict from yfinance (lazy import)."""
    import yfinance as yf

    try:
        info = yf.Ticker(yahoo_symbol(ticker)).info or {}
    except Exception:
        info = {}
    if not isinstance(info, dict):
        info = {}
    if info:
        return info
    return yahoo_chart_meta(ticker)


def yahoo_chart_meta(ticker: str) -> dict[str, Any]:
    """Minimal quote/profile fallback from the Yahoo chart meta block."""
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(ticker)}"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(
        url,
        params={"range": "1d", "interval": "1d"},
        headers=headers,
        timeout=25.0,
    )
    response.raise_for_status()
    payload = response.json()
    result = (payload or {}).get("chart", {}).get("result") or []
    meta = result[0].get("meta") if result else None
    if not isinstance(meta, dict):
        return {}
    return {
        "symbol": meta.get("symbol") or ticker,
        "currency": meta.get("currency") or "USD",
        "exchange": meta.get("exchangeName") or meta.get("fullExchangeName") or "",
        "fullExchangeName": meta.get("fullExchangeName") or "",
        "currentPrice": meta.get("regularMarketPrice"),
        "regularMarketPrice": meta.get("regularMarketPrice"),
        "previousClose": meta.get("previousClose") or meta.get("chartPreviousClose"),
        "source": "yahoo_chart_meta",
    }
    return info


def yahoo_chart_history(
    ticker: str,
    period: str = "3y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Yahoo v8 chart history (avoids yfinance rate-limit/crumb handling)."""
    import requests

    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol(ticker)}"
    params = {"range": period, "interval": interval, "includePrePost": "false"}
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    response = requests.get(url, params=params, headers=headers, timeout=25.0)
    response.raise_for_status()
    payload = response.json()
    result = (payload or {}).get("chart", {}).get("result") or []
    if not result:
        return pd.DataFrame()
    item = result[0]
    timestamps = item.get("timestamp") or []
    quote = (item.get("indicators") or {}).get("quote") or [{}]
    quote = quote[0] or {}
    if not timestamps or not quote.get("close"):
        return pd.DataFrame()
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(None),
            "open": quote.get("open"),
            "high": quote.get("high"),
            "low": quote.get("low"),
            "close": quote.get("close"),
            "volume": quote.get("volume"),
        }
    )
    return _normalize_us_frame(frame)


def yf_ticker_history(
    ticker: str,
    period: str = "3y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Daily/monthly OHLCV: Yahoo chart API first, yfinance as fallback."""
    try:
        frame = yahoo_chart_history(ticker, period, interval)
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    import yfinance as yf

    try:
        frame = yf.Ticker(yahoo_symbol(ticker)).history(
            period=period,
            interval=interval,
            auto_adjust=True,
            actions=False,
        )
    except Exception:
        return pd.DataFrame()
    return _normalize_us_frame(frame)


def stooq_daily_history(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Stooq CSV fallback for daily OHLCV (non-China, no API key)."""
    import requests

    symbol = f"{ticker.lower()}.us"
    url = (
        f"https://stooq.com/q/d/l/?s={symbol}"
        f"&d1={start_date.replace('-', '')}&d2={end_date.replace('-', '')}&i=d"
    )
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=25.0,
    )
    response.raise_for_status()
    frame = pd.read_csv(io.StringIO(response.text))
    if frame.empty or "Date" not in frame.columns:
        raise ValueError(f"Stooq returned no data for {ticker}")
    return _normalize_us_frame(frame)


async def fetch_us_history(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """yfinance first, Stooq fallback, wrapped for the async event loop."""
    try:
        frame = await run_blocking(
            yf_ticker_history,
            ticker,
            timeout=45.0,
            retries=1,
        )
        if frame is not None and not frame.empty:
            return frame
    except Exception:
        pass
    return await run_blocking(
        stooq_daily_history,
        ticker,
        start_date,
        end_date,
        timeout=40.0,
        retries=1,
    )


def us_frame_from_history(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure a raw history DataFrame is in canonical stock_common format."""
    return _normalize_us_frame(frame)


__all__ = [
    "US_SYMBOL_RE",
    "validate_us_symbol",
    "yahoo_symbol",
    "display_symbol",
    "yahoo_chart_history",
    "yf_ticker_info",
    "yf_ticker_history",
    "stooq_daily_history",
    "fetch_us_history",
    "us_frame_from_history",
]
