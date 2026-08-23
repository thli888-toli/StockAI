"""Shared helpers for the StockAI agent plugins."""

from __future__ import annotations

import asyncio
import os
import json
import math
import re
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd


SYMBOL_RE = re.compile(r"^\d{6}$")

AKSHARE_COLUMN_ALIASES = {
    "日期": "date",
    "date": "date",
    "开盘": "open",
    "open": "open",
    "收盘": "close",
    "close": "close",
    "最高": "high",
    "high": "high",
    "最低": "low",
    "low": "low",
    "成交量": "volume",
    "volume": "volume",
    "成交额": "amount",
    "amount": "amount",
    "换手率": "turnover",
    "turnover": "turnover",
    "涨跌幅": "pct_change",
    "pct_change": "pct_change",
}


def validate_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip()
    if not SYMBOL_RE.fullmatch(symbol):
        raise ValueError("query must be a 6-digit A-share code, e.g. 600519")
    return symbol


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def json_loads(text: str, default: Any) -> Any:
    try:
        return json.loads(text or "")
    except Exception:
        return default


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return number


def normalize_akshare_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    rename: dict[str, str] = {}
    for column in frame.columns:
        key = str(column).strip()
        alias = AKSHARE_COLUMN_ALIASES.get(key)
        if alias:
            rename[column] = alias

    df = frame.rename(columns=rename)
    required = ["date", "open", "close", "high", "low", "volume"]
    for column in required:
        if column not in df.columns:
            raise ValueError(f"missing required market data column: {column}")

    df = df[[column for column in ("date", "open", "high", "low", "close", "volume", "amount", "turnover", "pct_change") if column in df.columns]].copy()
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover", "pct_change"):
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close", "volume"])
    df = df.sort_values("date").reset_index(drop=True)

    if "amount" not in df.columns:
        df["amount"] = 0.0
    if "turnover" not in df.columns:
        df["turnover"] = 0.0
    if "pct_change" not in df.columns:
        df["pct_change"] = df["close"].pct_change() * 100.0

    df[["amount", "turnover", "pct_change"]] = df[["amount", "turnover", "pct_change"]].fillna(0.0)
    return df


def compute_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return pd.DataFrame({"macd": macd, "signal": signal_line, "histogram": histogram})


def resample_ohlcv(df: pd.DataFrame, freq: str) -> pd.DataFrame:
    if df.empty:
        return df
    frame = df.set_index("date")
    aggregated = frame.resample(freq).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
            "amount": "sum",
            "turnover": "sum",
        }
    )
    return aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi.fillna(50.0)


def prepare_daily_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = normalize_akshare_frame(frame)
    if df.empty:
        return df

    close = df["close"]
    macd = compute_macd(close)
    df["macd"] = macd["macd"]
    df["macd_signal"] = macd["signal"]
    df["macd_histogram"] = macd["histogram"]
    df["rsi14"] = _rsi(close, 14)
    df["ma20"] = close.rolling(20).mean()
    df["ma66"] = close.rolling(66).mean()
    df["ma154"] = close.rolling(154).mean()
    df["ma250"] = close.rolling(250).mean()
    df["volatility20"] = df["pct_change"].rolling(20).std()
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["close_ma20_ratio"] = close / df["ma20"]
    df["close_ma66_ratio"] = close / df["ma66"]
    df["close_ma154_ratio"] = close / df["ma154"]
    df["close_ma250_ratio"] = close / df["ma250"]

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
        "turnover",
        "pct_change",
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
    ]
    df[numeric_columns] = df[numeric_columns].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["macd", "macd_signal", "macd_histogram", "rsi14", "ma20"])
    df[numeric_columns] = df[numeric_columns].fillna(0.0)
    return df


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text)
    if not compact:
        return set()
    if len(compact) == 1:
        return {compact}
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def normalize_title(title: str) -> str:
    title = (title or "").strip().lower()
    title = re.sub(r"\d{6}", "", title)
    title = re.sub(
        r"关于|公告|股份有限公司|有限公司|公司|披露|事项|的|及|与|暨",
        "",
        title,
    )
    title = re.sub(r"[^\w\u4e00-\u9fff]+", "", title)
    return title


def jaccard_similarity(left: str, right: str) -> float:
    left_set = _bigrams(normalize_title(left))
    right_set = _bigrams(normalize_title(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def parse_flexible_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc)
        except Exception:
            return None
    try:
        return pd.to_datetime(value, errors="raise").to_pydatetime()
    except Exception:
        return None


def dates_within_days(left: Any, right: Any, days: int = 2) -> bool:
    left_dt = parse_flexible_datetime(left)
    right_dt = parse_flexible_datetime(right)
    if left_dt is None or right_dt is None:
        return False
    if left_dt.tzinfo is None:
        left_dt = left_dt.replace(tzinfo=timezone.utc)
    if right_dt.tzinfo is None:
        right_dt = right_dt.replace(tzinfo=timezone.utc)
    return abs((left_dt - right_dt).total_seconds()) <= days * 86400


def direction_from_probability(probability: float) -> str:
    probability = _as_float(probability, 0.5)
    if probability > 0.52:
        return "up"
    if probability < 0.48:
        return "down"
    return "flat"


def confidence_from_probability(probability: float) -> float:
    probability = _as_float(probability, 0.5)
    return round(2 * abs(probability - 0.5), 6)


async def run_blocking(
    func: Callable[..., Any],
    *args: Any,
    timeout: float = 20.0,
    retries: int = 1,
    **kwargs: Any,
) -> Any:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(func, *args, **kwargs),
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(0.4 * (2**attempt))
    assert last_error is not None
    raise last_error


class TTLCache:
    def __init__(self, ttl_seconds: float) -> None:
        self.ttl_seconds = ttl_seconds
        self._values: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._values.get(key)
            if not item:
                return None
            created_at, value = item
            if now - created_at > self.ttl_seconds:
                self._values.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self._values[key] = (time.monotonic(), value)


def compact_news_summary(content: str, limit: int = 500) -> str:
    text = re.sub(r"<[^>]+>", "", content or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:limit]


@contextmanager
def disable_http_proxy():
    """Temporarily bypass any broken global HTTP(S) proxy for AkShare calls."""
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "NO_PROXY", "no_proxy")
    old_values = {key: os.environ.get(key) for key in keys}
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    try:
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
