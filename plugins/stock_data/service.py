from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from framework.config import STOCK_CACHE_DB
from plugins.stock_cache import StockHistoryStore
from plugins.stock_common import (
    TTLCache,
    compute_macd,
    disable_http_proxy,
    json_dumps,
    normalize_akshare_frame,
    prepare_daily_features,
    resample_ohlcv,
    run_blocking,
    validate_symbol,
)
from framework.schemas import TaskRequest


LOOKBACK_YEARS = 3
DATA_CACHE = TTLCache(ttl_seconds=900)


def _trend(histogram: float) -> str:
    if histogram > 0:
        return "bullish"
    if histogram < 0:
        return "bearish"
    return "flat"


def _macd_summary(close) -> dict[str, Any]:
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


class StockDataHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        cache_key = f"stock_data:{symbol}:{date.today().isoformat()}"
        cached = DATA_CACHE.get(cache_key)
        if cached:
            return cached

        end_date = date.today()
        start_date = end_date - timedelta(days=365 * LOOKBACK_YEARS)
        start_str = start_date.isoformat()
        end_str = end_date.isoformat()

        history_store = StockHistoryStore(STOCK_CACHE_DB)
        missing_ranges = await run_blocking(
            history_store.missing_ranges,
            symbol,
            "qfq",
            start_str,
            end_str,
            timeout=5.0,
            retries=0,
        )
        for fetch_start, fetch_end in missing_ranges:
            raw = await run_blocking(
                self._fetch_daily,
                symbol,
                fetch_start.replace("-", ""),
                fetch_end.replace("-", ""),
                timeout=30.0,
                retries=1,
            )
            normalized = normalize_akshare_frame(raw)
            if normalized.empty:
                continue
            await run_blocking(
                history_store.merge,
                symbol,
                "qfq",
                normalized,
                timeout=10.0,
                retries=0,
            )

        frame = await run_blocking(
            history_store.load,
            symbol,
            "qfq",
            start_str,
            end_str,
            timeout=10.0,
            retries=0,
        )
        if len(frame) < 60:
            raise ValueError("insufficient daily history: need at least 60 trading rows")

        history_meta = await run_blocking(
            history_store.get_meta,
            symbol,
            "qfq",
            timeout=5.0,
            retries=0,
        )
        company_info = await self._fetch_company_info_optional(symbol)
        daily = prepare_daily_features(frame)
        weekly = resample_ohlcv(frame, "W-FRI")
        monthly = resample_ohlcv(frame, "ME")

        latest_row = frame.iloc[-1]
        close = frame["close"]
        recent_daily = frame.tail(10).copy()
        recent_daily["date"] = recent_daily["date"].dt.strftime("%Y-%m-%d")

        feature_columns = [
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
        ]
        features = daily[feature_columns].copy()
        features["date"] = features["date"].dt.strftime("%Y-%m-%d")

        payload = {
            "symbol": symbol,
            "company_name": company_info.get("company_name", symbol),
            "industry": company_info.get("industry", ""),
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
                "weekly": _macd_summary(weekly["close"]) if not weekly.empty else _macd_summary(daily["close"]),
                "monthly": _macd_summary(monthly["close"]) if not monthly.empty else _macd_summary(daily["close"]),
            },
            "daily_features": features.to_dict(orient="records"),
            "history_cache": history_meta or {},
        }

        result = json_dumps(payload)
        DATA_CACHE.set(cache_key, result)
        return result

    @staticmethod
    def _fetch_daily(symbol: str, start_date: str, end_date: str):
        import akshare as ak

        with disable_http_proxy():
            try:
                return ak.stock_zh_a_hist(
                    symbol=symbol,
                    period="daily",
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )
            except Exception:
                tx_symbol = StockDataHandler._tx_symbol(symbol)
                return ak.stock_zh_a_hist_tx(
                    symbol=tx_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    adjust="qfq",
                )

    @staticmethod
    def _tx_symbol(symbol: str) -> str:
        if symbol.startswith(("6", "9")):
            return f"sh{symbol}"
        return f"sz{symbol}"

    @staticmethod
    def _xq_symbol(symbol: str) -> str:
        if symbol.startswith(("6", "9")):
            return f"SH{symbol}"
        return f"SZ{symbol}"

    @staticmethod
    def _fetch_company_info(symbol: str) -> dict[str, str]:
        import akshare as ak

        merged: dict[str, str] = {"company_name": symbol, "industry": ""}
        name_found = False
        industry_found = False
        for source in (
            StockDataHandler._company_info_from_em,
            StockDataHandler._company_info_from_cninfo,
            StockDataHandler._company_info_from_xq,
        ):
            info = source(ak, symbol)
            if not info:
                continue
            name = (info.get("company_name") or "").strip()
            industry = (info.get("industry") or "").strip()
            if not name_found and name and name != symbol:
                merged["company_name"] = name
                name_found = True
            if not industry_found and industry:
                merged["industry"] = industry
                industry_found = True
            if name_found and industry_found:
                break
        return merged

    @staticmethod
    def _company_info_from_em(ak, symbol: str) -> dict[str, str]:
        try:
            with disable_http_proxy():
                frame = ak.stock_individual_info_em(symbol=symbol)
        except Exception:
            return {"company_name": symbol, "industry": ""}
        if frame is None or frame.empty:
            return {"company_name": symbol, "industry": ""}
        item_column = "item"
        value_column = "value"
        if item_column not in frame.columns or value_column not in frame.columns:
            return {"company_name": symbol, "industry": ""}
        mapping = dict(zip(frame[item_column].astype(str), frame[value_column].astype(str)))
        return {
            "company_name": mapping.get("股票简称") or mapping.get("股票名称") or symbol,
            "industry": mapping.get("行业") or mapping.get("所属行业") or "",
        }

    @staticmethod
    def _company_info_from_xq(ak, symbol: str) -> dict[str, str] | None:
        xq_symbol = StockDataHandler._xq_symbol(symbol)
        try:
            with disable_http_proxy():
                frame = ak.stock_individual_basic_info_xq(symbol=xq_symbol)
        except Exception:
            return None
        if frame is None or frame.empty:
            return None
        item_column = "item"
        value_column = "value"
        if item_column not in frame.columns or value_column not in frame.columns:
            return None
        mapping = dict(zip(frame[item_column].astype(str), frame[value_column].astype(str)))
        company_name = (
            mapping.get("股票简称")
            or mapping.get("公司简称")
            or mapping.get("org_name_cn")
            or mapping.get("name")
            or symbol
        )
        industry = (
            mapping.get("行业")
            or mapping.get("所属行业")
            or mapping.get("industry")
            or mapping.get("industry_sw")
        )
        if not industry:
            industry = StockDataHandler._extract_affiliate_industry(mapping.get("affiliate_industry"))
        if company_name == symbol and not industry:
            return None
        return {"company_name": company_name, "industry": industry}

    @staticmethod
    def _extract_affiliate_industry(value: Any) -> str:
        if not value:
            return ""
        if isinstance(value, dict):
            return str(value.get("ind_name") or value.get("industry") or "")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return ""
            try:
                import json

                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return str(parsed.get("ind_name") or parsed.get("industry") or "")
            except Exception:
                import re

                match = re.search(r"['\"]ind_name['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
                if match:
                    return match.group(1)
                match = re.search(r"['\"]industry['\"]\s*:\s*['\"]([^'\"]+)['\"]", text)
                if match:
                    return match.group(1)
        return ""

    @staticmethod
    def _company_info_from_cninfo(ak, symbol: str) -> dict[str, str] | None:
        try:
            with disable_http_proxy():
                frame = ak.stock_profile_cninfo(symbol=symbol)
        except Exception:
            return None
        if frame is None or frame.empty:
            return None
        row = frame.iloc[0]

        def _get(*names: str) -> str:
            for name in names:
                if name in frame.columns:
                    value = row[name]
                    if value is not None and str(value).strip():
                        return str(value).strip()
            return ""

        company_name = (
            _get("A股简称", "公司简称", "证券简称")
            or _get("公司名称")
            or symbol
        )
        industry = _get("所属行业", "行业", "行业分类")
        if company_name == symbol and not industry:
            return None
        return {"company_name": company_name, "industry": industry}

    async def _fetch_company_info_optional(self, symbol: str) -> dict[str, str]:
        try:
            return await run_blocking(self._fetch_company_info, symbol, timeout=12.0, retries=0)
        except Exception:
            return {"company_name": symbol, "industry": ""}


def build_agent() -> StockDataHandler:
    return StockDataHandler()
