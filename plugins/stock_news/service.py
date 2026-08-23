from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pandas as pd

from framework.schemas import TaskRequest
from plugins.stock_common import (
    TTLCache,
    compact_news_summary,
    dates_within_days,
    disable_http_proxy,
    jaccard_similarity,
    json_dumps,
    json_loads,
    run_blocking,
    utcnow_iso,
    validate_symbol,
)


NEWS_CACHE = TTLCache(ttl_seconds=300)

EASTMONEY_COLUMNS = {
    "title": ("新闻标题", "标题", "title"),
    "content": ("新闻内容", "内容", "content"),
    "published_at": ("发布时间", "时间", "public_time", "published_at", "日期"),
    "url": ("新闻链接", "链接", "url"),
    "source": ("文章来源", "来源", "source"),
}

CNINFO_COLUMNS = {
    "title": ("公告标题", "标题", "title"),
    "content": ("公告内容", "内容", "content"),
    "published_at": ("公告时间", "发布时间", "时间", "published_at"),
    "url": ("公告链接", "链接", "url"),
    "source": ("来源", "source"),
}

TELEGRAPH_TITLE = ("标题", "title")
TELEGRAPH_CONTENT = ("内容", "content")
TELEGRAPH_TIME = ("发布时间", "时间", "published_at")
TELEGRAPH_DATE = ("发布日期", "日期")
TELEGRAPH_URL = ("链接", "url", "新闻链接")


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _frame_records(
    frame: pd.DataFrame,
    column_map: dict[str, tuple[str, ...]],
    source: str,
    source_type: str,
    limit: int,
) -> list[dict[str, str]]:
    if frame is None or frame.empty:
        return []

    title_column = _pick_column(frame, column_map["title"])
    if title_column is None:
        return []
    time_column = _pick_column(frame, column_map["published_at"])
    url_column = _pick_column(frame, column_map["url"])
    content_column = _pick_column(frame, column_map["content"])
    source_column = _pick_column(frame, column_map["source"])

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        title = str(row.get(title_column, "")).strip()
        if not title:
            continue
        url = str(row.get(url_column, "")).strip() if url_column else ""
        key = title.strip().lower() or url
        if key in seen:
            continue
        seen.add(key)
        published_at = str(row.get(time_column, "")).strip() if time_column else ""
        if time_column:
            try:
                published_at = pd.to_datetime(row[time_column]).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                published_at = str(row.get(time_column, "")).strip()
        summary = compact_news_summary(str(row.get(content_column, "")), limit=500) if content_column else ""
        record_source = str(row.get(source_column, "")).strip() if source_column else source
        records.append(
            {
                "title": title,
                "published_at": published_at,
                "url": url,
                "source": record_source or source,
                "summary": summary,
                "source_type": source_type,
            }
        )
        if len(records) >= limit:
            break
    return records


def _telegraph_records(
    frame: pd.DataFrame,
    source: str,
    source_type: str,
    keywords: list[str],
    limit: int = 10,
) -> list[dict[str, str]]:
    if frame is None or frame.empty:
        return []

    title_column = _pick_column(frame, TELEGRAPH_TITLE)
    if title_column is None:
        return []
    content_column = _pick_column(frame, TELEGRAPH_CONTENT)
    time_column = _pick_column(frame, TELEGRAPH_TIME)
    date_column = _pick_column(frame, TELEGRAPH_DATE)
    url_column = _pick_column(frame, TELEGRAPH_URL)

    records: list[dict[str, str]] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        title = str(row.get(title_column, "")).strip()
        content = str(row.get(content_column, "")).strip() if content_column else ""
        if not title:
            continue
        text = f"{title} {content}"
        if not any(keyword and keyword in text for keyword in keywords):
            continue
        key = title.strip().lower() or (
            str(row.get(url_column, "")).strip() if url_column else ""
        )
        if key in seen:
            continue
        seen.add(key)

        published_at = ""
        if date_column:
            date_text = str(row.get(date_column, "")).strip()
            if date_text and date_text.lower() != "nan":
                published_at = date_text
        if time_column:
            time_text = str(row.get(time_column, "")).strip()
            if time_text and time_text.lower() != "nan":
                published_at = f"{published_at} {time_text}".strip()
        if published_at:
            try:
                published_at = pd.to_datetime(published_at).strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass

        url = str(row.get(url_column, "")).strip() if url_column else ""
        records.append(
            {
                "title": title,
                "published_at": published_at,
                "url": url,
                "source": source,
                "summary": compact_news_summary(content, limit=500),
                "source_type": source_type,
            }
        )
        if len(records) >= limit:
            break
    return records


def _cross_validate(
    eastmoney_items: list[dict[str, str]],
    cninfo_items: list[dict[str, str]],
) -> tuple[list[dict[str, Any]], list[int]]:
    matched: list[dict[str, Any]] = []
    used_cninfo: set[int] = set()
    used_eastmoney: set[int] = set()

    for east_idx, east_item in enumerate(eastmoney_items):
        for cninfo_idx, cninfo_item in enumerate(cninfo_items):
            if cninfo_idx in used_cninfo:
                continue
            if not dates_within_days(
                east_item.get("published_at"),
                cninfo_item.get("published_at"),
                days=2,
            ):
                continue
            similarity = jaccard_similarity(east_item["title"], cninfo_item["title"])
            if similarity < 0.5:
                continue
            matched.append(
                {
                    "title": east_item["title"],
                    "eastmoney_url": east_item.get("url", ""),
                    "cninfo_url": cninfo_item.get("url", ""),
                    "published_at": east_item.get("published_at", cninfo_item.get("published_at", "")),
                    "confidence": "high",
                    "similarity": round(similarity, 4),
                }
            )
            used_cninfo.add(cninfo_idx)
            used_eastmoney.add(east_idx)
            break

    return matched, sorted(used_eastmoney)


class StockNewsHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        company_name = str(market_data.get("company_name") or symbol)
        industry = str(market_data.get("industry") or "")
        cache_key = f"stock_news:{symbol}:{industry}:{date.today().isoformat()}"
        cached = NEWS_CACHE.get(cache_key)
        if cached:
            return cached

        warnings: list[str] = []
        eastmoney_items = await self._fetch_eastmoney(symbol, industry, warnings)
        cninfo_items = await self._fetch_cninfo(symbol, warnings)
        secondary_media_items, secondary_media_source = await self._fetch_second_media(
            company_name, industry, warnings
        )

        if not eastmoney_items and not cninfo_items:
            raise RuntimeError("both Eastmoney and CNINFO news sources failed")

        cross_validated, _ = _cross_validate(eastmoney_items, cninfo_items)
        payload = {
            "symbol": symbol,
            "company_name": company_name,
            "industry": industry,
            "retrieved_at": utcnow_iso(),
            "eastmoney_news": eastmoney_items,
            "cninfo_disclosures": cninfo_items,
            "secondary_media_news": secondary_media_items,
            "secondary_media_source": secondary_media_source,
            "cross_validated": cross_validated,
            "source_counts": {
                "eastmoney": len(eastmoney_items),
                "cninfo": len(cninfo_items),
                "secondary_media": len(secondary_media_items),
                "cross_validated": len(cross_validated),
            },
            "warnings": warnings,
        }
        result = json_dumps(payload)
        NEWS_CACHE.set(cache_key, result)
        return result

    async def _fetch_eastmoney(
        self,
        symbol: str,
        industry: str,
        warnings: list[str],
    ) -> list[dict[str, str]]:
        records: list[dict[str, str]] = []
        for keyword in (symbol, industry):
            if not keyword:
                continue
            try:
                frame = await run_blocking(self._call_stock_news_em, keyword, timeout=20.0, retries=1)
                records.extend(_frame_records(frame, EASTMONEY_COLUMNS, "东方财富", "eastmoney_news", 10))
            except Exception as exc:
                warnings.append(f"Eastmoney query failed for '{keyword}': {exc}")
        return records

    async def _fetch_cninfo(self, symbol: str, warnings: list[str]) -> list[dict[str, str]]:
        end_date = date.today()
        start_date = end_date - timedelta(days=30)
        try:
            frame = await run_blocking(
                self._call_cninfo,
                symbol,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d"),
                timeout=30.0,
                retries=1,
            )
            if frame is None or frame.empty:
                warnings.append("CNINFO returned no disclosures for this period")
                return []
            return _frame_records(frame, CNINFO_COLUMNS, "巨潮资讯", "cninfo_disclosure", 20)
        except Exception as exc:
            warnings.append(f"CNINFO query failed: {exc}")
            return []

    async def _fetch_second_media(
        self,
        company_name: str,
        industry: str,
        warnings: list[str],
    ) -> tuple[list[dict[str, str]], str]:
        keywords = [key.strip() for key in (company_name, industry) if key and key.strip()]
        if not keywords:
            return [], ""
        for label, fetcher, source_type in (
            ("财联社", self._call_cls, "cls_news"),
            ("同花顺", self._call_ths, "ths_news"),
        ):
            try:
                frame = await run_blocking(fetcher, timeout=20.0, retries=1)
                records = _telegraph_records(frame, label, source_type, keywords, limit=10)
                if records:
                    return records, label
                warnings.append(f"{label} telegraph returned no matching items for {keywords}")
            except Exception as exc:
                warnings.append(f"{label} telegraph query failed: {exc}")
        return [], ""

    @staticmethod
    def _call_stock_news_em(keyword: str):
        import akshare as ak

        with disable_http_proxy():
            try:
                return ak.stock_news_em(symbol=keyword)
            except TypeError:
                return ak.stock_news_em(stock=keyword)

    @staticmethod
    def _call_cninfo(symbol: str, start_date: str, end_date: str):
        import akshare as ak

        with disable_http_proxy():
            try:
                return ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=symbol,
                    market="沪深京",
                    start_date=start_date,
                    end_date=end_date,
                )
            except TypeError:
                return ak.stock_zh_a_disclosure_report_cninfo(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                )

    @staticmethod
    def _call_cls():
        import akshare as ak

        with disable_http_proxy():
            return ak.stock_info_global_cls(symbol="全部")

    @staticmethod
    def _call_ths():
        import akshare as ak

        with disable_http_proxy():
            return ak.stock_info_global_ths()


def build_agent() -> StockNewsHandler:
    return StockNewsHandler()
