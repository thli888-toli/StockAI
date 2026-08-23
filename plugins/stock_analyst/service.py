from __future__ import annotations

import asyncio
import json
import os
import re
from typing import Any

from framework.llm import llm_configured, llm_reply
from framework.schemas import TaskRequest
from plugins.stock_common import json_loads, validate_symbol


HORIZONS = ("5d", "15d", "1w", "1mo")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "150"))


def _extract_json_block(text: str) -> dict[str, Any] | None:
    match = re.search(r"```(?:json|JSON)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    if match:
        candidate = match.group(1)
    else:
        start = (text or "").find("{")
        end = (text or "").rfind("}")
        if start < 0 or end <= start:
            return None
        candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def _signal_from_llm(parsed: dict[str, Any] | None) -> dict[str, Any]:
    if not parsed:
        return {}
    signal = parsed.get("signal") or parsed.get("signals") or {}
    if not isinstance(signal, dict):
        return {}
    return signal


def _fallback_signal(market_data: dict[str, Any], quant: dict[str, Any]) -> dict[str, Any]:
    macd = market_data.get("macd", {})
    daily_macd = macd.get("daily", {}) or {}
    macd_direction = "flat"
    if daily_macd.get("trend") == "bullish":
        macd_direction = "up"
    elif daily_macd.get("trend") == "bearish":
        macd_direction = "down"

    signal: dict[str, Any] = {}
    horizons = quant.get("horizons", {})
    for horizon in HORIZONS:
        quant_signal = horizons.get(horizon, {}) or {}
        signal[horizon] = {
            "direction": quant_signal.get("direction", macd_direction),
            "confidence": float(quant_signal.get("confidence", 0.0)),
            "rationale": "基于 MACD 与量化模型概率的确定性回退信号。",
        }
    return signal


def cross_validate(
    llm_signal: dict[str, Any],
    quant: dict[str, Any],
) -> dict[str, Any]:
    horizons = quant.get("horizons", {})
    details: dict[str, Any] = {}
    overall_rank = 2
    for horizon in HORIZONS:
        llm_direction = str((llm_signal.get(horizon, {}) or {}).get("direction", "flat")).lower()
        quant_item = horizons.get(horizon, {}) or {}
        quant_direction = str(quant_item.get("direction", "flat")).lower()
        quant_confidence = float(quant_item.get("confidence", 0.0))
        if llm_direction == quant_direction:
            conviction = "high" if quant_confidence >= 0.6 else "medium"
            rank = 2 if conviction == "high" else 1
        else:
            conviction = "low"
            rank = 0
        details[horizon] = {
            "llm_direction": llm_direction,
            "quant_direction": quant_direction,
            "quant_confidence": quant_confidence,
            "conviction": conviction,
        }
        overall_rank = min(overall_rank, rank)

    overall = "low" if overall_rank == 0 else ("medium" if overall_rank == 1 else "high")
    return {"overall": overall, "horizons": details}


def _direction_cn(value: Any) -> str:
    return {"up": "上行", "down": "下行", "flat": "中性"}.get(str(value).lower(), str(value))


def _conviction_cn(value: Any) -> str:
    return {"high": "高", "medium": "中", "low": "低"}.get(str(value).lower(), str(value))


def _format_horizon(horizon: str, item: dict[str, Any]) -> str:
    return (
        f"- {horizon}：方向 {_direction_cn(item.get('direction', 'flat'))}，"
        f"上行概率 {item.get('up_probability', 0.5)}，置信度 {item.get('confidence', 0)}"
    )


def _normalize_summary(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    overall = str(value.get("overall") or "").strip().lower()
    text = str(value.get("text") or "").strip()
    if overall not in ("bullish", "bearish", "neutral") or not text:
        return None
    return {"overall": overall, "text": text}


def _fallback_summary(signal: dict[str, Any]) -> dict[str, Any]:
    labels = {"5d": "5日", "15d": "15日", "1w": "周线", "1mo": "月线"}
    directions: list[str] = []
    parts: list[str] = []
    for horizon in HORIZONS:
        direction = str((signal.get(horizon, {}) or {}).get("direction", "flat")).lower()
        directions.append(direction)
        parts.append(f"{labels[horizon]}{_direction_cn(direction)}")
    up_count = directions.count("up")
    down_count = directions.count("down")
    if up_count > down_count:
        overall = "bullish"
        stance = "偏多"
    elif down_count > up_count:
        overall = "bearish"
        stance = "偏空"
    else:
        overall = "neutral"
        stance = "中性"
    text = f"量化模型观点：{'，'.join(parts)}。整体{stance}。"
    return {"overall": overall, "text": text}


def _fallback_report(
    symbol: str,
    market_data: dict[str, Any],
    news: dict[str, Any],
    quant: dict[str, Any],
    signal: dict[str, Any],
    cross: dict[str, Any],
) -> str:
    company_name = market_data.get("company_name") or symbol
    industry = market_data.get("industry") or "未知"
    latest = market_data.get("latest", {}) or {}
    macd = market_data.get("macd", {}) or {}
    source_counts = news.get("source_counts", {}) or {}
    horizons = quant.get("horizons", {}) or {}
    backtest = quant.get("backtest", {}) or {}
    weekly_backtest = quant.get("weekly_backtest", {}) or {}

    lines = [
        f"# {company_name}（{symbol}）A 股分析报告",
        "",
        "> 确定性回退报告：LLM 分析不可用或无法解析。",
        "",
        "## 一、数据概览",
        f"- 代码：{symbol}",
        f"- 公司：{company_name}",
        f"- 行业：{industry}",
        f"- 数据日期：{market_data.get('as_of', '')}",
        f"- 最新收盘价：{latest.get('close', '')}",
        f"- 最新涨跌幅：{latest.get('pct_change', '')}%",
        "",
        "## 二、MACD 技术信号",
    ]
    period_names = {"daily": "日线", "weekly": "周线", "monthly": "月线"}
    for period in ("daily", "weekly", "monthly"):
        item = macd.get(period, {}) or {}
        lines.append(
            f"- {period_names[period]}：MACD {item.get('macd', '')}，信号线 {item.get('signal', '')}，"
            f"柱状值 {item.get('histogram', '')}，趋势 {item.get('trend', '')}"
        )

    lines.extend(
        [
            "",
            "## 三、新闻与公告催化",
            f"- 东方财富新闻：{source_counts.get('eastmoney', 0)} 条",
            f"- 巨潮资讯公告：{source_counts.get('cninfo', 0)} 条",
            f"- 第二媒体源（{news.get('secondary_media_source') or '无'}）：{source_counts.get('secondary_media', 0)} 条",
            f"- 交叉验证配对：{source_counts.get('cross_validated', 0)} 条",
            f"- 数据警告：{len(news.get('warnings', []))} 条",
            "",
            "## 四、量化模型信号（LSTM）",
            f"- 日线前推 AUC：{backtest.get('walk_forward_auc')}；样本数：{backtest.get('sample_count', 0)}",
            f"- 周线前推 AUC：{weekly_backtest.get('walk_forward_auc')}；样本数：{weekly_backtest.get('sample_count', 0)}",
        ]
    )
    for horizon in HORIZONS:
        lines.append(_format_horizon(horizon, horizons.get(horizon, {}) or {}))

    lines.extend(
        [
            "",
            "## 五、LLM 与量化模型交叉验证",
            f"- 综合置信度：{_conviction_cn(cross.get('overall', 'medium'))}",
        ]
    )
    for horizon, detail in (cross.get("horizons", {}) or {}).items():
        lines.append(
            f"- {horizon}：LLM {_direction_cn(detail.get('llm_direction'))}，"
            f"量化 {_direction_cn(detail.get('quant_direction'))}，"
            f"一致性 {_conviction_cn(detail.get('conviction'))}"
        )

    lines.extend(
        [
            "",
            "## 六、情景展望",
            "- 请结合上述多空一致性与置信度综合判断；低/中置信度下不宜给出强方向性结论。",
            "- 官方公告或重大市场变化后应重新评估。",
            "",
            "## 七、风险提示",
            "- 历史数据与模型输出不代表未来表现。",
            "- 东方财富等媒体信息可能不完整或延迟；巨潮资讯官方公告为主要证据。",
            "- 仍需关注市场风险、流动性风险、政策风险及个股特有风险。",
            "",
            "## 八、免责声明",
            "本报告仅供信息与研究参考，不构成任何投资建议。",
        ]
    )
    return "\n".join(lines)


class StockAnalystHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        news = json_loads(request.inputs.get("news", ""), {})
        quant = json_loads(request.inputs.get("quant", ""), {})

        parsed: dict[str, Any] | None = None
        llm_report = ""
        llm_summary: dict[str, Any] | None = None
        if llm_configured():
            prompt = self._build_prompt(symbol, market_data, news, quant)
            try:
                response = await asyncio.wait_for(
                    llm_reply(self._system_prompt(), prompt, max_tokens=2000),
                    timeout=LLM_TIMEOUT_SECONDS,
                )
                parsed = _extract_json_block(response)
                if parsed:
                    llm_report = str(parsed.get("report") or "")
                    llm_summary = _normalize_summary(parsed.get("summary"))
            except Exception:
                parsed = None
                llm_report = ""
                llm_summary = None

        signal = _signal_from_llm(parsed) if parsed else _fallback_signal(market_data, quant)
        cross = cross_validate(signal, quant)
        summary = llm_summary or _fallback_summary(signal)

        if llm_report:
            report = llm_report
            if "免责声明" not in report:
                report += "\n\n## 八、免责声明\n本报告仅供信息与研究参考，不构成任何投资建议。\n"
        else:
            report = _fallback_report(symbol, market_data, news, quant, signal, cross)

        return json.dumps({"report": report, "summary": summary}, ensure_ascii=False)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是一名谨慎的 A 股研究助手。只能使用提供的数据，不得编造事实或保证收益。"
            "巨潮资讯官方公告是主要证据，东方财富等媒体是次要解读。"
            "当量化模型前推 AUC 接近 0.5（尤其低于 0.5）时，必须对高概率方向信号折价处理，避免给出强方向性建议。"
            "输出纯中文 Markdown 报告，并清楚区分高/中/低交叉验证置信度。"
        )

    @staticmethod
    def _build_prompt(
        symbol: str,
        market_data: dict[str, Any],
        news: dict[str, Any],
        quant: dict[str, Any],
    ) -> str:
        recent_daily: list[dict[str, Any]] = []
        for row in (market_data.get("recent_daily") or [])[-5:]:
            recent_daily.append(
                {
                    "date": row.get("date", ""),
                    "open": row.get("open"),
                    "high": row.get("high"),
                    "low": row.get("low"),
                    "close": row.get("close"),
                    "volume": row.get("volume"),
                    "pct_change": row.get("pct_change"),
                }
            )

        indicators: dict[str, Any] = {}
        daily_features = market_data.get("daily_features") or []
        if daily_features:
            last_row = daily_features[-1] or {}
            indicator_keys = [
                "rsi14",
                "ma20",
                "ma66",
                "ma154",
                "ma250",
                "volatility20",
                "return_1d",
                "return_5d",
                "return_20d",
                "volume_ratio",
                "close_ma20_ratio",
                "close_ma66_ratio",
                "close_ma154_ratio",
                "close_ma250_ratio",
                "macd",
                "macd_signal",
                "macd_histogram",
            ]
            indicators = {key: last_row.get(key) for key in indicator_keys if key in last_row}

        compact_market = {
            "symbol": market_data.get("symbol", symbol),
            "company_name": market_data.get("company_name", ""),
            "industry": market_data.get("industry", ""),
            "as_of": market_data.get("as_of", ""),
            "latest": market_data.get("latest", {}),
            "macd": market_data.get("macd", {}),
            "stats": market_data.get("stats", {}),
            "recent_daily": recent_daily,
            "indicators": indicators,
        }

        def compact_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
            compacted: list[dict[str, Any]] = []
            for item in (items or [])[:limit]:
                compacted.append(
                    {
                        "title": item.get("title", ""),
                        "published_at": item.get("published_at", ""),
                        "summary": item.get("summary", ""),
                    }
                )
            return compacted

        compact_news = {
            "source_counts": news.get("source_counts", {}),
            "cross_validated": news.get("cross_validated", []),
            "eastmoney_news": compact_items(news.get("eastmoney_news", []), 5),
            "cninfo_disclosures": compact_items(news.get("cninfo_disclosures", []), 10),
            "secondary_media_news": compact_items(news.get("secondary_media_news", []), 5),
            "secondary_media_source": news.get("secondary_media_source", ""),
            "warnings": news.get("warnings", []),
        }
        return (
            "返回一个 JSON 对象，包含 `signal`、`summary` 和 `report` 三个字段。\n"
            "`signal` 必须包含 `5d`、`15d`、`1w`、`1mo`，每项包含 `direction`（up/down/flat）、"
            "`confidence`（0-1）和 `rationale`。\n"
            "`summary` 必须包含 `overall`（bullish/bearish/neutral）和 `text`（一句中文总结）。\n"
            "`report` 必须为纯中文 Markdown，包含数据概览、MACD 技术信号、新闻与公告催化、"
            "LSTM 量化信号、LLM 与量化模型交叉验证、情景展望、风险提示和免责声明。\n\n"
            f"SYMBOL: {symbol}\n"
            f"MARKET_DATA: {json.dumps(compact_market, ensure_ascii=False)}\n"
            f"NEWS: {json.dumps(compact_news, ensure_ascii=False)}\n"
            f"QUANT: {json.dumps(quant, ensure_ascii=False)}\n"
        )


def build_agent() -> StockAnalystHandler:
    return StockAnalystHandler()
