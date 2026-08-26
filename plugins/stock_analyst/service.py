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
    direction = _direction_cn(item.get("direction", "flat"))
    probability = float(item.get("up_probability", 0.5)) * 100
    confidence = float(item.get("confidence", 0.0))
    return (
        f"- {horizon}：上行概率 {probability:.2f}%，方向 {direction}，置信度 {confidence:.3f}。"
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


async def generate_llm_summary(
    market_data: dict[str, Any],
    quant: dict[str, Any],
    use_llm: bool = True,
) -> dict[str, Any]:
    """Return a compact summary (overall + text) with optional LLM call."""
    signal = _fallback_signal(market_data, quant)
    if use_llm and llm_configured():
        compact = {
            "as_of": market_data.get("as_of", ""),
            "latest": market_data.get("latest", {}),
            "macd": market_data.get("macd", {}),
            "indicators": market_data.get("indicators", {}),
        }
        prompt = (
            "只返回一个 JSON 对象，包含 `summary`：`overall`（bullish/bearish/neutral）"
            "和 `text`（一句中文总结）。只依据提供的行情与量化信号判断，不要输出报告。\n\n"
            f"MARKET_DATA: {json.dumps(compact, ensure_ascii=False)}\n"
            f"QUANT: {json.dumps(quant, ensure_ascii=False)}\n"
        )
        try:
            response = await asyncio.wait_for(
                llm_reply(_system_prompt(), prompt, max_tokens=120),
                timeout=LLM_TIMEOUT_SECONDS,
            )
            parsed = _extract_json_block(response)
            summary = _normalize_summary(parsed.get("summary")) if parsed else None
            if summary:
                return summary
        except Exception:
            pass
    return _fallback_summary(signal)


def _fallback_report(
    symbol: str,
    market_data: dict[str, Any],
    news: dict[str, Any],
    quant: dict[str, Any],
    signal: dict[str, Any],
    cross: dict[str, Any],
    fundamental: dict[str, Any] | None = None,
) -> str:
    company_name = market_data.get("company_name") or symbol
    industry = market_data.get("industry") or "未知"
    latest = market_data.get("latest", {}) or {}
    macd = market_data.get("macd", {}) or {}
    source_counts = news.get("source_counts", {}) or {}
    horizons = quant.get("horizons", {}) or {}
    backtest = quant.get("backtest", {}) or {}
    weekly_backtest = quant.get("weekly_backtest", {}) or {}
    stats = market_data.get("stats", {}) or {}
    features = market_data.get("daily_features") or []
    last = features[-1] if features else {}
    recent = (market_data.get("recent_daily") or [])[-5:]

    def fnum(key: str, default: float = 0.0) -> float:
        try:
            return float(last.get(key, default))
        except (TypeError, ValueError):
            return default

    try:
        amount_yi = float(latest.get("amount", 0.0)) / 1e8
    except (TypeError, ValueError):
        amount_yi = 0.0
    recent_text = "、".join(
        f"{row.get('date', '')[-5:]}:{row.get('pct_change', '')}%" for row in recent
    )

    lines = [
        f"# {company_name}（{symbol}）A股市场观察报告",
        "",
        "## 数据概览",
        f"- 截至{market_data.get('as_of', '')}，收盘价{latest.get('close', '')}元，"
        f"当日上涨约{latest.get('pct_change', '')}%（return_1d），"
        f"成交额{amount_yi:.2f}亿元，换手率{fnum('turnover'):.2f}%。",
        f"- 近5日涨幅{fnum('return_5d') * 100:.2f}%，"
        f"近20日涨幅{float(stats.get('pct_change_20d') or 0.0):.2f}%，"
        f"近60日涨幅{float(stats.get('pct_change_60d') or 0.0):.2f}%；"
        f"RSI14为{fnum('rsi14'):.2f}，20日历史波动率{fnum('volatility20') * 100:.2f}%，"
        f"量比{fnum('volume_ratio'):.3f}。",
        f"- 价格相对均线：收盘/MA20={fnum('close_ma20_ratio'):.3f}，"
        f"收盘/MA66={fnum('close_ma66_ratio'):.3f}，"
        f"收盘/MA154={fnum('close_ma154_ratio'):.3f}，"
        f"收盘/MA250={fnum('close_ma250_ratio'):.3f}。",
        f"- 最近5个交易日：{recent_text}。",
        "",
    ]
    if fundamental:
        section = fundamental.get("report_section")
        if section:
            lines.append(str(section).strip())
        else:
            lines.append("")
            lines.append("- 基本面数据暂不可用，估值区间未纳入本报告。")
    else:
        lines.extend(["", "## 基本面与估值", "- 基本面数据暂不可用，估值区间未纳入本报告。"])
    lines.extend(
        [
            "",
        "## MACD 技术信号",
        ]
    )
    period_names = {"daily": "日线", "weekly": "周线", "monthly": "月线"}
    for period in ("daily", "weekly", "monthly"):
        item = macd.get(period, {}) or {}
        trend = "多头" if item.get("trend") == "bullish" else "空头" if item.get("trend") == "bearish" else "中性"
        lines.append(
            f"- {period_names[period]}MACD为{item.get('macd', '')}，信号线{item.get('signal', '')}，"
            f"柱值{item.get('histogram', '')}，趋势为{trend}。"
        )
    lines.append(
        "- 需结合RSI与均线偏离判断：若RSI超买或价格明显高于MA20，短线技术性回撤压力上升。"
    )

    lines.extend(
        [
            "",
            "## 新闻与公告催化",
            f"- 东方财富新闻：{source_counts.get('eastmoney', 0)} 条；巨潮公告：{source_counts.get('cninfo', 0)} 条；"
            f"第二媒体（{news.get('secondary_media_source') or '无'}）：{source_counts.get('secondary_media', 0)} 条；"
            f"交叉验证配对：{source_counts.get('cross_validated', 0)} 条。",
            f"- 公告证据缺口：本周期CNINFO披露为{source_counts.get('cninfo', 0)}，"
            "新闻证据主要来自媒体，交叉验证证据有限。",
            "",
            "## LSTM+LightGBM 量化信号",
        ]
    )
    for horizon in HORIZONS:
        lines.append(_format_horizon(horizon, horizons.get(horizon, {}) or {}))
    lines.append(
        f"- 回测：日线 walk-forward AUC {backtest.get('walk_forward_auc')}"
        f"（样本{backtest.get('sample_count', 0)}）；周线 walk-forward AUC {weekly_backtest.get('walk_forward_auc')}"
        f"（样本{weekly_backtest.get('sample_count', 0)}）。"
    )
    lines.append("- 提示：AUC 接近 0.5，量化方向信号存在失效风险，1w 接近随机水平，需折价使用。")

    lines.extend(
        [
            "",
            "## LLM 与量化模型交叉验证",
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
            "## 情景展望",
            "- 强势延续：若守住近期支撑并继续放量，月线多头与板块催化可能推动1个月维度缓慢抬升。",
            "- 技术回调：若跌破近期支撑或出现放量滞涨，可能触发获利盘回吐，向均线方向回归。",
            "- 宽幅震荡：若换手率与波动率维持高位，短期日内波动可能显著扩大。",
            "",
            "## 风险提示",
            "- 短期涨幅过大、RSI超买，技术指标存在钝化与均值回归风险。",
            "- 公告与新闻催化若主要来自次级来源且原始披露缺失，信息可靠性打折。",
            "- LSTM模型AUC若低于0.5，回测无统计优势，不能作为交易依据。",
            "- 板块受金价、宏观政策、汇率等外部因素影响，存在快速反转可能。",
            "",
            "## 免责声明",
            "本报告仅基于所提供的数据生成，不构成投资建议或收益保证。市场有风险，投资者应独立判断并自行承担决策风险。",
        ]
    )
    return "\n".join(lines)


class StockAnalystHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        news = json_loads(request.inputs.get("news", ""), {})
        quant = json_loads(request.inputs.get("quant", ""), {})
        fundamental = json_loads(request.inputs.get("fundamental", ""), {})

        parsed: dict[str, Any] | None = None
        llm_report = ""
        llm_summary: dict[str, Any] | None = None
        if llm_configured():
            prompt = self._build_prompt(symbol, market_data, news, quant, fundamental)
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
                report += (
                    "\n\n## 免责声明\n"
                    "本报告仅基于所提供的数据生成，不构成投资建议或收益保证。"
                    "市场有风险，投资者应独立判断并自行承担决策风险。\n"
                )
        else:
            report = _fallback_report(symbol, market_data, news, quant, signal, cross, fundamental)

        return json.dumps({"report": report, "summary": summary}, ensure_ascii=False)

    @staticmethod
    def _system_prompt() -> str:
        return (
            "你是一名谨慎的 A 股研究助手。只能使用提供的数据，不得编造事实或保证收益。"
            "巨潮资讯官方公告是主要证据，东方财富等媒体是次要解读。"
            "当量化模型前推 AUC 接近 0.5（尤其低于 0.5）时，必须对高概率方向信号折价处理，避免给出强方向性建议。"
            "基本面与估值章节应直接采用 stock_fundamental 提供的估值区间、方法与结论，不得修改其中数字。"
            "输出纯中文 Markdown 报告，并清楚区分高/中/低交叉验证置信度。"
            "报告必须严格使用固定模板的章节结构。"
        )

    @staticmethod
    def _build_prompt(
        symbol: str,
        market_data: dict[str, Any],
        news: dict[str, Any],
        quant: dict[str, Any],
        fundamental: dict[str, Any] | None = None,
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

        compact_fundamental: dict[str, Any] = {}
        if isinstance(fundamental, dict) and fundamental:
            analysis = fundamental.get("analysis") or {}
            valuation = analysis.get("valuation") or {}
            compact_fundamental = {
                "fair_value_range": valuation.get("fair_value_range"),
                "verdict": valuation.get("verdict"),
                "available_methods": valuation.get("available_methods"),
                "assumptions": valuation.get("assumptions"),
                "warnings": analysis.get("warnings"),
                "report_section": fundamental.get("report_section"),
                "summary": fundamental.get("summary"),
            }
        return (
            "返回一个 JSON 对象，包含 `signal`、`summary` 和 `report` 三个字段。\n"
            "`signal` 必须包含 `5d`、`15d`、`1w`、`1mo`，每项包含 `direction`（up/down/flat）、"
            "`confidence`（0-1）和 `rationale`。\n"
            "`summary` 必须包含 `overall`（bullish/bearish/neutral）和 `text`（一句中文总结）。\n"
            "`report` 必须严格使用以下固定模板，标题与章节名完全一致，内容按提供数据填充：\n"
            "# {公司名称}（{代码}）A股市场观察报告\n"
            "## 数据概览\n"
            "## 基本面与估值\n"
            "## MACD 技术信号\n"
            "## 新闻与公告催化\n"
            "## LSTM+LightGBM 量化信号\n"
            "## LLM 与量化模型交叉验证\n"
            "## 情景展望\n"
            "## 风险提示\n"
            "## 免责声明\n"
            "基本面与估值章节必须直接采用 FUNDAMENTAL 中 `report_section` 的内容"
            "（标题为 `## 基本面与估值`），不得改写其中数字；若 FUNDAMENTAL 为空，"
            "该章节只输出一行：基本面数据暂不可用，估值区间未纳入本报告。\n"
            "LSTM+LightGBM 量化信号段必须严格使用以下格式：\n"
            "- 5d：上行概率 xx.xx%，方向 上行/下行/中性，置信度 x.xxx。\n"
            "- 15d：上行概率 xx.xx%，方向 上行/下行/中性，置信度 x.xxx。\n"
            "- 1w：上行概率 xx.xx%，方向 上行/下行/中性，置信度 x.xxx。\n"
            "- 1mo：上行概率 xx.xx%，方向 上行/下行/中性，置信度 x.xxx。\n"
            "- 回测：日线 walk-forward AUC x.xxx（样本n）；周线 walk-forward AUC x.xxx（样本n）。\n"
            "- 提示：AUC 接近 0.5，量化方向信号存在失效风险，1w 接近随机水平，需折价使用。\n"
            "其中第五段必须严格使用以下格式：\n"
            "- 综合置信度：高/中/低\n"
            "- 5d：LLM 上行/下行/中性，量化 上行/下行/中性，一致性 高/中/低\n"
            "- 15d：LLM 上行/下行/中性，量化 上行/下行/中性，一致性 高/中/低\n"
            "- 1w：LLM 上行/下行/中性，量化 上行/下行/中性，一致性 高/中/低\n"
            "- 1mo：LLM 上行/下行/中性，量化 上行/下行/中性，一致性 高/中/低\n\n"
            f"SYMBOL: {symbol}\n"
            f"MARKET_DATA: {json.dumps(compact_market, ensure_ascii=False)}\n"
            f"NEWS: {json.dumps(compact_news, ensure_ascii=False)}\n"
            f"QUANT: {json.dumps(quant, ensure_ascii=False)}\n"
            f"FUNDAMENTAL: {json.dumps(compact_fundamental, ensure_ascii=False)}\n"
        )


def build_agent() -> StockAnalystHandler:
    return StockAnalystHandler()
