"""Handler for the us_fundamental agent.

Assembles normalized metrics in the same shape as the A-share fundamental
agent, reuses the multi-method fair-value engine with USD configs, and emits a
deterministic (or optional LLM) report section.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from typing import Any

from framework.llm import llm_configured, llm_reply
from framework.schemas import TaskRequest
from plugins.stock_common import json_dumps, json_loads
from plugins.us_common import validate_us_symbol
from plugins.us_fundamental.tools import run_us_tool


LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "150"))
CORE_TOOLS = {
    "get_us_financial_statements",
    "get_us_valuation_snapshot",
    "get_us_historical_percentile",
}
TOOL_SEQUENCE = [
    "get_us_profile",
    "get_us_financial_statements",
    "get_us_valuation_snapshot",
    "get_us_historical_percentile",
    "get_us_peer_comparison",
    "get_us_earnings_forecast",
]


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _per_share(value: Any, shares: float | None) -> float | None:
    if value is None or not shares or shares <= 0:
        return None
    return float(value) / shares


def _revenue_cagr(annual_records: list[dict[str, Any]]) -> float | None:
    """2-3 year revenue CAGR from SEC annual records."""
    annuals = sorted(
        [
            (str(item.get("end") or ""), _num(item.get("revenue")))
            for item in (annual_records or [])
            if item.get("end") and _num(item.get("revenue")) is not None
        ],
        key=lambda item: item[0],
    )
    if len(annuals) < 2:
        return None
    latest_end, latest_revenue = annuals[-1]
    latest_year = int(latest_end[:4])
    older = next(
        (item for item in reversed(annuals[:-1]) if latest_year - int(item[0][:4]) >= 2),
        annuals[-2],
    )
    older_end, older_revenue = older
    years = latest_year - int(older_end[:4])
    if years <= 0 or older_revenue <= 0:
        return None
    try:
        return (latest_revenue / older_revenue) ** (1.0 / years) - 1.0
    except ZeroDivisionError:
        return None


def _fmt_bn(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value / 1e9:.{digits}f}"


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _build_metrics(
    ticker: str,
    market_data: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    profile = results.get("get_us_profile") or {}
    statements = results.get("get_us_financial_statements") or {}
    snapshot = results.get("get_us_valuation_snapshot") or {}
    historical = results.get("get_us_historical_percentile") or {}
    peers = results.get("get_us_peer_comparison") or {}
    forecast = results.get("get_us_earnings_forecast") or {}

    current_price = _num(market_data.get("latest", {}).get("close"))
    if current_price is None:
        current_price = _num(snapshot.get("close"))
    total_shares = _num(snapshot.get("total_shares"))

    revenue_ttm = _num(statements.get("revenue_ttm"))
    net_income_ttm = _num(statements.get("net_income_ttm"))
    ocf_ttm = _num(statements.get("ocf_ttm"))
    capex_ttm = _num(statements.get("capex_ttm"))
    dividends_ttm = _num(statements.get("dividends_ttm"))
    equity = _num(statements.get("latest_equity"))

    eps_ttm = _per_share(net_income_ttm, total_shares)
    if eps_ttm is None:
        eps_ttm = _num(snapshot.get("trailing_eps"))
    bps = _per_share(equity, total_shares)
    sps_ttm = _per_share(revenue_ttm, total_shares)
    fcf = ocf_ttm - capex_ttm if ocf_ttm is not None and capex_ttm is not None else None
    dps = _per_share(dividends_ttm, total_shares)
    payout_ratio = None
    if dividends_ttm is not None and net_income_ttm and net_income_ttm > 0:
        payout_ratio = min(1.0, max(0.0, dividends_ttm / net_income_ttm))
    roe = None
    if net_income_ttm is not None and equity and equity > 0:
        roe = net_income_ttm / equity

    consensus_growth = _num(forecast.get("consensus_growth"))
    forecast_reports = forecast.get("research_reports") or []
    forecast_eps = _num(
        forecast_reports[0].get("eps_avg")
        if forecast_reports
        else snapshot.get("forward_eps")
    )
    forecast_year = forecast_reports[0].get("year") if forecast_reports else None

    valuation = {
        "pe_ttm": _num(snapshot.get("pe_ttm")),
        "pe_forward": _num(snapshot.get("pe_forward")),
        "pb": _num(snapshot.get("pb")),
        "ps": _num(snapshot.get("ps")),
        "dividend_yield": _num(snapshot.get("dividend_yield")),
    }
    if valuation.get("dividend_yield") is None and dps and current_price:
        valuation["dividend_yield"] = dps / current_price * 100.0

    growth_yoy = _num(snapshot.get("earnings_growth"))
    revenue_yoy = _num(snapshot.get("revenue_growth"))
    revenue_cagr = _revenue_cagr(statements.get("annual") or [])
    annual_records = statements.get("annual") or []
    earnings_annuals = sorted(
        [
            (str(item.get("end") or ""), _num(item.get("net_income")))
            for item in annual_records
            if item.get("end") and _num(item.get("net_income")) is not None
        ],
        key=lambda item: item[0],
    )
    earnings_cagr = None
    if len(earnings_annuals) >= 2:
        latest_end, latest_earnings = earnings_annuals[-1]
        latest_year = int(latest_end[:4])
        older = next(
            (
                item
                for item in reversed(earnings_annuals[:-1])
                if latest_year - int(item[0][:4]) >= 2
            ),
            earnings_annuals[-2],
        )
        years = latest_year - int(older[0][:4])
        if years > 0 and older[1] and older[1] > 0 and latest_earnings > 0:
            earnings_cagr = (latest_earnings / older[1]) ** (1.0 / years) - 1.0
    return {
        "symbol": ticker,
        "company_name": str(profile.get("company_name") or market_data.get("company_name") or ticker),
        "industry_name": str(profile.get("industry") or market_data.get("industry") or ""),
        "currency": "USD",
        "current_price": current_price,
        "total_shares": total_shares,
        "eps_ttm": eps_ttm,
        "bps": bps,
        "sps_ttm": sps_ttm,
        "fcf": fcf,
        "ocf_ttm": ocf_ttm,
        "dps": dps,
        "roe": roe,
        "gross_margin": _num(snapshot.get("gross_margin")),
        "net_profit_yoy": growth_yoy,
        "payout_ratio": payout_ratio,
        "revenue_growth_yoy": revenue_yoy,
        "revenue_growth_cagr": revenue_cagr,
        "earnings_growth_cagr": earnings_cagr,
        "growth_source": (
            "sec_3y_revenue_cagr"
            if consensus_growth is None and revenue_cagr is not None
            else "consensus_and_history"
            if consensus_growth is not None
            else ""
        ),
        "forecast_growth": consensus_growth,
        "forecast_eps": forecast_eps,
        "forecast_year": forecast_year,
        "valuation": valuation,
        "historical": {key: stats for key, stats in (historical.get("metrics") or {}).items()},
        "industry_peers": peers.get("peers") or {},
        "industry_bench": peers.get("industry") or {},
        "statements": {
            "revenue_ttm": revenue_ttm,
            "net_income_ttm": net_income_ttm,
            "ocf_ttm": ocf_ttm,
            "capex_ttm": capex_ttm,
            "dividends_ttm": dividends_ttm,
            "latest_equity": equity,
            "latest_report_date": statements.get("latest_report_date"),
            "source": statements.get("source") or "",
        },
    }


def _data_quality(metrics: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    missing = [
        key
        for key, value in {
            "eps_ttm": metrics.get("eps_ttm"),
            "bps": metrics.get("bps"),
            "fcf": metrics.get("fcf"),
            "dps": metrics.get("dps"),
            "total_shares": metrics.get("total_shares"),
            "current_price": metrics.get("current_price"),
            "roe": metrics.get("roe"),
            "historical_percentile": bool(metrics.get("historical")),
            "peer_comparison": bool(metrics.get("industry_peers")),
            "growth_anchor": (
                metrics.get("forecast_growth") or metrics.get("revenue_growth_cagr")
            ),
        }.items()
        if not value
    ]
    return {
        "missing": missing,
        "methods_available": [],
        "warnings": warnings,
        "currency": "USD",
    }


def _report_section(analysis: dict[str, Any]) -> str:
    metrics = analysis.get("metrics") or {}
    valuation = analysis.get("valuation") or {}
    statements = metrics.get("statements") or {}
    snapshot = analysis.get("valuation_snapshot") or {}
    warnings = analysis.get("warnings") or []
    quality = analysis.get("data_quality") or {}
    name = analysis.get("company_name") or analysis.get("symbol") or ""
    symbol = analysis.get("symbol") or ""

    lines = [
        f"## 基本面与估值（美股 {name} {symbol}）",
        f"- 币种：美元。数据来源：SEC EDGAR（财务报表）+ Yahoo Finance（行情/一致预期）"
        f"；最新财报截至 {statements.get('latest_report_date') or '—'}。",
        "- 财务概览（TTM）："
        f"营收 {_fmt_bn(statements.get('revenue_ttm'))} 亿美元，"
        f"净利润 {_fmt_bn(statements.get('net_income_ttm'))} 亿美元，"
        f"经营现金流 {_fmt_bn(statements.get('ocf_ttm'))} 亿美元，"
        f"资本开支 {_fmt_bn(statements.get('capex_ttm'))} 亿美元。",
        "- 每股数据：EPS-TTM " + _fmt_price(metrics.get("eps_ttm"))
        + " 美元，BPS " + _fmt_price(metrics.get("bps"))
        + " 美元，SPS-TTM " + _fmt_price(metrics.get("sps_ttm"))
        + " 美元；ROE " + _fmt_pct(metrics.get("roe"))
        + "，毛利率 " + _fmt_pct(metrics.get("gross_margin"))
        + "。",
    ]
    snapshot_pe = snapshot.get("pe_ttm")
    hist_pe = (metrics.get("historical") or {}).get("pe_ttm") or {}
    lines.append(
        "- 估值水平：当前 PE-TTM "
        + _fmt_price(snapshot_pe)
        + f"（近3年分位 {hist_pe.get('percentile', '—')}%），PB "
        + _fmt_price(snapshot.get("pb"))
        + "，PS-TTM " + _fmt_price(snapshot.get("ps"))
        + "；股息率 " + _fmt_pct(valuation.get("dividend_yield") or metrics.get("valuation", {}).get("dividend_yield"), 2)
        + "。",
    )
    peers = metrics.get("industry_peers") or {}
    if peers.get("median") or peers.get("pe"):
        source_label = {
            "us_config": "配置可比公司",
            "us_llm": "LLM 建议可比公司",
        }.get(peers.get("source"), "可比公司")
        lines.append(
            f"- 可比公司倍数（{source_label}，{len(peers.get('peer_list') or [])} 家）："
            f"PE 中位数 {_fmt_price(((peers.get('pe') or {}).get('median')))}、"
            f"PB 中位数 {_fmt_price(((peers.get('pb') or {}).get('median')))}、"
            f"PS 中位数 {_fmt_price(((peers.get('ps') or {}).get('median')))}。"
        )
    forecast = analysis.get("forecast") or {}
    if forecast.get("consensus_growth") is not None:
        lines.append(
            f"- 一致预期：盈利增速 {_fmt_pct(forecast.get('consensus_growth'))}，"
            f"目标均价 {_fmt_price(forecast.get('target_mean_price'))} 美元"
            + (f"（{forecast.get('recommendation_key')}）" if forecast.get("recommendation_key") else "")
            + "。"
        )
    elif metrics.get("revenue_growth_cagr") is not None:
        lines.append(
            "- 增长假设来源：Yahoo 一致预期暂不可用，采用 SEC 近 3 年营收 CAGR "
            + _fmt_pct(metrics.get("revenue_growth_cagr"))
            + "（历史口径，非一致预期）。"
        )

    fair_value = valuation.get("fair_value_range") or {}
    verdict = valuation.get("verdict") or {}
    if fair_value.get("mid") is not None:
        method_names = {"relative": "相对估值", "dcf": "DCF", "ddm": "股息折现"}
        used = [
            method_names.get(name, name)
            for name in (valuation.get("available_methods") or [])
        ]
        lines.append(
            f"- 合理股价估算（纳入{'、'.join(used) if used else '可用方法'}）：区间 "
            f"{_fmt_price(fair_value.get('low'))}–{_fmt_price(fair_value.get('high'))} 美元，"
            f"中枢 {_fmt_price(fair_value.get('mid'))} 美元。"
        )
        per_method = valuation.get("per_method") or {}
        for key, label in (("relative", "相对估值"), ("dcf", "DCF"), ("ddm", "股息折现")):
            item = per_method.get(key) or {}
            if item.get("available"):
                if key == "relative":
                    peer_names = item.get("peer_names") or []
                    lines.append(
                        f"  - 相对估值：{_fmt_price(item.get('price'))} 美元"
                        f"（目标倍数：{item.get('basis') or '可比公司/历史'}；"
                        f"可比公司：{'、'.join(peer_names[:8]) if peer_names else '历史/行业'}）。"
                    )
                elif key == "dcf":
                    lines.append(
                        f"  - DCF：{_fmt_price(item.get('price'))} 美元"
                        f"（折现率 {item.get('discount_rate', 0.10):.0%}，"
                        f"永续增速 {item.get('terminal_growth', 0.02):.0%}，"
                        f"假设增速 {_fmt_pct(item.get('growth'))}）。"
                    )
                else:
                    lines.append(
                        f"  - 股息折现：{_fmt_price(item.get('price'))} 美元"
                        f"（假设增速 {_fmt_pct(item.get('growth'))}）。"
                    )
        for item in valuation.get("excluded_methods") or []:
            lines.append(
                f"  - 未纳入：{method_names.get(item.get('method'), item.get('method', ''))}"
                f"（{item.get('reason', '不适用')}）。"
            )
        lines.append(
            f"- 当前股价 {_fmt_price(metrics.get('current_price'))} 美元，"
            f"相对估值中枢偏离 {_fmt_pct(verdict.get('margin'), 1)}，"
            f"判断：{verdict.get('label', '—')}。"
        )
        if (
            metrics.get("current_price") is not None
            and fair_value.get("high") is not None
            and float(metrics["current_price"]) > float(fair_value["high"])
        ):
            above_high = (
                float(metrics["current_price"]) / float(fair_value["high"]) - 1.0
            )
            lines.append(
                f"  - 现价同时高于估值区间上沿 "
                f"{above_high * 100:+.1f}%（上沿 {_fmt_price(fair_value.get('high'))} 美元）。"
            )
    else:
        lines.append("- 合理股价估算：数据不足，无法给出估值区间。")

    if warnings:
        lines.append("- 数据质量提示：" + "；".join(warnings[:5]) + "。")
    missing = quality.get("missing") or []
    if missing:
        lines.append(f"- 数据缺口：{'、'.join(missing)}。")
    lines.append(
        "- 风险提示：以上为基于 SEC 财报与公开一致预期的估算区间，非精确价格，"
        "不构成投资建议或收益保证；美股数据（Yahoo/Stooq/SEC）可能存在延迟或修订。"
    )
    return "\n".join(lines)


async def _llm_section(analysis: dict[str, Any]) -> str | None:
    if not llm_configured():
        return None
    prompt = (
        "只输出一个 Markdown 章节，标题固定为 `## 基本面与估值（美股）`。只允许使用下面数据，"
        "不得编造数字；币种全部为美元；必须包含估值区间、方法与关键假设、低估/合理/高估判断"
        "与风险提示。\n\n"
        f"DATA: {json.dumps(analysis, ensure_ascii=False, default=str)[:12000]}"
    )
    try:
        return await asyncio.wait_for(
            llm_reply(
                "你是谨慎的美股基本面研究助手，输出纯中文 Markdown，保留风险提示。",
                prompt,
                max_tokens=1200,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception:
        return None


class UsFundamentalHandler:
    async def run(self, request: TaskRequest) -> str:
        ticker = validate_us_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        warnings: list[str] = []
        results: dict[str, Any] = {}
        core_failures = 0
        for name in TOOL_SEQUENCE:
            try:
                results[name] = await run_us_tool(name, ticker, market_data)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{name} 失败: {exc}")
                if name in CORE_TOOLS:
                    core_failures += 1
        if core_failures == len(CORE_TOOLS):
            raise RuntimeError("核心美股工具（财报/估值快照/历史分位）全部失败")

        metrics = _build_metrics(ticker, market_data, results)
        valuation: dict[str, Any] = {}
        try:
            valuation = await run_us_tool(
                "estimate_fair_value",
                ticker,
                market_data,
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"estimate_fair_value 失败: {exc}")
            valuation = {"error": str(exc)}

        analysis = {
            "symbol": ticker,
            "company_name": metrics.get("company_name") or ticker,
            "industry": metrics.get("industry_name") or "",
            "currency": "USD",
            "as_of": results.get("get_us_valuation_snapshot", {}).get("data_date")
            or date.today().isoformat(),
            "profile": results.get("get_us_profile"),
            "statements": results.get("get_us_financial_statements"),
            "valuation_snapshot": results.get("get_us_valuation_snapshot"),
            "historical": results.get("get_us_historical_percentile"),
            "peer_comparison": results.get("get_us_peer_comparison"),
            "forecast": results.get("get_us_earnings_forecast"),
            "metrics": metrics,
            "valuation": valuation,
            "warnings": warnings,
            "data_quality": _data_quality(metrics, warnings),
        }
        section = await _llm_section(analysis)
        if not section:
            section = _report_section(analysis)
        fair_value = valuation.get("fair_value_range") or {}
        summary = {
            "valuation_verdict": (valuation.get("verdict") or {}).get("label", "数据不足"),
            "fair_value_range": fair_value,
            "current_price": metrics.get("current_price"),
            "as_of": analysis["as_of"],
            "currency": "USD",
        }
        return json_dumps(
            {"analysis": analysis, "report_section": section, "summary": summary}
        )


def build_agent() -> UsFundamentalHandler:
    return UsFundamentalHandler()
