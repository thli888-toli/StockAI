"""Handler for the stock_fundamental agent.

Runs the in-plugin tool pipeline in a fixed order, assembles fundamental
metrics, estimates a fair-value range, and emits a Markdown report section
plus a compact summary for downstream agents.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
from datetime import date
from typing import Any

from framework.llm import llm_configured, llm_reply
from framework.schemas import TaskRequest
from plugins.stock_common import json_dumps, json_loads, validate_symbol
from plugins.stock_fundamental.tools import run_tool


LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "150"))
CORE_TOOLS = {
    "get_financial_statements",
    "get_financial_indicators",
    "get_valuation_snapshot",
}
TOOL_SEQUENCE = [
    "get_company_profile",
    "get_financial_statements",
    "get_financial_indicators",
    "get_financial_abstract",
    "get_valuation_snapshot",
    "get_historical_valuation_percentile",
    "get_industry_valuation_comparison",
    "get_earnings_forecast",
]


def _ttm(records: list[dict[str, Any]], key: str) -> float | None:
    """Trailing-twelve-month value from cumulative per-period records."""
    valid = [
        item
        for item in records
        if item.get("report_date") and item.get(key) is not None
    ]
    if not valid:
        return None
    valid = sorted(valid, key=lambda item: item["report_date"])
    latest = valid[-1]
    latest_value = float(latest[key])
    latest_month_day = latest["report_date"][5:]
    if latest_month_day == "12-31":
        return latest_value
    if len(valid) < 2:
        return None
    prior_year = next(
        (item for item in reversed(valid[:-1]) if item["report_date"][5:] == latest_month_day),
        None,
    )
    full_year = next(
        (
            item
            for item in reversed(valid)
            if item["report_date"][5:] == "12-31"
            and item["report_date"] < latest["report_date"]
        ),
        None,
    )
    if prior_year is not None and full_year is not None:
        return latest_value + float(full_year[key]) - float(prior_year[key])
    if len(valid) >= 5:
        diffs = [
            float(valid[index][key]) - float(valid[index - 1][key])
            for index in range(1, len(valid))
        ]
        if len(diffs) >= 4:
            return sum(diffs[-4:])
    return None


def _latest_annual(records: list[dict[str, Any]], key: str) -> float | None:
    annuals = [
        item
        for item in records
        if item.get("report_date", "").endswith("12-31") and item.get(key) is not None
    ]
    if not annuals:
        return None
    latest = max(annuals, key=lambda item: item["report_date"])
    return float(latest[key])


def _fraction(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if abs(number) > 1.0:
        return number / 100.0
    return number


def _revenue_cagr(income_records: list[dict[str, Any]]) -> float | None:
    annuals = sorted(
        [
            item
            for item in income_records
            if item.get("report_date", "").endswith("12-31")
            and item.get("operate_income") is not None
        ],
        key=lambda item: item["report_date"],
    )
    if len(annuals) < 2:
        return None
    latest = annuals[-1]
    latest_year = int(latest["report_date"][:4])
    older = next(
        (
            item
            for item in reversed(annuals[:-1])
            if latest_year - int(item["report_date"][:4]) >= 2
        ),
        None,
    )
    if older is None:
        older = annuals[-2]
    years = latest_year - int(older["report_date"][:4])
    if years <= 0:
        return None
    try:
        ratio = float(latest["operate_income"]) / float(older["operate_income"])
        if ratio <= 0:
            return None
        return ratio ** (1.0 / years) - 1.0
    except ZeroDivisionError:
        return None


def _forecast_growth(reports: list[dict[str, Any]], eps_ttm: float | None) -> float | None:
    if not eps_ttm or eps_ttm <= 0:
        return None
    year = date.today().year
    values = [
        float(item[f"eps_{year}"] / eps_ttm - 1.0)
        for item in reports
        if item.get(f"eps_{year}") is not None
    ]
    if not values:
        return None
    return float(statistics.median(values))


def _guidance_growth(guidance: list[dict[str, Any]]) -> float | None:
    values = [
        float(item["change_pct"] / 100.0)
        for item in guidance
        if item.get("change_pct") is not None
    ]
    if not values:
        return None
    return float(statistics.median(values))


def _build_metrics(
    symbol: str,
    market_data: dict[str, Any],
    results: dict[str, Any],
) -> dict[str, Any]:
    statements = results.get("get_financial_statements") or {}
    indicators = results.get("get_financial_indicators") or {}
    snapshot = results.get("get_valuation_snapshot") or {}
    profile = results.get("get_company_profile") or {}
    historical = results.get("get_historical_valuation_percentile") or {}
    industry_comparison = results.get("get_industry_valuation_comparison") or {}
    forecast = results.get("get_earnings_forecast") or {}

    current_price = _num_or_none(market_data.get("latest", {}).get("close"))
    if current_price is None:
        current_price = _num_or_none(snapshot.get("close"))
    total_shares = _num_or_none(snapshot.get("total_shares"))
    if total_shares is None:
        total_shares = _num_or_none(profile.get("total_shares"))

    balance_sheet = statements.get("balance_sheet") or []
    income_statement = statements.get("income_statement") or []
    cash_flow = statements.get("cash_flow") or []

    net_profit_ttm = _ttm(income_statement, "parent_net_profit")
    revenue_ttm = _ttm(income_statement, "operate_income")
    ocf_ttm = _ttm(cash_flow, "netcash_operate")
    capex_ttm = _ttm(cash_flow, "construct_long_asset")
    dividend_ttm = _ttm(cash_flow, "assign_dividend")

    eps_ttm = None
    if net_profit_ttm is not None and total_shares:
        eps_ttm = net_profit_ttm / total_shares
    elif income_statement:
        eps_ttm = _ttm(income_statement, "basic_eps")

    sps_ttm = None
    if revenue_ttm is not None and total_shares:
        sps_ttm = revenue_ttm / total_shares

    parent_equity = None
    if balance_sheet:
        parent_equity = _num_or_none(balance_sheet[0].get("parent_equity"))
    bps = None
    if parent_equity is not None and total_shares:
        bps = parent_equity / total_shares
    latest_indicator = indicators.get("latest") or {}
    if bps is None:
        bps = _num_or_none(latest_indicator.get("bps"))

    fcf = None
    if ocf_ttm is not None and capex_ttm is not None:
        fcf = ocf_ttm - capex_ttm

    dps = None
    if dividend_ttm is not None and total_shares:
        dps = dividend_ttm / total_shares
    annual_dps = None
    if total_shares:
        annual_dividend = _latest_annual(cash_flow, "assign_dividend")
        if annual_dividend is not None:
            annual_dps = annual_dividend / total_shares
    if dps is None:
        dps = annual_dps

    payout_ratio = None
    if dividend_ttm is not None and net_profit_ttm is not None and net_profit_ttm > 0:
        payout_ratio = min(1.0, max(0.0, dividend_ttm / net_profit_ttm))

    roe = _fraction(latest_indicator.get("roe"))
    gross_margin = _fraction(latest_indicator.get("gross_margin"))
    net_profit_yoy = _fraction(latest_indicator.get("net_profit_yoy"))
    revenue_growth_cagr = _revenue_cagr(income_statement)
    revenue_growth_yoy = None
    if income_statement:
        revenue_growth_yoy = _fraction(income_statement[0].get("operate_income_yoy"))
    forecast_growth = _forecast_growth(
        forecast.get("research_reports") or [],
        eps_ttm,
    )
    if forecast_growth is None:
        forecast_growth = _guidance_growth(forecast.get("earnings_guidance") or [])
    forecast_eps, forecast_year = _consensus_eps(forecast.get("research_reports") or [])

    valuation = {
        "pe_ttm": _num_or_none(snapshot.get("pe_ttm")),
        "pe_static": _num_or_none(snapshot.get("pe_static")),
        "pe_dynamic": _num_or_none(snapshot.get("pe_dynamic")),
        "pb": _num_or_none(snapshot.get("pb")),
        "ps": _num_or_none(snapshot.get("ps")),
        "pcf": _num_or_none(snapshot.get("pcf")),
        "dividend_yield": _num_or_none(snapshot.get("dividend_yield")),
    }
    if valuation.get("dividend_yield") is None and dps and current_price:
        valuation["dividend_yield"] = dps / current_price * 100.0

    return {
        "symbol": symbol,
        "company_name": str(
            profile.get("company_name")
            or market_data.get("company_name")
            or symbol
        ),
        "industry_name": str(profile.get("industry") or market_data.get("industry") or ""),
        "current_price": current_price,
        "total_shares": total_shares,
        "eps_ttm": eps_ttm,
        "bps": bps,
        "sps_ttm": sps_ttm,
        "fcf": fcf,
        "ocf_ttm": ocf_ttm,
        "dps": dps,
        "annual_dps": annual_dps,
        "roe": roe,
        "gross_margin": gross_margin,
        "net_profit_yoy": net_profit_yoy,
        "payout_ratio": payout_ratio,
        "revenue_growth_cagr": revenue_growth_cagr,
        "revenue_growth_yoy": revenue_growth_yoy,
        "forecast_growth": forecast_growth,
        "forecast_eps": forecast_eps,
        "forecast_year": forecast_year,
        "valuation": valuation,
        "historical": {
            key: stats for key, stats in (historical.get("metrics") or {}).items()
        },
        "industry_peers": industry_comparison.get("peers") or {},
        "industry_bench": industry_comparison.get("industry") or {},
    }


def _consensus_eps(reports: list[dict[str, Any]]) -> tuple[float | None, int | None]:
    """Median consensus EPS for the current year, falling back to next year."""
    year = date.today().year
    for offset in (0, 1):
        values = [
            float(item[f"eps_{year + offset}"])
            for item in reports
            if item.get(f"eps_{year + offset}") is not None
        ]
        if values:
            return float(statistics.median(values)), year + offset
    return None, None


def _num_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:  # NaN
        return None
    return number


def _data_quality(metrics: dict[str, Any], valuation: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
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
            "growth": metrics.get("revenue_growth_cagr") or metrics.get("forecast_growth"),
            "historical_percentile": bool(metrics.get("historical")),
            "industry_comparison": bool(
                metrics.get("industry_peers") or metrics.get("industry_bench")
            ),
            "forecast": metrics.get("forecast_growth"),
        }.items()
        if not value
    ]
    return {
        "missing": missing,
        "methods_available": valuation.get("available_methods") or [],
        "warnings": warnings,
    }


def _fmt_yi(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value / 1e8:.2f}"


def _fmt_pct(value: float | None, digits: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.{digits}f}%"


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:.2f}"


def _build_report_section(analysis: dict[str, Any]) -> str:
    """Deterministic Markdown section used as fallback (and default)."""
    metrics = analysis.get("metrics") or {}
    valuation = analysis.get("valuation") or {}
    indicators = analysis.get("indicators") or {}
    latest = indicators.get("latest") or {}
    statements = analysis.get("statements") or {}
    periods = statements.get("periods") or []
    snapshot = analysis.get("valuation_snapshot") or {}
    data_quality = analysis.get("data_quality") or {}
    warnings = analysis.get("warnings") or []
    company_name = analysis.get("company_name") or analysis.get("symbol") or ""
    symbol = analysis.get("symbol") or ""

    lines = [
        f"## 基本面与估值（{company_name} {symbol}）",
        f"- 财报窗口：最近4个报告季度+最近年度报告，共{len(periods)}期"
        + (f"（截至{periods[0]}）" if periods else ""),
    ]
    report_date = latest.get("report_date") or (periods[0] if periods else "最新报告期")
    revenue = latest.get("total_operate_revenue")
    net_profit = latest.get("parent_net_profit")
    revenue_yoy = _fraction(latest.get("revenue_yoy")) if latest.get("revenue_yoy") is not None else None
    net_profit_yoy = _fraction(latest.get("net_profit_yoy")) if latest.get("net_profit_yoy") is not None else None
    roe = _fraction(latest.get("roe")) if latest.get("roe") is not None else metrics.get("roe")
    gross_margin = _fraction(latest.get("gross_margin")) if latest.get("gross_margin") is not None else None
    debt_ratio = _fraction(latest.get("debt_ratio")) if latest.get("debt_ratio") is not None else None
    lines.append(
        f"- 盈利概览（{report_date}）：营业总收入 {_fmt_yi(revenue)} 亿元"
        f"（同比{_fmt_pct(revenue_yoy)}），归母净利润 {_fmt_yi(net_profit)} 亿元"
        f"（同比{_fmt_pct(net_profit_yoy)}）；TTM每股收益 {_fmt_price(metrics.get('eps_ttm'))} 元，"
        f"每股净资产 {_fmt_price(metrics.get('bps'))} 元，ROE {_fmt_pct(roe)}，"
        f"毛利率 {_fmt_pct(gross_margin)}，资产负债率 {_fmt_pct(debt_ratio)}。"
    )
    dividend_yield = metrics.get("valuation", {}).get("dividend_yield")
    yield_text = f"，股息率 {dividend_yield:.2f}%" if dividend_yield else ""
    lines.append(
        f"- 现金流与分红：TTM 经营现金流 {_fmt_yi(metrics.get('ocf_ttm'))} 亿元，"
        f"自由现金流 {_fmt_yi(metrics.get('fcf'))} 亿元；"
        f"每股股息 {_fmt_price(metrics.get('dps'))} 元{yield_text}。"
    )
    historical = metrics.get("historical") or {}
    pe_hist = historical.get("pe_ttm") or {}
    pb_hist = historical.get("pb") or {}
    pe_text = (
        f"{_fmt_price(snapshot.get('pe_ttm'))}（近3年分位 {pe_hist.get('percentile', '—')}%）"
        if snapshot.get("pe_ttm") is not None
        else "—"
    )
    pb_text = (
        f"{_fmt_price(snapshot.get('pb'))}（近3年分位 {pb_hist.get('percentile', '—')}%）"
        if snapshot.get("pb") is not None
        else "—"
    )
    peers = metrics.get("industry_peers") or {}
    bench = metrics.get("industry_bench") or {}
    industry_pe = (peers.get("pe") or bench.get("pe")) or {}
    industry_text = _fmt_price(industry_pe.get("median")) if industry_pe.get("median") is not None else "—"
    if (peers.get("pe") or {}).get("median") is not None:
        if peers.get("source") == "manual":
            pe_bench_label = "手动指定类似公司PE中位数"
        elif peers.get("source") == "llm":
            pe_bench_label = "大模型建议类似公司PE中位数"
        else:
            pe_bench_label = "类似公司PE中位数"
    else:
        pe_bench_label = "行业PE中位数"
    lines.append(
        f"- 估值水平：PE-TTM {pe_text}，PB {pb_text}，"
        f"PS {_fmt_price(snapshot.get('ps'))}；{pe_bench_label} {industry_text}。"
    )
    model_targets = metrics.get("model_targets") or {}
    if model_targets.get("available"):
        mt_confidence = model_targets.get("confidence")
        mt_confidence_text = (
            f"{mt_confidence:.0%}" if mt_confidence is not None else "—"
        )
        lines.append(
            "- 本地模型参考倍数："
            f"PE {_fmt_price(model_targets.get('pe'))}、"
            f"PB {_fmt_price(model_targets.get('pb'))}、"
            f"PS {_fmt_price(model_targets.get('ps'))}"
            f"（置信度 {mt_confidence_text}，"
            f"模型 v{model_targets.get('model_version', '—')}"
            f"，训练报告期 {model_targets.get('report_date') or '—'}）。"
        )

    fair_value = valuation.get("fair_value_range")
    verdict = valuation.get("verdict") or {}
    if verdict.get("label") == "重组/注入中":
        lines.append(
            "- 估值说明：疑似重组/资产注入中，当前报表的每股净资产与每股营收相对市值严重偏低，"
            "相对估值不适用，暂不给出估值中枢。"
        )
        if metrics.get("current_price") is not None:
            lines.append(
                f"- 当前股价 {_fmt_price(metrics.get('current_price'))} 元，"
                "判断：重组/注入中（相对估值不适用）。"
            )
    elif fair_value:
        low, mid, high = fair_value["low"], fair_value["mid"], fair_value["high"]
        manual_flag = bool(valuation.get("manual"))
        if manual_flag:
            label = "（人工估值，待并表确认）"
        else:
            method_names = {"relative": "相对估值", "dcf": "DCF", "ddm": "股息折现"}
            used_names = [
                method_names.get(name, name)
                for name in (valuation.get("available_methods") or [])
            ]
            if len(used_names) == 1:
                label = f"（采用{used_names[0]}）"
            elif used_names:
                label = f"（纳入{'、'.join(used_names)}）"
            else:
                label = ""
        lines.append(
            f"- 合理股价估算{label}：区间 {_fmt_price(low)}–{_fmt_price(high)} 元，"
            f"中枢 {_fmt_price(mid)} 元。"
        )
        if manual_flag:
            note = valuation.get("manual_note") or ""
            if note:
                lines.append(f"  - 说明：{note}")
        else:
            per_method = valuation.get("per_method") or {}
            relative = per_method.get("relative") or {}
            dcf = per_method.get("dcf") or {}
            ddm = per_method.get("ddm") or {}
            if relative.get("available"):
                basis = str(relative.get("basis") or "历史/行业倍数")
                peer_names = relative.get("peer_names") or []
                peer_text = ""
                if "类似公司" in basis and peer_names:
                    shown = "、".join(peer_names[:10])
                    if len(peer_names) > 10:
                        shown += "等"
                    peer_text = f"（类似公司：{shown}，共{len(peer_names)}家）"
                lines.append(
                    f"  - 相对估值：{_fmt_price(relative.get('price'))} 元"
                    f"（目标倍数：{basis}{peer_text}）。"
                )
            if dcf.get("available"):
                lines.append(
                    f"  - DCF：{_fmt_price(dcf.get('price'))} 元"
                    f"（折现率{dcf.get('discount_rate', 0.1):.0%}、永续增速{dcf.get('terminal_growth', 0.02):.0%}、"
                    f"假设增速{_fmt_pct(dcf.get('growth'), 1)}）。"
                )
            if ddm.get("available"):
                lines.append(
                    f"  - 股息折现：{_fmt_price(ddm.get('price'))} 元"
                    f"（假设增速{_fmt_pct(ddm.get('growth'), 1)}）。"
                )
            sensitivity = dcf.get("sensitivity") or {}
            if sensitivity:
                lines.append(
                    "  - DCF敏感性："
                    + "；".join(f"折现率{rate}→{_fmt_price(price)}元" for rate, price in sensitivity.items())
                    + "。"
                )
            for item in valuation.get("excluded_methods") or []:
                name = method_names.get(item.get("method"), item.get("method", ""))
                reason = (item.get("reason") or "不适用").rstrip("。")
                lines.append(f"  - 未纳入：{name}（{reason}）。")
            for name in ("relative", "dcf", "ddm"):
                item = per_method.get(name) or {}
                if not item.get("available"):
                    reason = next(iter(item.get("notes") or []), "不适用").rstrip("。")
                    lines.append(
                        f"  - 未纳入：{method_names.get(name, name)}（{reason}）。"
                    )
        lines.append(
            f"- 当前股价 {_fmt_price(metrics.get('current_price'))} 元，"
            f"估值中枢相对现价偏离 {_fmt_pct(verdict.get('margin'), 1)}，"
            f"判断：{verdict.get('label', '—')}。"
        )
    else:
        lines.append("- 合理股价估算：数据不足，无法给出估值区间。")

    if warnings:
        lines.append(
            "- 数据质量提示：" + "；".join(warnings[:5]) + ("。" if len(warnings) <= 5 else "。")
        )
    missing = data_quality.get("missing") or []
    if missing:
        lines.append(f"- 数据缺口：{'、'.join(missing)}。")
    lines.extend(
        [
            "- 风险提示：以上估算基于历史财务数据与公开一致预期，不构成投资建议或收益保证；"
            "市场有风险，投资者应独立判断并自行承担决策风险。",
        ]
    )
    return "\n".join(lines)


def _compact_prompt_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    metrics = analysis.get("metrics") or {}
    valuation = analysis.get("valuation") or {}
    return {
        "symbol": analysis.get("symbol"),
        "company_name": analysis.get("company_name"),
        "industry": analysis.get("industry"),
        "as_of": analysis.get("as_of"),
        "metrics": metrics,
        "valuation": valuation,
        "warnings": analysis.get("warnings"),
    }


async def _llm_report_section(analysis: dict[str, Any]) -> str | None:
    if not llm_configured():
        return None
    payload = _compact_prompt_payload(analysis)
    prompt = (
        "只输出一个 Markdown 章节，标题固定为 `## 基本面与估值`。"
        "只允许使用下面提供的数据，不得编造数字；必须包含估值区间、方法、关键假设、"
        "低估/合理/高估判断与风险提示；若提供本地模型参考倍数，需在关键假设中说明"
        "模型版本与置信度；若估值标记为人工估值，须注明'人工估值区间（待并表确认）'；"
        "末尾必须保留风险提示与免责声明。\n\n"
        f"DATA: {json.dumps(payload, ensure_ascii=False, default=str)}"
    )
    try:
        return await asyncio.wait_for(
            llm_reply(_system_prompt(), prompt, max_tokens=1200),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception:
        return None


def _system_prompt() -> str:
    return (
        "你是一名谨慎的 A 股基本面研究助手。只能使用提供的数据，不得编造事实或保证收益。"
        "估值是估算区间而非精确价格，必须提示不构成投资建议。输出纯中文 Markdown。"
    )


class StockFundamentalHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        warnings: list[str] = []
        results: dict[str, Any] = {}
        core_failures = 0
        for name in TOOL_SEQUENCE:
            try:
                results[name] = await run_tool(name, symbol, market_data, warnings)
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"{name} 失败: {exc}")
                if name in CORE_TOOLS:
                    core_failures += 1
        if core_failures == len(CORE_TOOLS):
            raise RuntimeError("核心基本面工具（报表/指标/估值快照）全部失败")

        metrics = _build_metrics(symbol, market_data, results)
        model_targets: dict[str, Any] = {}
        try:
            model_targets = await run_tool(
                "get_model_targets",
                symbol,
                market_data,
                warnings,
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"get_model_targets 失败: {exc}")
        metrics["model_targets"] = model_targets
        valuation: dict[str, Any] = {}
        try:
            valuation = await run_tool(
                "estimate_fair_value",
                symbol,
                market_data,
                warnings,
                metrics=metrics,
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"estimate_fair_value 失败: {exc}")
            valuation = {"error": str(exc)}

        analysis = {
            "symbol": symbol,
            "company_name": metrics.get("company_name") or symbol,
            "industry": metrics.get("industry_name") or "",
            "as_of": str(results.get("get_valuation_snapshot", {}).get("data_date") or date.today()),
            "profile": results.get("get_company_profile"),
            "statements": results.get("get_financial_statements"),
            "indicators": results.get("get_financial_indicators"),
            "abstract": results.get("get_financial_abstract"),
            "valuation_snapshot": results.get("get_valuation_snapshot"),
            "historical": results.get("get_historical_valuation_percentile"),
            "industry_comparison": results.get("get_industry_valuation_comparison"),
            "forecast": results.get("get_earnings_forecast"),
            "model_targets": model_targets,
            "metrics": metrics,
            "valuation": valuation,
            "warnings": warnings,
            "data_quality": _data_quality(metrics, valuation, warnings),
        }

        section = await _llm_report_section(analysis)
        if not section:
            section = _build_report_section(analysis)
        if "风险提示" not in section:
            section += "\n- 风险提示：以上估算基于历史财务数据与公开一致预期，不构成投资建议。"

        fair_value = valuation.get("fair_value_range") or {}
        summary = {
            "valuation_verdict": (valuation.get("verdict") or {}).get("label", "数据不足"),
            "fair_value_range": fair_value,
            "current_price": metrics.get("current_price"),
            "as_of": analysis["as_of"],
        }
        return json_dumps(
            {"analysis": analysis, "report_section": section, "summary": summary}
        )


def build_agent() -> StockFundamentalHandler:
    return StockFundamentalHandler()
