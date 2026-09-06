"""In-plugin tool registry for the us_fundamental agent.

Every tool is an async callable returning a JSON-serializable dict. Network
work runs behind ``run_blocking`` and results are cached in the US cache DB.
``estimate_fair_value`` is a pure computation reusing the A-share valuation
engine with US parameter files.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
from datetime import date
from typing import Any, Awaitable, Callable
from pathlib import Path

import pandas as pd
import yaml

from framework.llm import llm_configured, llm_reply
from plugins.stock_common import json_dumps, json_loads, run_blocking
from plugins.us_cache import US_CACHE
from plugins.us_common import (
    stockanalysis_forecast,
    yf_ticker_history,
    yf_ticker_info,
)
from plugins.us_fundamental.edgar import (
    extract_financials,
    financials_summary,
    sec_company_title,
    sec_fundamentals,
    ttm_as_of,
)
from plugins.us_fundamental.us_config import (
    load_us_valuation_config,
    us_manual_peers,
)
from plugins.stock_fundamental.valuation import estimate_fair_value


CACHE_VERSION = 1
LLM_TIMEOUT_SECONDS = 60.0


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _fraction(value: Any) -> float | None:
    number = _num(value)
    if number is None:
        return None
    if abs(number) > 1.0:
        return number / 100.0
    return number


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none", "<na>"):
        return ""
    return text


def _today_key() -> str:
    return f"{date.today().isoformat()}#v{CACHE_VERSION}"


def _percentile_stats(values: pd.Series) -> dict[str, Any]:
    series = pd.to_numeric(values, errors="coerce").dropna()
    series = series[series > 0]
    if len(series) < 20:
        return {}
    latest = float(series.iloc[-1])
    return {
        "latest": _num(latest),
        "p25": _num(series.quantile(0.25)),
        "p50": _num(series.quantile(0.50)),
        "p75": _num(series.quantile(0.75)),
        "percentile": round(float((series < latest).mean() * 100), 1),
        "min": _num(series.min()),
        "max": _num(series.max()),
        "samples": int(len(series)),
    }


def _fetch_profile(ticker: str) -> dict[str, Any]:
    info = yf_ticker_info(ticker)
    company_name = (
        _text(info.get("longName") or info.get("shortName") or ticker)
    )
    if company_name == ticker:
        company_name = sec_company_title(ticker) or company_name
    sector = _text(info.get("sector"))
    industry = _text(info.get("industry"))
    return {
        "symbol": ticker,
        "company_name": company_name,
        "industry": industry or sector,
        "sector": sector,
        "exchange": _text(info.get("exchange") or info.get("fullExchangeName")),
        "currency": _text(info.get("currency") or "USD"),
        "source": "yfinance",
    }


def _fetch_statements(ticker: str) -> dict[str, Any]:
    return sec_fundamentals(ticker)


def _rolling_ttm(
    records: list[dict[str, Any]],
    as_of: str,
    field: str,
) -> float | None:
    """Trailing-twelve-month value as of a date using quarterly facts."""
    return ttm_as_of(records, field, as_of)


def _latest_equity(
    equity_records: list[dict[str, Any]],
    as_of: str,
) -> float | None:
    matching = [
        item
        for item in equity_records
        if item.get("end") is not None
        and str(item["end"]) <= as_of
        and item.get("equity") is not None
    ]
    if not matching:
        return None
    return float(sorted(matching, key=lambda item: str(item["end"]))[-1]["equity"])


def _fetch_historical_percentile(ticker: str, shares: float | None) -> dict[str, Any]:
    sec = sec_fundamentals(ticker)
    facts = {}
    # Re-read facts to get granular records for rolling calculations.
    from plugins.us_fundamental.edgar import company_facts, ticker_to_cik

    records = extract_financials(company_facts(ticker_to_cik(ticker)))
    all_records = list(records.get("annual") or []) + list(
        records.get("quarterly") or []
    )
    equity_records = records.get("equity") or []
    history = yf_ticker_history(ticker, period="3y", interval="1mo")
    if history.empty or shares is None or shares <= 0:
        raise RuntimeError("历史分位数据不足（缺少月线或股本）")
    pe_values: list[float] = []
    pb_values: list[float] = []
    ps_values: list[float] = []
    for _, row in history.iterrows():
        bar_date = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        close = float(row["close"])
        if close <= 0:
            continue
        ni = _rolling_ttm(all_records, bar_date, "net_income")
        revenue = _rolling_ttm(all_records, bar_date, "revenue")
        equity = _latest_equity(equity_records, bar_date)
        if ni is not None:
            pe_values.append(close / (ni / shares))
        if equity is not None and equity > 0:
            pb_values.append(close / (equity / shares))
        if revenue is not None and revenue > 0:
            ps_values.append(close / (revenue / shares))
    metrics: dict[str, Any] = {}
    for key, series in (
        ("pe_ttm", pd.Series(pe_values)),
        ("pb", pd.Series(pb_values)),
        ("ps", pd.Series(ps_values)),
    ):
        stats = _percentile_stats(series)
        if stats:
            metrics[key] = stats
    if not metrics:
        raise RuntimeError("历史估值分位计算失败（样本不足）")
    return {
        "symbol": ticker,
        "window": "近三年(月频近似)",
        "metrics": metrics,
        "note": "近似算法：月收盘价 ÷ 当季TTM每股值，股本按最新值估算。",
    }


def _load_peer_config(ticker: str) -> list[Any]:
    peers, present = us_manual_peers(ticker)
    if present:
        return peers
    path = Path(__file__).resolve().parents[2] / "config" / "us_peers.yaml"
    if not path.exists():
        return []
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []
    return list(((data.get("peers") or {}).get(ticker.upper()) or []))


def _peer_stats(peers: list[Any], target: str) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    pe_values: list[float] = []
    pb_values: list[float] = []
    ps_values: list[float] = []
    peer_list: list[dict[str, Any]] = []
    for entry in peers:
        if isinstance(entry, dict):
            code = _text(entry.get("code")).upper()
            name = _text(entry.get("name")) or code
        else:
            code = _text(entry).upper()
            name = code
        if not code or code == target.upper():
            continue
        info = yf_ticker_info(code)
        if not info:
            continue
        price = _num(
            info.get("currentPrice")
            or info.get("regularMarketPrice")
            or info.get("previousClose")
        )
        pe = _num(info.get("trailingPE"))
        pb = _num(info.get("priceToBook"))
        ps = _num(info.get("priceToSalesTrailing12Months"))
        if price and (pe is None or pb is None or ps is None):
            try:
                sec = sec_fundamentals(code)
                shares = _num(sec.get("latest_shares"))
                net_income = _num(sec.get("net_income_ttm"))
                revenue = _num(sec.get("revenue_ttm"))
                equity = _num(sec.get("latest_equity"))
                if pe is None and net_income is not None and shares and net_income > 0:
                    eps = net_income / shares
                    if eps > 0:
                        pe = price / eps
                if pb is None and equity is not None and shares and equity > 0:
                    pb = price / (equity / shares)
                if ps is None and revenue is not None and shares and revenue > 0:
                    ps = price / (revenue / shares)
            except Exception:
                pass
        if pe is not None and pe > 0:
            pe_values.append(pe)
        if pb is not None and pb > 0:
            pb_values.append(pb)
        if ps is not None and ps > 0:
            ps_values.append(ps)
        peer_list.append(
            {"code": code, "name": name, "pe_ttm": pe, "pb": pb, "ps": ps}
        )
    if not peer_list:
        raise RuntimeError("可比公司均获取失败")
    stats: dict[str, Any] = {}
    if pe_values:
        stats["pe"] = {
            "median": _num(statistics.median(pe_values)),
            "mean": _num(statistics.mean(pe_values)),
        }
    if pb_values:
        stats["pb"] = {
            "median": _num(statistics.median(pb_values)),
            "mean": _num(statistics.mean(pb_values)),
        }
    if ps_values:
        stats["ps"] = {
            "median": _num(statistics.median(ps_values)),
            "mean": _num(statistics.mean(ps_values)),
        }
    return stats, peer_list, len(peer_list)


def _extract_json_object(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = match.group(1) if match else text
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


async def _validate_peers_llm(
    ticker: str,
    company_name: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    if not llm_configured() or not candidates:
        return {"changed": False, "reason": "", "peers": []}
    prompt = (
        "你是美股可比公司评审专家。判断候选列表是否与目标公司真正可比"
        "（同行业、业务/规模/盈利模式相近）。只返回 JSON："
        '{"changed": true/false, "reason": "一句话", '
        '"peers": [{"code": "TICKER", "name": "名称"}]}。'
        "changed=true 时给出最合适的 5-10 只美股。\n\n"
        f"目标公司：{ticker} {company_name}\n"
        f"候选：{json.dumps(candidates, ensure_ascii=False)}"
    )
    try:
        response = await asyncio.wait_for(
            llm_reply("你只输出严格的 JSON。", prompt, max_tokens=500),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception:
        return {"changed": False, "reason": "", "peers": []}
    parsed = _extract_json_object(response)
    if not parsed:
        return {"changed": False, "reason": "", "peers": []}
    peers: list[dict[str, Any]] = []
    for item in parsed.get("peers") or []:
        code = _text(item.get("code") if isinstance(item, dict) else item).upper()
        name = _text(item.get("name")) if isinstance(item, dict) else code
        if code and code != ticker.upper():
            peers.append({"code": code, "name": name or code})
    return {
        "changed": bool(parsed.get("changed")),
        "reason": _text(parsed.get("reason")),
        "peers": peers,
    }


async def _fetch_peer_comparison(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "symbol": ticker,
        "source": "",
        "basis": "",
        "peer_count": None,
    }
    peers = _load_peer_config(ticker)
    if peers:
        stats, peer_list, peer_count = await run_blocking(
            _peer_stats,
            peers,
            ticker,
            timeout=60.0,
            retries=0,
        )
        stats["peer_list"] = peer_list
        stats["source"] = "us_config"
        result.update(
            {
                "source": "us_config",
                "basis": "peers",
                "peer_count": peer_count,
                "peers": stats,
            }
        )
    us_cfg, _, _ = load_us_valuation_config(ticker)
    if not bool(us_cfg.get("skip_llm_peer_validation", False)):
        candidates = [
            {"code": item.get("code"), "name": item.get("name") or item.get("code")}
            for item in ((result.get("peers") or {}).get("peer_list") or [])
            if isinstance(item, dict) and item.get("code")
        ]
        try:
            llm_result = await _validate_peers_llm(
                ticker,
                str(market_data.get("company_name") or ticker),
                candidates,
            )
        except Exception:
            llm_result = {"changed": False, "reason": "", "peers": []}
        if llm_result.get("changed") and llm_result.get("peers"):
            stats, peer_list, peer_count = await run_blocking(
                _peer_stats,
                llm_result["peers"],
                ticker,
                timeout=60.0,
                retries=0,
            )
            if peer_count:
                stats["peer_list"] = peer_list
                stats["source"] = "us_llm"
                stats["reason"] = llm_result.get("reason", "")
                result.update(
                    {
                        "source": "us_llm",
                        "basis": "peers",
                        "peer_count": peer_count,
                        "peers": stats,
                        "llm_validated": True,
                    }
                )
            else:
                result["llm_validated"] = True
                result["llm_reason"] = llm_result.get("reason", "")
        else:
            result["llm_validated"] = True
            result["llm_reason"] = llm_result.get("reason", "")
    if not result.get("peers"):
        raise RuntimeError("美股可比公司数据获取失败")
    return result


def _fetch_earnings_forecast(ticker: str) -> dict[str, Any]:
    info = yf_ticker_info(ticker)
    year = date.today().year
    forward_eps = _num(info.get("forwardEps"))
    earnings_growth = _fraction(info.get("earningsGrowth"))
    revenue_growth = _fraction(info.get("revenueGrowth"))
    growth = earnings_growth if earnings_growth is not None else revenue_growth
    target_mean_price = _num(info.get("targetMeanPrice"))
    recommendation_key = _text(info.get("recommendationKey"))
    result: dict[str, Any] = {
        "symbol": ticker,
        "research_reports": (
            [{"year": year, "eps_avg": forward_eps, "source": "yfinance"}]
            if forward_eps is not None
            else []
        ),
        "consensus_growth": growth,
        "earnings_growth": earnings_growth,
        "revenue_growth": revenue_growth,
        "target_mean_price": target_mean_price,
        "recommendation_key": recommendation_key,
        "source": "yfinance",
    }
    useful = forward_eps is not None or growth is not None
    if not useful:
        try:
            fallback = stockanalysis_forecast(ticker)
        except Exception:
            result["source"] = "unavailable"
            return result
        result.update(
            {
                "research_reports": fallback.get("research_reports") or [],
                "consensus_growth": fallback.get("consensus_growth"),
                "earnings_growth": fallback.get("earnings_growth"),
                "revenue_growth": fallback.get("revenue_growth"),
                "source": "stockanalysis",
                "note": fallback.get("note"),
            }
        )
    return result


def _fetch_valuation_snapshot(ticker: str) -> dict[str, Any]:
    info = yf_ticker_info(ticker)
    price = _num(
        info.get("currentPrice")
        or info.get("regularMarketPrice")
        or info.get("previousClose")
    )
    market_cap = _num(info.get("marketCap"))
    shares = _num(info.get("sharesOutstanding"))
    if shares is None:
        try:
            shares = _num(sec_fundamentals(ticker).get("latest_shares"))
        except Exception:
            shares = None
    if shares is None and price and market_cap:
        shares = market_cap / price
    if market_cap is None and price and shares:
        market_cap = price * shares
    dividend_yield = _fraction(info.get("dividendYield"))
    pe_ttm = _num(info.get("trailingPE"))
    pb = _num(info.get("priceToBook"))
    ps = _num(info.get("priceToSalesTrailing12Months"))
    try:
        sec = sec_fundamentals(ticker)
        equity = _num(sec.get("latest_equity"))
        revenue = _num(sec.get("revenue_ttm"))
        net_income = _num(sec.get("net_income_ttm"))
        dividends = _num(sec.get("dividends_ttm"))
        if pe_ttm is None and net_income is not None and shares:
            eps = net_income / shares
            if eps > 0:
                pe_ttm = price / eps
        if pb is None and equity is not None and shares and equity > 0:
            pb = price / (equity / shares)
        if ps is None and revenue is not None and shares and revenue > 0:
            ps = price / (revenue / shares)
        if dividend_yield is None and dividends is not None and market_cap:
            dividend_yield = dividends / market_cap
    except Exception:
        pass
    snapshot: dict[str, Any] = {
        "symbol": ticker,
        "data_date": date.today().isoformat(),
        "close": price,
        "total_market_cap": market_cap,
        "total_shares": shares,
        "pe_ttm": pe_ttm,
        "pe_forward": _num(info.get("forwardPE")),
        "pb": pb,
        "ps": ps,
        "dividend_yield": dividend_yield * 100.0 if dividend_yield is not None else None,
        "trailing_eps": _num(info.get("trailingEps")),
        "forward_eps": _num(info.get("forwardEps")),
        "target_mean_price": _num(info.get("targetMeanPrice")),
        "gross_margin": _fraction(info.get("grossMargins")),
        "net_margin": _fraction(info.get("profitMargins")),
        "source": "yfinance",
    }
    if price is None or shares is None:
        raise RuntimeError(f"yfinance 报价缺少价格或股本: {ticker}")
    return snapshot


# ---------------------------------------------------------------------------
# Tool callables (async)
# ---------------------------------------------------------------------------

async def _get_us_profile(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    return await run_blocking(_fetch_profile, ticker, timeout=20.0, retries=1)


async def _get_us_financial_statements(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    return await run_blocking(_fetch_statements, ticker, timeout=40.0, retries=1)


async def _get_us_valuation_snapshot(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    return await run_blocking(_fetch_valuation_snapshot, ticker, timeout=30.0, retries=1)


async def _get_us_historical_percentile(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    info = yf_ticker_info(ticker)
    shares = _num(info.get("sharesOutstanding"))
    if shares is None:
        try:
            shares = await run_blocking(
                lambda: sec_fundamentals(ticker).get("latest_shares"),
                timeout=30.0,
                retries=0,
            )
        except Exception:
            shares = None
    return await run_blocking(
        _fetch_historical_percentile,
        ticker,
        shares,
        timeout=90.0,
        retries=0,
    )


async def _get_us_peer_comparison(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    return await _fetch_peer_comparison(ticker, market_data)


async def _get_us_earnings_forecast(
    ticker: str,
    market_data: dict[str, Any],
) -> dict[str, Any]:
    return await run_blocking(_fetch_earnings_forecast, ticker, timeout=25.0, retries=1)


def _estimate_fair_value_tool(
    metrics: dict[str, Any],
    ticker: str = "",
) -> dict[str, Any]:
    cfg, source, overrides = load_us_valuation_config(ticker)
    result = estimate_fair_value(
        metrics,
        cfg=cfg,
        config_source=source,
        config_overrides=overrides,
    )
    verdict = result.get("verdict") or {}
    if isinstance(verdict, dict) and verdict.get("text"):
        verdict["text"] = str(verdict["text"]).replace("元", "美元")
        result["verdict"] = verdict
    return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ToolFunc = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]

TOOLS: dict[str, dict[str, Any]] = {
    "get_us_profile": {"cache_seconds": 6 * 3600, "func": _get_us_profile},
    "get_us_financial_statements": {
        "cache_seconds": 12 * 3600,
        "func": _get_us_financial_statements,
    },
    "get_us_valuation_snapshot": {
        "cache_seconds": 30 * 60,
        "func": _get_us_valuation_snapshot,
    },
    "get_us_historical_percentile": {
        "cache_seconds": 6 * 3600,
        "func": _get_us_historical_percentile,
    },
    "get_us_peer_comparison": {
        "cache_seconds": 3600,
        "func": _get_us_peer_comparison,
    },
    "get_us_earnings_forecast": {
        "cache_seconds": 6 * 3600,
        "func": _get_us_earnings_forecast,
    },
    "estimate_fair_value": {"cache_seconds": 0, "func": _estimate_fair_value_tool},
}


async def run_us_tool(
    name: str,
    ticker: str,
    market_data: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = TOOLS.get(name)
    if spec is None:
        raise ValueError(f"未知美股工具: {name}")
    func = spec["func"]
    cache_seconds = int(spec.get("cache_seconds", 0))
    key = f"{ticker}:{name}:{_today_key()}"
    if cache_seconds > 0:
        cached = US_CACHE.get("us_tool", key, cache_seconds)
        cached_payload = cached.get("payload") if isinstance(cached, dict) else None
        stale_empty_forecast = (
            name == "get_us_earnings_forecast"
            and isinstance(cached_payload, dict)
            and not cached_payload.get("research_reports")
            and cached_payload.get("consensus_growth") is None
            and cached_payload.get("target_mean_price") is None
        )
        if cached_payload is not None and not stale_empty_forecast:
            return cached["payload"]
    if name == "estimate_fair_value":
        result = func(metrics or {}, ticker)
    else:
        result = await func(ticker, market_data)
    if not isinstance(result, dict):
        raise RuntimeError(f"工具 {name} 返回了非字典结果")
    empty_forecast = (
        name == "get_us_earnings_forecast"
        and not result.get("research_reports")
        and result.get("consensus_growth") is None
        and result.get("target_mean_price") is None
    )
    if cache_seconds > 0 and not empty_forecast:
        US_CACHE.put("us_tool", key, {"payload": result})
    return result


__all__ = ["TOOLS", "run_us_tool", "estimate_fair_value"]
