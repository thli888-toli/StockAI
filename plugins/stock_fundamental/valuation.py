"""Multi-method fair-value estimation built on fundamental metrics.

Three methods are combined with equal weight when available:
  1. Relative valuation  - target multiples from historical percentiles or
     industry medians applied to TTM per-share metrics.
  2. DCF                - two-stage free cash flow model.
  3. Dividend discount  - Gordon growth model.
"""

from __future__ import annotations

import math
import statistics
from typing import Any


DEFAULT_DISCOUNT_RATE = 0.10
DEFAULT_TERMINAL_GROWTH = 0.02
DEFAULT_FORECAST_YEARS = 5
DEFAULT_TARGET_PERCENTILE = 0.50
DEFAULT_FALLBACK_GROWTH = 0.05
SENSITIVITY_RATES = (0.08, 0.12)
VERDICT_BAND = 0.10
MAX_GROWTH = 0.30
MIN_DIVIDEND_YIELD = 0.01
METHOD_WEIGHTS = {"relative": 1.0, "dcf": 0.8, "ddm": 0.5}
OUTLIER_LOW_FACTOR = 0.2
OUTLIER_HIGH_FACTOR = 5.0


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


def _weighted_median(pairs: list[tuple[float, float]]) -> float:
    """Weighted median: the price where cumulative weight crosses half."""
    ordered = sorted(pairs, key=lambda item: item[0])
    total = sum(weight for _, weight in ordered)
    half = total / 2.0
    accumulated = 0.0
    for price, weight in ordered:
        accumulated += weight
        if accumulated >= half:
            return price
    return ordered[-1][0]


def _weighted_growth(
    cagr: float | None,
    forecast: float | None,
    notes: list[str],
) -> float:
    if cagr is not None and forecast is not None:
        value = 0.6 * cagr + 0.4 * forecast
        notes.append(f"增长假设采用近3年营收CAGR({cagr:.1%})与一致预期增速({forecast:.1%})加权。")
    elif cagr is not None:
        value = cagr
        notes.append(f"增长假设采用近3年营收CAGR({cagr:.1%})；缺少一致预期数据。")
    elif forecast is not None:
        value = forecast
        notes.append(f"增长假设采用一致预期增速({forecast:.1%})；缺少历史营收CAGR。")
    else:
        value = DEFAULT_FALLBACK_GROWTH
        notes.append(f"缺少历史与一致预期增速数据，按默认 {DEFAULT_FALLBACK_GROWTH:.1%} 假设。")
    return _clamp(value, 0.0, MAX_GROWTH)


def relative_valuation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Implied price from target multiples (historical p50 or industry median)."""
    valuation = metrics.get("valuation") or {}
    historical = metrics.get("historical") or {}
    industry = metrics.get("industry") or {}
    per_share = {
        "pe_ttm": metrics.get("eps_ttm"),
        "pb": metrics.get("bps"),
        "ps": metrics.get("sps_ttm"),
    }
    multiples: dict[str, dict[str, Any]] = {
        "pe_ttm": {"name": "PE-TTM", "base": per_share["pe_ttm"]},
        "pb": {"name": "PB", "base": per_share["pb"]},
        "ps": {"name": "PS", "base": per_share["ps"]},
    }
    results: list[dict[str, Any]] = []
    notes: list[str] = []
    for key, item in multiples.items():
        base = _num(item["base"])
        hist = historical.get(key) or {}
        industry_item = industry.get(key) or {}
        if base is None or base <= 0:
            continue
        target: float | None = None
        source = ""
        hist_p50 = _num(hist.get("p50"))
        industry_median = _num(industry_item.get("median"))
        if hist_p50 is not None and hist_p50 > 0:
            target = hist_p50
            source = "自身历史50分位"
        elif industry_median is not None and industry_median > 0:
            target = industry_median
            source = "行业中位数"
        if target is None:
            continue
        implied = base * target
        if implied <= 0:
            continue
        results.append(
            {
                "metric": key,
                "name": item["name"],
                "base": _round(base),
                "target_multiple": _round(target),
                "target_source": source,
                "implied_price": _round(implied),
            }
        )
    prices = [float(item["implied_price"]) for item in results]
    if not prices:
        return {
            "available": False,
            "method": "relative",
            "notes": notes + ["缺少足够的倍数或每股指标，相对估值不可用。"],
        }
    notes.append("目标倍数优先取自身近3年历史50分位，缺失时取行业中位数。")
    return {
        "available": True,
        "method": "relative",
        "price": _round(_median(prices)),
        "low": _round(min(prices)),
        "high": _round(max(prices)),
        "detail": results,
        "notes": notes,
    }


def _discounted_cash_flow(
    fcf_per_share: float,
    growth: float,
    discount_rate: float,
    terminal_growth: float,
    years: int,
) -> float:
    if discount_rate <= terminal_growth:
        raise ValueError("discount rate must exceed terminal growth")
    present_value = 0.0
    fcf = fcf_per_share
    for year in range(1, years + 1):
        fcf *= 1.0 + growth
        present_value += fcf / ((1.0 + discount_rate) ** year)
    terminal_value = fcf * (1.0 + terminal_growth) / (discount_rate - terminal_growth)
    present_value += terminal_value / ((1.0 + discount_rate) ** years)
    return present_value


def dcf_valuation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Two-stage DCF with 5 explicit years and a terminal value."""
    fcf = _num(metrics.get("fcf"))
    total_shares = _num(metrics.get("total_shares"))
    if fcf is None or total_shares is None or total_shares <= 0:
        return {
            "available": False,
            "method": "dcf",
            "notes": ["缺少自由现金流或股本数据，DCF 不可用。"],
        }
    if fcf <= 0:
        return {
            "available": False,
            "method": "dcf",
            "notes": ["自由现金流≤0（亏损或高资本开支），DCF 模型不适用。"],
        }
    notes: list[str] = []
    growth = _weighted_growth(
        _num(metrics.get("revenue_growth_cagr")),
        _num(metrics.get("forecast_growth")),
        notes,
    )
    fcf_per_share = fcf / total_shares
    base = _discounted_cash_flow(
        fcf_per_share,
        growth,
        DEFAULT_DISCOUNT_RATE,
        DEFAULT_TERMINAL_GROWTH,
        DEFAULT_FORECAST_YEARS,
    )
    prices = {rate: _discounted_cash_flow(fcf_per_share, growth, rate, DEFAULT_TERMINAL_GROWTH, DEFAULT_FORECAST_YEARS) for rate in SENSITIVITY_RATES}
    notes.append(
        f"折现率 {DEFAULT_DISCOUNT_RATE:.0%}，永续增速 {DEFAULT_TERMINAL_GROWTH:.0%}，"
        f"显性期 {DEFAULT_FORECAST_YEARS} 年；敏感性折现率 {'/'.join(f'{r:.0%}' for r in SENSITIVITY_RATES)}。"
    )
    return {
        "available": True,
        "method": "dcf",
        "price": _round(base),
        "low": _round(min(prices.values())),
        "high": _round(max(prices.values())),
        "fcf_per_share": _round(fcf_per_share),
        "growth": _round(growth, 4),
        "discount_rate": DEFAULT_DISCOUNT_RATE,
        "terminal_growth": DEFAULT_TERMINAL_GROWTH,
        "years": DEFAULT_FORECAST_YEARS,
        "sensitivity": {
            f"{rate:.0%}": _round(price) for rate, price in sorted(prices.items())
        },
        "notes": notes,
    }


def ddm_valuation(metrics: dict[str, Any]) -> dict[str, Any]:
    """Gordon growth dividend discount model."""
    dps = _num(metrics.get("dps"))
    roe = _num(metrics.get("roe"))
    if dps is None or dps <= 0:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["缺少每股股息数据，股息折现不可用。"],
        }
    if roe is None or roe <= 0:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["ROE 为负或缺失，可持续增长率无法计算，股息折现不适用。"],
        }
    payout = _num(metrics.get("payout_ratio"))
    if payout is None:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["缺少分红率数据，股息折现不适用。"],
        }
    current_price = _num(metrics.get("current_price"))
    if current_price is not None and current_price > 0:
        dividend_yield = dps / current_price
        if dividend_yield < MIN_DIVIDEND_YIELD:
            return {
                "available": False,
                "method": "ddm",
                "notes": [
                    f"股息率 {dividend_yield:.2%} 低于 {MIN_DIVIDEND_YIELD:.0%}，"
                    "股息折现不适用。"
                ],
            }
    notes: list[str] = []
    payout = _clamp(payout, 0.0, 1.0)
    growth = _clamp(roe * (1.0 - payout), 0.0, 0.10)
    if DEFAULT_DISCOUNT_RATE <= growth:
        return {
            "available": False,
            "method": "ddm",
            "notes": ["股息增长率不低于折现率，戈登模型不适用。"],
        }
    price = dps * (1.0 + growth) / (DEFAULT_DISCOUNT_RATE - growth)
    sensitivity: dict[float, float] = {}
    for rate in SENSITIVITY_RATES:
        if rate > growth + 0.01:
            sensitivity[rate] = dps * (1.0 + growth) / (rate - growth)
    all_prices = [price] + list(sensitivity.values())
    notes.append(
        f"增长假设 g=ROE×(1-分红率)={growth:.1%}，折现率 {DEFAULT_DISCOUNT_RATE:.0%}。"
    )
    return {
        "available": True,
        "method": "ddm",
        "price": _round(price),
        "low": _round(min(all_prices)),
        "high": _round(max(all_prices)),
        "dps": _round(dps),
        "growth": _round(growth, 4),
        "payout_ratio": _round(payout, 4),
        "discount_rate": DEFAULT_DISCOUNT_RATE,
        "sensitivity": {
            f"{rate:.0%}": _round(value) for rate, value in sorted(sensitivity.items())
        },
        "notes": notes,
    }


DISCLAIMER = "以上为基于历史财务数据与公开预期的估算区间，不构成投资建议或收益保证。"


def estimate_fair_value(metrics: dict[str, Any]) -> dict[str, Any]:
    """Combine available valuation methods into a low/mid/high range."""
    methods = [
        relative_valuation(metrics),
        dcf_valuation(metrics),
        ddm_valuation(metrics),
    ]
    available = [method for method in methods if method.get("available")]
    if not available:
        raise ValueError("没有足够的财务数据计算合理股价估值")

    prices = [float(method["price"]) for method in available]
    median_price = float(statistics.median(prices))
    kept = [
        method
        for method in available
        if OUTLIER_LOW_FACTOR * median_price
        <= float(method["price"])
        <= OUTLIER_HIGH_FACTOR * median_price
    ]
    excluded = [method for method in available if method not in kept]
    if not kept:
        kept = available
        excluded = []

    weighted_pairs = [
        (method, METHOD_WEIGHTS.get(method["method"], 1.0)) for method in kept
    ]
    low = _weighted_median(
        [(float(method.get("low") or method["price"]), weight) for method, weight in weighted_pairs]
    )
    high = _weighted_median(
        [(float(method.get("high") or method["price"]), weight) for method, weight in weighted_pairs]
    )
    mid = _weighted_median(
        [(float(method["price"]), weight) for method, weight in weighted_pairs]
    )
    low = min(low, mid)
    high = max(high, mid)

    current_price = _num(metrics.get("current_price"))
    if current_price is None or current_price <= 0:
        verdict: dict[str, Any] = {
            "label": "数据不足",
            "text": "缺少当前股价，无法给出低估/合理/高估判断。",
        }
    else:
        margin = (current_price - mid) / mid
        if margin < -VERDICT_BAND:
            label = "低估"
        elif margin > VERDICT_BAND:
            label = "高估"
        else:
            label = "合理"
        verdict = {
            "label": label,
            "current_price": _round(current_price),
            "margin": _round(margin, 4),
            "text": (
                f"当前股价 {_round(current_price)} 元相对估值中枢 {_round(mid)} 元"
                f"偏离 {margin:+.1%}，判断为{label}。"
            ),
        }

    per_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        per_method[method["method"]] = dict(method)

    return {
        "fair_value_range": {
            "low": _round(low),
            "mid": _round(mid),
            "high": _round(high),
        },
        "per_method": per_method,
        "verdict": verdict,
        "available_methods": [method["method"] for method in kept],
        "excluded_methods": [
            {
                "method": method["method"],
                "reason": next(iter(method.get("notes") or []), "与其他方法偏差过大"),
            }
            for method in excluded
        ],
        "assumptions": {
            "discount_rate": DEFAULT_DISCOUNT_RATE,
            "terminal_growth": DEFAULT_TERMINAL_GROWTH,
            "forecast_years": DEFAULT_FORECAST_YEARS,
            "target_percentile": DEFAULT_TARGET_PERCENTILE,
            "method_weights": dict(METHOD_WEIGHTS),
            "outlier_band": [OUTLIER_LOW_FACTOR, OUTLIER_HIGH_FACTOR],
        },
        "disclaimer": DISCLAIMER,
    }
