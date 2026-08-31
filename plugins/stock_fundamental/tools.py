"""In-plugin tool registry for the stock_fundamental agent.

Every tool is an async callable returning a JSON-serializable dict. Network
tools wrap AkShare behind ``run_blocking``/``disable_http_proxy`` and are
cached in SQLite keyed by (symbol, tool, period_key) with per-tool TTLs.
``estimate_fair_value`` is a pure computation tool with no network access.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
import statistics
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Awaitable, Callable

import pandas as pd
import yaml

from framework.llm import llm_configured, llm_reply
from plugins.stock_common import (
    disable_http_proxy,
    json_dumps,
    json_loads,
    run_blocking,
)
from plugins.stock_fundamental.config import (
    load_valuation_config,
    symbol_manual_peers,
    symbol_manual_peers_present,
)
from plugins.stock_fundamental.fundamental_cache import FundamentalCacheStore
from plugins.stock_fundamental.valuation import estimate_fair_value
from plugins.stock_fundamental.valuation_model import model_targets_from_metrics


FUNDAMENTAL_CACHE = FundamentalCacheStore()
PEER_MIN_COUNT = 5
CACHE_VERSION = 8
LLM_VALIDATION_TIMEOUT = 60.0


def _load_manual_peers(path: str | Path | None = None) -> dict[str, list[Any]]:
    config_path = Path(path) if path else Path(__file__).resolve().parents[2] / "config" / "fundamental_peers.yaml"
    if not config_path.exists():
        return {}
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    peers = data.get("peers") or {}
    return {str(symbol): list(entries or []) for symbol, entries in peers.items()}


MANUAL_PEERS = _load_manual_peers()


def _manual_peers_for_symbol(symbol: str) -> list[Any]:
    """Manual peers for a symbol: JSON config wins, YAML is the fallback."""
    if symbol_manual_peers_present(symbol):
        return symbol_manual_peers(symbol)
    return list(MANUAL_PEERS.get(symbol) or [])


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in ("nan", "none", "nat", "<na>"):
        return ""
    return text


def _pick_column(frame: pd.DataFrame, candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def _em_symbol(symbol: str) -> str:
    if symbol.startswith(("6", "9")):
        return f"SH{symbol}"
    if symbol.startswith(("4", "8")):
        return f"BJ{symbol}"
    return f"SZ{symbol}"


def _dot_symbol(symbol: str) -> str:
    if symbol.startswith(("6", "9")):
        return f"{symbol}.SH"
    if symbol.startswith(("4", "8")):
        return f"{symbol}.BJ"
    return f"{symbol}.SZ"


def _report_date(value: Any) -> str | None:
    try:
        return pd.to_datetime(value, errors="coerce").strftime("%Y-%m-%d")
    except Exception:
        return None


def _records_from_frame(
    frame: pd.DataFrame,
    column_map: dict[str, tuple[str, ...]],
    limit: int = 8,
    sort_by: str | None = None,
    ascending: bool = False,
) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    working = frame.copy()
    if sort_by and sort_by in working.columns:
        working = working.sort_values(sort_by, ascending=ascending)
    records: list[dict[str, Any]] = []
    for _, row in working.head(limit).iterrows():
        record: dict[str, Any] = {}
        for key, candidates in column_map.items():
            column = _pick_column(working, tuple(candidates))
            value = row.get(column) if column else None
            if key in ("report_date", "data_date"):
                record[key] = _report_date(value)
            else:
                record[key] = _num(value)
        if record.get("report_date") or "report_date" not in column_map:
            records.append(record)
    return records


def _latest_trading_date() -> date:
    today = date.today()
    if today.weekday() >= 5:
        return today - timedelta(days=today.weekday() - 4)
    return today


def _latest_quarter_end() -> date:
    today = date.today()
    candidates = [
        date(today.year, 3, 31),
        date(today.year, 6, 30),
        date(today.year, 9, 30),
        date(today.year, 12, 31),
    ]
    past = [item for item in candidates if item <= today]
    if past:
        return max(past)
    return date(today.year - 1, 12, 31)


def _call_ak(function: Callable[..., pd.DataFrame]):
    import akshare as ak

    with disable_http_proxy():
        return function(ak)


# ---------------------------------------------------------------------------
# get_company_profile
# ---------------------------------------------------------------------------

PROFILE_CNINFO_COLUMNS = {
    "company_name": ("公司名称",),
    "short_name": ("A股简称",),
    "industry": ("所属行业",),
    "listing_date": ("上市日期",),
    "registered_capital": ("注册资金",),
    "main_business": ("主营业务",),
    "business_scope": ("经营范围",),
}


async def _get_company_profile(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "symbol": symbol,
        "company_name": str(market_data.get("company_name") or symbol),
        "industry": str(market_data.get("industry") or ""),
        "listing_date": None,
        "registered_capital": None,
        "main_business": "",
        "business_scope": "",
        "source": "market_data",
    }
    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_profile_cninfo(symbol=symbol)),
            timeout=15.0,
            retries=1,
        )
        if frame is not None and not frame.empty:
            row = frame.iloc[0]
            for key, candidates in PROFILE_CNINFO_COLUMNS.items():
                column = _pick_column(frame, tuple(candidates))
                if column is None:
                    continue
                value = row.get(column)
                if key in ("company_name", "short_name", "industry", "main_business", "business_scope"):
                    text = _text(value)
                    if key == "company_name" and text and text != symbol:
                        profile[key] = text
                    elif key == "short_name" and text:
                        profile["short_name"] = text
                    elif key == "industry" and text:
                        profile[key] = text
                    elif key in ("main_business", "business_scope") and text:
                        profile[key] = text[:500]
                else:
                    profile[key] = _text(value) or None
            profile["source"] = "cninfo"
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_company_profile cninfo 失败: {exc}")

    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_individual_info_em(symbol=symbol)),
            timeout=15.0,
            retries=1,
        )
        if frame is not None and not frame.empty and {"item", "value"}.issubset(frame.columns):
            mapping = dict(zip(frame["item"].astype(str), frame["value"].astype(str)))
            if not profile.get("company_name") or profile.get("company_name") == symbol:
                profile["company_name"] = mapping.get("股票简称") or profile["company_name"] or symbol
            if not profile.get("industry"):
                profile["industry"] = mapping.get("行业") or ""
            profile["total_market_cap"] = _num(mapping.get("总市值"))
            profile["float_market_cap"] = _num(mapping.get("流通市值"))
            profile["total_shares"] = _num(mapping.get("总股本"))
            profile["float_shares"] = _num(mapping.get("流通股本"))
            profile["listing_date"] = profile.get("listing_date") or mapping.get("上市时间") or None
            profile["source"] = profile["source"] if profile["source"] != "market_data" else "eastmoney"
    except Exception as exc:  # noqa: BLE001
        # Eastmoney profile is an enrichment on top of CNINFO; only warn when
        # we still lack basic name/industry so reports don't drown in noise.
        if (
            not profile.get("company_name")
            or profile.get("company_name") == symbol
            or not profile.get("industry")
        ):
            warnings.append(f"get_company_profile eastmoney 失败: {exc}")

    if not profile.get("company_name") or profile.get("company_name") == symbol:
        profile["company_name"] = market_data.get("company_name") or symbol
    if not profile.get("industry"):
        profile["industry"] = market_data.get("industry") or ""
    return profile


# ---------------------------------------------------------------------------
# get_financial_statements
# ---------------------------------------------------------------------------

BALANCE_SHEET_COLUMNS = {
    "report_date": ("REPORT_DATE",),
    "report_type_name": ("REPORT_DATE_NAME",),
    "total_assets": ("TOTAL_ASSETS",),
    "total_liabilities": ("TOTAL_LIABILITIES",),
    "total_equity": ("TOTAL_EQUITY",),
    "parent_equity": ("TOTAL_PARENT_EQUITY",),
    "monetary_funds": ("MONETARYFUNDS",),
    "inventory": ("INVENTORY",),
    "accounts_receivable": ("ACCOUNTS_RECE",),
    "current_assets": ("TOTAL_CURRENT_ASSETS",),
    "current_liabilities": ("TOTAL_CURRENT_LIAB",),
    "goodwill": ("GOODWILL",),
    "intangible_assets": ("INTANGIBLE_ASSET",),
    "share_capital": ("SHARE_CAPITAL",),
    "capital_reserve": ("CAPITAL_RESERVE",),
    "retained_earnings": ("UNASSIGN_RPOFIT",),
    "surplus_reserve": ("SURPLUS_RESERVE",),
    "contract_liabilities": ("CONTRACT_LIAB",),
    "short_loans": ("SHORT_LOAN",),
    "long_loans": ("LONG_LOAN",),
    "noncurrent_assets": ("TOTAL_NONCURRENT_ASSETS",),
    "noncurrent_liabilities": ("TOTAL_NONCURRENT_LIAB",),
    "total_liab_equity": ("TOTAL_LIAB_EQUITY",),
}

INCOME_STATEMENT_COLUMNS = {
    "report_date": ("REPORT_DATE",),
    "report_type_name": ("REPORT_DATE_NAME",),
    "total_operate_income": ("TOTAL_OPERATE_INCOME",),
    "operate_income": ("OPERATE_INCOME",),
    "operate_cost": ("OPERATE_COST",),
    "operate_profit": ("OPERATE_PROFIT",),
    "total_profit": ("TOTAL_PROFIT",),
    "net_profit": ("NETPROFIT",),
    "parent_net_profit": ("PARENT_NETPROFIT",),
    "deducted_net_profit": ("DEDUCT_PARENT_NETPROFIT",),
    "basic_eps": ("BASIC_EPS",),
    "diluted_eps": ("DILUTED_EPS",),
    "income_tax": ("INCOME_TAX",),
    "total_operate_income_yoy": ("TOTAL_OPERATE_INCOME_YOY",),
    "operate_income_yoy": ("OPERATE_INCOME_YOY",),
    "parent_net_profit_yoy": ("PARENT_NETPROFIT_YOY",),
}

CASH_FLOW_COLUMNS = {
    "report_date": ("REPORT_DATE",),
    "report_type_name": ("REPORT_DATE_NAME",),
    "netcash_operate": ("NETCASH_OPERATE",),
    "construct_long_asset": ("CONSTRUCT_LONG_ASSET",),
    "netcash_invest": ("NETCASH_INVEST",),
    "netcash_finance": ("NETCASH_FINANCE",),
    "assign_dividend": ("ASSIGN_DIVIDEND_PORFIT",),
    "total_operate_inflow": ("TOTAL_OPERATE_INFLOW",),
    "total_operate_outflow": ("TOTAL_OPERATE_OUTFLOW",),
    "end_cash": ("END_CASH",),
    "cce_add": ("CCE_ADD",),
}


async def _get_financial_statements(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    em_symbol = _em_symbol(symbol)
    calls = (
        (
            "balance_sheet",
            lambda ak: ak.stock_balance_sheet_by_report_em(symbol=em_symbol),
            BALANCE_SHEET_COLUMNS,
        ),
        (
            "income_statement",
            lambda ak: ak.stock_profit_sheet_by_report_em(symbol=em_symbol),
            INCOME_STATEMENT_COLUMNS,
        ),
        (
            "cash_flow",
            lambda ak: ak.stock_cash_flow_sheet_by_report_em(symbol=em_symbol),
            CASH_FLOW_COLUMNS,
        ),
    )
    result: dict[str, Any] = {"symbol": symbol}
    for name, call, column_map in calls:
        try:
            frame = await run_blocking(
                lambda call=call: _call_ak(call),
                timeout=60.0,
                retries=1,
            )
            records = _records_from_frame(
                frame,
                column_map,
                limit=8,
                sort_by="REPORT_DATE",
                ascending=False,
            )
            result[name] = records
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"get_financial_statements {name} 失败: {exc}")
            result[name] = []
    if not any(result.get(name) for name in ("balance_sheet", "income_statement", "cash_flow")):
        raise RuntimeError("三大财务报表均获取失败")
    result["periods"] = _period_list(result)
    return result


def _period_list(statements: dict[str, Any]) -> list[str]:
    periods: set[str] = set()
    for key in ("balance_sheet", "income_statement", "cash_flow"):
        for record in statements.get(key) or []:
            if record.get("report_date"):
                periods.add(record["report_date"])
    return sorted(periods, reverse=True)


# ---------------------------------------------------------------------------
# get_financial_indicators
# ---------------------------------------------------------------------------

EM_INDICATOR_COLUMNS = {
    "report_date": ("REPORT_DATE",),
    "report_type_name": ("REPORT_DATE_NAME",),
    "eps": ("EPSJB",),
    "eps_deducted": ("EPSKCJB",),
    "eps_diluted": ("EPSXS",),
    "bps": ("BPS",),
    "ocf_per_share": ("MGJYXJJE",),
    "per_share_revenue": ("PER_TOI",),
    "per_share_ebit": ("PER_EBIT",),
    "total_operate_revenue": ("TOTALOPERATEREVE",),
    "gross_margin": ("XSMLL",),
    "parent_net_profit": ("PARENTNETPROFIT",),
    "deducted_net_profit": ("KCFJCXSYJLR",),
    "revenue_yoy": ("TOTALOPERATEREVETZ",),
    "net_profit_yoy": ("PARENTNETPROFITTZ",),
    "deducted_net_profit_yoy": ("KCFJCXSYJLRTZ",),
    "sq_revenue_yoy": ("DJD_TOI_YOY",),
    "sq_net_profit_yoy": ("DJD_DPNP_YOY",),
    "sq_net_profit_qoq": ("DJD_DPNP_QOQ",),
    "roe": ("ROEJQ",),
    "roe_deducted": ("ROEKCJQ",),
    "roa": ("ZZCJLL",),
    "net_margin": ("XSJLL",),
    "tax_rate": ("TAXRATE",),
    "debt_ratio": ("ZCFZL",),
    "current_ratio": ("LD",),
    "quick_ratio": ("SD",),
    "cash_ratio": ("CASH_RATIO",),
    "roic": ("ROIC",),
    "fcff_forward": ("FCFF_FORWARD",),
    "fcff_back": ("FCFF_BACK",),
    "operating_cycle": ("OPERATE_CYCLE",),
    "inventory_days": ("CHZZTS",),
    "receivable_days": ("YSZKZZTS",),
    "staff_num": ("STAFF_NUM",),
}

SINA_INDICATOR_COLUMNS = {
    "report_date": ("日期",),
    "eps_diluted": ("摊薄每股收益(元)",),
    "eps_weighted": ("加权每股收益(元)",),
    "bps": ("每股净资产_调整后(元)",),
    "ocf_per_share": ("每股经营性现金流(元)",),
    "roe": ("净资产收益率(%)",),
    "roe_weighted": ("加权净资产收益率(%)",),
    "gross_margin": ("销售毛利率(%)",),
    "net_margin": ("销售净利率(%)",),
    "debt_ratio": ("资产负债率(%)",),
    "revenue_yoy": ("主营业务收入增长率(%)",),
    "net_profit_yoy": ("净利润增长率(%)",),
    "current_ratio": ("流动比率",),
    "quick_ratio": ("速动比率",),
    "dividend_payout": ("股息发放率(%)",),
    "total_assets": ("总资产(元)",),
}

ABSTRACT_METRICS = {
    "net_profit": "净利润",
    "net_profit_yoy": "净利润同比增长率",
    "deducted_net_profit": "扣非净利润",
    "deducted_net_profit_yoy": "扣非净利润同比增长率",
    "total_operate_revenue": "营业总收入",
    "revenue_yoy": "营业总收入同比增长率",
    "basic_eps": "基本每股收益",
    "bps": "每股净资产",
    "ocf_per_share": "每股经营现金流",
    "net_margin": "销售净利率",
    "gross_margin": "销售毛利率",
    "roe": "净资产收益率",
    "debt_ratio": "资产负债率",
    "current_ratio": "流动比率",
    "quick_ratio": "速动比率",
    "inventory_days": "存货周转天数",
    "receivable_days": "应收账款周转天数",
}


async def _get_financial_indicators(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    em_records: list[dict[str, Any]] = []
    sina_records: list[dict[str, Any]] = []
    abstract_records: list[dict[str, Any]] = []
    try:
        frame = await run_blocking(
            lambda: _call_ak(
                lambda ak: ak.stock_financial_analysis_indicator_em(
                    symbol=_dot_symbol(symbol),
                    indicator="按报告期",
                )
            ),
            timeout=30.0,
            retries=1,
        )
        em_records = _records_from_frame(
            frame,
            EM_INDICATOR_COLUMNS,
            limit=8,
            sort_by="REPORT_DATE",
            ascending=False,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_financial_indicators eastmoney 失败: {exc}")
    try:
        frame = await run_blocking(
            lambda: _call_ak(
                lambda ak: ak.stock_financial_analysis_indicator(
                    symbol=symbol,
                    start_year=str(date.today().year - 2),
                )
            ),
            timeout=30.0,
            retries=1,
        )
        sina_records = _records_from_frame(
            frame,
            SINA_INDICATOR_COLUMNS,
            limit=8,
            sort_by="日期",
            ascending=False,
        )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_financial_indicators sina 失败: {exc}")
    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_financial_abstract(symbol=symbol)),
            timeout=30.0,
            retries=1,
        )
        abstract_records = _abstract_records(frame, limit=8)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_financial_indicators abstract 失败: {exc}")

    if not em_records and not sina_records and not abstract_records:
        raise RuntimeError("财务指标数据获取失败")
    return {
        "symbol": symbol,
        "em": em_records,
        "sina": sina_records,
        "abstract": abstract_records,
        "latest": _latest_indicator(em_records, sina_records, abstract_records),
    }


def _abstract_records(frame: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    if "指标" not in frame.columns:
        return []
    metric_rows: dict[str, pd.Series] = {}
    for metric_name in set(ABSTRACT_METRICS.values()):
        row = frame[frame["指标"].astype(str) == metric_name]
        if not row.empty:
            metric_rows[metric_name] = row.iloc[0]
    date_columns = [
        column
        for column in frame.columns
        if column not in ("选项", "指标") and str(column).isdigit() and len(str(column)) == 8
    ]
    date_columns = sorted(date_columns, reverse=True)[:limit]
    reverse_map = {value: key for key, value in ABSTRACT_METRICS.items()}
    records: list[dict[str, Any]] = []
    for column in date_columns:
        record: dict[str, Any] = {"report_date": f"{column[:4]}-{column[4:6]}-{column[6:8]}"}
        for metric_name, series in metric_rows.items():
            key = reverse_map.get(metric_name)
            if key:
                record[key] = _num(series.get(column))
        records.append(record)
    return records


def _latest_indicator(
    em_records: list[dict[str, Any]],
    sina_records: list[dict[str, Any]],
    abstract_records: list[dict[str, Any]],
) -> dict[str, Any]:
    latest: dict[str, Any] = {"report_date": None}
    for record in em_records:
        if record.get("report_date"):
            latest.update({key: record[key] for key in record if key != "report_date"})
            latest["report_date"] = record["report_date"]
            latest["source"] = "eastmoney"
            break
    if not latest.get("report_date"):
        for record in sina_records:
            if record.get("report_date"):
                latest.update({key: record[key] for key in record if key != "report_date"})
                latest["report_date"] = record["report_date"]
                latest["source"] = "sina"
                break
    if not latest.get("report_date"):
        for record in abstract_records:
            if record.get("report_date"):
                latest.update({key: record[key] for key in record if key != "report_date"})
                latest["report_date"] = record["report_date"]
                latest["source"] = "abstract"
                break
    return latest


# ---------------------------------------------------------------------------
# get_financial_abstract
# ---------------------------------------------------------------------------


async def _get_financial_abstract(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": symbol, "periods": [], "business_composition": []}
    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_financial_abstract(symbol=symbol)),
            timeout=30.0,
            retries=1,
        )
        result["periods"] = _abstract_records(frame, limit=8)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_financial_abstract 摘要失败: {exc}")
    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_zygc_em(symbol=_em_symbol(symbol))),
            timeout=30.0,
            retries=1,
        )
        result["business_composition"] = _business_composition(frame, limit=10)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_financial_abstract 主营构成失败: {exc}")
    if not result["periods"] and not result["business_composition"]:
        raise RuntimeError("财务摘要获取失败")
    return result


def _business_composition(frame: pd.DataFrame, limit: int = 10) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    date_column = _pick_column(frame, ("报告日期",))
    item_column = _pick_column(frame, ("主营构成",))
    revenue_column = _pick_column(frame, ("主营收入",))
    if not date_column or not item_column:
        return []
    working = frame.copy()
    latest_date = working[date_column].dropna().max()
    working = working[working[date_column] == latest_date]
    if revenue_column:
        working = working.sort_values(revenue_column, ascending=False)
    records: list[dict[str, Any]] = []
    for _, row in working.head(limit).iterrows():
        records.append(
            {
                "report_date": _report_date(row.get(date_column)),
                "category": _text(row.get(_pick_column(frame, ("分类类型",)))),
                "item": _text(row.get(item_column)),
                "revenue": _num(row.get(revenue_column)),
                "revenue_ratio": _num(row.get(_pick_column(frame, ("收入比例",)))),
                "gross_margin": _num(row.get(_pick_column(frame, ("毛利率",)))),
            }
        )
    return records


# ---------------------------------------------------------------------------
# get_valuation_snapshot
# ---------------------------------------------------------------------------

VALUE_EM_COLUMNS = {
    "data_date": ("数据日期",),
    "close": ("当日收盘价",),
    "pct_change": ("当日涨跌幅",),
    "total_market_cap": ("总市值",),
    "float_market_cap": ("流通市值",),
    "total_shares": ("总股本",),
    "float_shares": ("流通股本",),
    "pe_ttm": ("PE(TTM)",),
    "pe_static": ("PE(静)",),
    "pb": ("市净率",),
    "peg": ("PEG值",),
    "pcf": ("市现率",),
    "ps": ("市销率",),
}


async def _get_valuation_snapshot(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_value_em(symbol=symbol)),
            timeout=30.0,
            retries=1,
        )
        records = _records_from_frame(
            frame,
            VALUE_EM_COLUMNS,
            limit=1,
            sort_by="数据日期",
            ascending=False,
        )
        if not records:
            raise RuntimeError("stock_value_em 返回空数据")
        snapshot = records[0]
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"估值快照获取失败: {exc}") from exc

    snapshot["symbol"] = symbol
    snapshot.setdefault("close", _num(market_data.get("latest", {}).get("close")))

    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_zh_a_spot_em()),
            timeout=40.0,
            retries=0,
        )
        if frame is not None and not frame.empty and "代码" in frame.columns:
            row = frame[frame["代码"].astype(str) == symbol]
            if not row.empty:
                item = row.iloc[0]
                snapshot["pe_dynamic"] = _num(item.get("市盈率-动态"))
                snapshot["turnover"] = _num(item.get("换手率"))
                snapshot["spot_pct_change"] = _num(item.get("涨跌幅"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_valuation_snapshot spot 失败: {exc}")

    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_a_ttm_lyr()),
            timeout=40.0,
            retries=0,
        )
        if frame is not None and not frame.empty and "代码" in frame.columns:
            row = frame[frame["代码"].astype(str) == symbol]
            if not row.empty:
                item = row.iloc[0]
                snapshot["dividend_yield"] = _num(item.get("股息率"))
                snapshot["eps"] = _num(item.get("每股收益"))
                snapshot["bps_alt"] = _num(item.get("每股净资产"))
                snapshot["ps_alt"] = _num(item.get("市销率"))
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_valuation_snapshot ttm_lyr 失败: {exc}")
    return snapshot


# ---------------------------------------------------------------------------
# get_historical_valuation_percentile
# ---------------------------------------------------------------------------

BAIDU_INDICATORS = (
    ("pe_ttm", "市盈率(TTM)"),
    ("pe_static", "市盈率(静)"),
    ("pb", "市净率"),
    ("pcf", "市现率"),
)


def _percentile_stats(series: pd.Series) -> dict[str, Any]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 20:
        return {}
    latest = float(values.iloc[-1])
    return {
        "latest": _num(latest),
        "p25": _num(values.quantile(0.25)),
        "p50": _num(values.quantile(0.50)),
        "p75": _num(values.quantile(0.75)),
        "percentile": round(float((values < latest).mean() * 100), 1),
        "min": _num(values.min()),
        "max": _num(values.max()),
        "samples": int(len(values)),
    }


async def _get_historical_valuation_percentile(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": symbol, "window": "近两年", "metrics": {}}
    for key, indicator in BAIDU_INDICATORS:
        try:
            frame = await run_blocking(
                lambda indicator=indicator: _call_ak(
                    lambda ak: ak.stock_zh_valuation_baidu(
                        symbol=symbol,
                        indicator=indicator,
                        period="近三年",
                    )
                ),
                timeout=20.0,
                retries=1,
            )
            if frame is None or frame.empty or "value" not in frame.columns:
                continue
            working = frame.copy()
            if "date" in working.columns:
                cutoff = pd.Timestamp.now() - pd.DateOffset(years=2)
                dates = pd.to_datetime(working["date"], errors="coerce")
                working = working[dates >= cutoff]
            stats = _percentile_stats(working["value"])
            if stats:
                result["metrics"][key] = stats
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"get_historical_valuation_percentile {indicator} 失败: {exc}")
    if not result["metrics"]:
        raise RuntimeError("历史估值分位数据获取失败")
    return result


# ---------------------------------------------------------------------------
# get_industry_valuation_comparison
# ---------------------------------------------------------------------------


def _match_industry_row(frame: pd.DataFrame, candidates: list[str]) -> dict[str, Any] | None:
    if frame is None or frame.empty or "行业名称" not in frame.columns:
        return None
    working = frame.copy()
    best: dict[str, Any] | None = None
    best_score = -1
    for _, row in working.iterrows():
        industry_name = str(row.get("行业名称") or "").strip()
        if not industry_name:
            continue
        for candidate in candidates:
            candidate = (candidate or "").strip()
            if not candidate:
                continue
            exact = candidate == industry_name
            if not exact and not (
                candidate in industry_name or industry_name in candidate
            ):
                continue
            # Exact matches always beat substring matches; longer names win
            # ties so the specific sub-industry row is preferred over coarse
            # level-1 rows like "制造业".
            score = 1000 + len(industry_name) if exact else len(candidate)
            if score > best_score:
                best_score = score
                best = {
                    "industry_name": industry_name,
                    "company_count": _num(row.get("公司数量")),
                    "pe_median": _num(row.get("静态市盈率-中位数")),
                    "pe_mean": _num(row.get("静态市盈率-算术平均")),
                    "pe_weighted": _num(row.get("静态市盈率-加权平均")),
                }
    return best


async def _get_industry_valuation_comparison(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "symbol": symbol,
        "source": "",
        "basis": "",
        "peer_count": None,
        "matched_industry": "",
    }
    manual_peers = _manual_peers_for_symbol(symbol)
    if manual_peers:
        try:
            stats, peer_list, peer_count = await _fetch_peer_stats(
                manual_peers, symbol, warnings
            )
            if peer_count > 0:
                stats["peer_list"] = peer_list
                stats["source"] = "manual"
                result.update(
                    {
                        "source": "manual",
                        "basis": "manual",
                        "peer_count": peer_count,
                        "peers": stats,
                        "matched_industry": "",
                    }
                )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"get_industry_valuation_comparison manual peers 失败: {exc}")
    try:
        frame = await run_blocking(
            lambda: _call_ak(
                lambda ak: ak.stock_zh_valuation_comparison_em(symbol=_em_symbol(symbol))
            ),
            timeout=30.0,
            retries=0,
        )
        if frame is not None and not frame.empty and "简称" in frame.columns:
            comparison = _comparison_stats(frame, symbol)
            peer_count = len(comparison.get("peer_list") or [])
            if comparison and peer_count >= PEER_MIN_COUNT:
                result["peers"] = comparison
                result["peer_count"] = peer_count
                result["basis"] = "peers"
                result["source"] = "eastmoney"
                result["matched_industry"] = str(
                    frame.get("行业", pd.Series([""])).iloc[0] or ""
                )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_industry_valuation_comparison eastmoney 失败: {exc}")

    if result.get("basis") == "peers":
        try:
            llm_result = await _validate_peers_with_llm(
                symbol,
                market_data,
                (result.get("peers") or {}).get("peer_list") or [],
            )
            if llm_result.get("changed") and llm_result.get("peers"):
                stats, peer_list, peer_count = await _fetch_peer_stats(
                    llm_result["peers"], symbol, warnings
                )
                if peer_count > 0:
                    stats["peer_list"] = peer_list
                    stats["source"] = "llm"
                    stats["reason"] = llm_result.get("reason", "")
                    result.update(
                        {
                            "source": "llm",
                            "basis": "llm",
                            "peer_count": peer_count,
                            "matched_industry": str(market_data.get("industry") or ""),
                            "peers": stats,
                            "llm_validated": True,
                        }
                    )
            else:
                result["llm_validated"] = True
                result["llm_reason"] = llm_result.get("reason", "")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"LLM 同行校验失败: {exc}")

    # Always try to attach the whole-industry PE benchmark (used to cross-check
    # peers and as the anchor when peers are missing or unreliable).
    trading_dates = [
        (_latest_trading_date() - timedelta(days=offset)).strftime("%Y%m%d")
        for offset in range(0, 6)
    ]
    for trading_date in trading_dates:
        try:
            frame = await run_blocking(
                lambda trading_date=trading_date: _call_ak(
                    lambda ak: ak.stock_industry_pe_ratio_cninfo(
                        symbol="证监会行业分类",
                        date=trading_date,
                    )
                ),
                timeout=30.0,
                retries=0,
            )
            candidates = [
                str(market_data.get("industry") or ""),
                str(result.get("matched_industry") or ""),
            ]
            matched = _match_industry_row(frame, candidates)
            if matched:
                result["industry"] = {
                    "pe": {
                        "median": matched["pe_median"],
                        "mean": matched["pe_mean"],
                        "weighted": matched["pe_weighted"],
                    }
                }
                result["matched_industry"] = matched["industry_name"]
                result["company_count"] = matched["company_count"]
                if not result.get("peers"):
                    result["basis"] = "industry"
                    result["source"] = "cninfo"
                break
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"get_industry_valuation_comparison cninfo {trading_date} 失败: {exc}")

    if not result.get("peers") and not result.get("industry"):
        raise RuntimeError("行业估值对比数据获取失败")
    return result


async def _fetch_peer_stats(
    peers_config: list[Any],
    symbol: str,
    warnings: list[str],
) -> tuple[dict[str, Any], list[dict[str, Any]], int]:
    """Fetch valuation multiples for a configured/suggested peer list."""
    pe_values: list[float] = []
    pb_values: list[float] = []
    ps_values: list[float] = []
    peer_list: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in peers_config:
        if isinstance(entry, dict):
            code = str(entry.get("code") or "").strip()
            name = str(entry.get("name") or "").strip()
        else:
            code = str(entry or "").strip()
            name = ""
        if not code or code == symbol:
            continue
        if code in seen:
            continue
        seen.add(code)
        try:
            frame = await run_blocking(
                lambda code=code: _call_ak(lambda ak: ak.stock_value_em(symbol=code)),
                timeout=20.0,
                retries=1,
            )
            records = _records_from_frame(
                frame,
                VALUE_EM_COLUMNS,
                limit=1,
                sort_by="数据日期",
                ascending=False,
            )
            if not records:
                continue
            row = records[0]
            pe = _num(row.get("pe_ttm"))
            pb = _num(row.get("pb"))
            ps = _num(row.get("ps"))
            if pe is None and pb is None and ps is None:
                continue
            if pe is not None and pe > 0:
                pe_values.append(pe)
            if pb is not None and pb > 0:
                pb_values.append(pb)
            if ps is not None and ps > 0:
                ps_values.append(ps)
            peer_list.append(
                {
                    "code": code,
                    "name": name or code,
                    "pe_ttm": pe,
                    "pb": pb,
                    "ps": ps,
                }
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"get_industry_valuation_comparison manual peer {code} 失败: {exc}")
    if not peer_list:
        raise RuntimeError("手动可比公司均获取失败")
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


async def _validate_peers_with_llm(
    symbol: str,
    market_data: dict[str, Any],
    peer_list: list[dict[str, Any]],
) -> dict[str, Any]:
    """Ask the LLM to validate/replace the AkShare-sourced comparable list."""
    if not llm_configured() or not peer_list:
        return {"changed": False, "reason": "", "peers": []}
    company_name = str(market_data.get("company_name") or symbol)
    industry = str(market_data.get("industry") or "")
    candidates = [
        {
            "code": str(item.get("code") or ""),
            "name": str(item.get("name") or item.get("code") or ""),
        }
        for item in peer_list
        if item.get("code")
    ]
    prompt = (
        "你是A股行业可比公司评审专家。下面是一只目标股票及其当前候选可比公司列表，"
        "请判断候选列表是否与目标公司真正可比（同行业、业务/规模/盈利模式相近）。\n"
        "只返回一个 JSON 对象："
        '{"changed": true/false, "reason": "一句话说明", '
        '"peers": [{"code": "6位代码", "name": "简称"}]}。'
        "changed=false 时 peers 可以为空；changed=true 时 peers 必须给出"
        "你认为最合适的 A 股可比公司（5-10 家，6 位代码）。\n\n"
        f"目标公司：{symbol} {company_name}（{industry}）\n"
        f"候选可比公司：{json.dumps(candidates, ensure_ascii=False)}\n"
    )
    response = await asyncio.wait_for(
        llm_reply(
            "你只输出严格的 JSON，不要输出任何其他文字。",
            prompt,
            max_tokens=500,
        ),
        timeout=LLM_VALIDATION_TIMEOUT,
    )
    parsed = _extract_json_object(response)
    if parsed is None:
        return {"changed": False, "reason": "", "peers": []}
    peers: list[dict[str, Any]] = []
    for item in parsed.get("peers") or []:
        if isinstance(item, dict):
            code = str(item.get("code") or "").strip()
            name = str(item.get("name") or "").strip()
        else:
            code = str(item or "").strip()
            name = ""
        if code and code != symbol:
            peers.append({"code": code, "name": name or code})
    return {
        "changed": bool(parsed.get("changed")),
        "reason": str(parsed.get("reason") or ""),
        "peers": peers,
    }


def _comparison_stats(frame: pd.DataFrame, symbol: str) -> dict[str, Any]:
    pe_column = _pick_column(frame, ("市盈率-TTM",))
    pb_column = _pick_column(frame, ("市净率-MRQ", "市净率-MRQ(最新)", "PB_MRQ"))
    ps_column = _pick_column(frame, ("市销率-TTM",))
    code_column = _pick_column(frame, ("代码", "证券代码"))
    name_column = _pick_column(frame, ("简称", "名称"))
    working = frame.copy()
    if name_column:
        working = working[
            ~working[name_column].astype(str).isin(["行业中值", "行业平均"])
        ]
    working = working.reset_index(drop=True)
    result: dict[str, Any] = {}
    pe_forward: dict[str, float | None] = {}
    for column in working.columns:
        match = re.fullmatch(r"市盈率-(\d{2})E", str(column))
        if not match:
            continue
        year = 2000 + int(match.group(1))
        values = pd.to_numeric(working[column], errors="coerce").dropna()
        if not values.empty:
            pe_forward[str(year)] = _num(values.median())
    if pe_forward:
        result["pe_forward"] = pe_forward
    if pe_column:
        values = pd.to_numeric(working[pe_column], errors="coerce").dropna()
        values = values[values > 0]
        if not values.empty:
            stock_row = working[
                working[code_column].astype(str).str.zfill(6) == symbol
            ] if code_column else pd.DataFrame()
            stock_pe = (
                _num(pd.to_numeric(stock_row[pe_column], errors="coerce").iloc[0])
                if not stock_row.empty
                else _num(values.iloc[0])
            )
            result["pe"] = {
                "stock": stock_pe,
                "median": _num(values.median()),
                "mean": _num(values.mean()),
            }
    if pb_column:
        values = pd.to_numeric(working[pb_column], errors="coerce").dropna()
        values = values[values > 0]
        if not values.empty:
            result["pb"] = {"median": _num(values.median()), "mean": _num(values.mean())}
    if ps_column:
        values = pd.to_numeric(working[ps_column], errors="coerce").dropna()
        values = values[values > 0]
        if not values.empty:
            result["ps"] = {"median": _num(values.median()), "mean": _num(values.mean())}
    peer_list: list[dict[str, Any]] = []
    for _, row in working.head(15).iterrows():
        code = _text(row.get(code_column)) if code_column else ""
        if code == symbol:
            continue
        peer_list.append(
            {
                "code": code,
                "name": _text(row.get(name_column)) if name_column else "",
                "pe_ttm": _num(row.get(pe_column)) if pe_column else None,
                "pb": _num(row.get(pb_column)) if pb_column else None,
                "ps": _num(row.get(ps_column)) if ps_column else None,
            }
        )
    result["peer_list"] = peer_list
    return result


# ---------------------------------------------------------------------------
# get_earnings_forecast
# ---------------------------------------------------------------------------


async def _get_earnings_forecast(
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {"symbol": symbol, "research_reports": [], "earnings_guidance": []}
    try:
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_research_report_em(symbol=symbol)),
            timeout=40.0,
            retries=1,
        )
        result["research_reports"] = _research_reports(frame, limit=5)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_earnings_forecast research 失败: {exc}")
    try:
        quarter = _latest_quarter_end().strftime("%Y%m%d")
        frame = await run_blocking(
            lambda: _call_ak(lambda ak: ak.stock_yjyg_em(date=quarter)),
            timeout=40.0,
            retries=1,
        )
        if frame is not None and not frame.empty and "股票代码" in frame.columns:
            rows = frame[frame["股票代码"].astype(str).str.zfill(6) == symbol]
            for _, row in rows.head(10).iterrows():
                result["earnings_guidance"].append(
                    {
                        "indicator": str(row.get("预测指标") or ""),
                        "change_pct": _num(row.get("业绩变动幅度")),
                        "forecast_value": _num(row.get("预测数值")),
                        "guidance_type": str(row.get("预告类型") or ""),
                        "announce_date": str(row.get("公告日期") or ""),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"get_earnings_forecast guidance 失败: {exc}")
    if not result["research_reports"] and not result["earnings_guidance"]:
        raise RuntimeError("业绩预测数据获取失败")
    return result


def _research_reports(frame: pd.DataFrame, limit: int = 5) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    year = date.today().year
    year_keys = [str(year + offset) for offset in (0, 1, 2)]
    records: list[dict[str, Any]] = []
    for _, row in frame.head(limit).iterrows():
        record: dict[str, Any] = {
            "date": str(row.get("日期") or ""),
            "rating": str(row.get("东财评级") or ""),
            "org": str(row.get("机构") or ""),
            "industry": str(row.get("行业") or ""),
        }
        for year_key in year_keys:
            eps_key = f"{year_key}-盈利预测-收益"
            pe_key = f"{year_key}-盈利预测-市盈率"
            if eps_key in frame.columns:
                record[f"eps_{year_key}"] = _num(row.get(eps_key))
            if pe_key in frame.columns:
                record[f"pe_{year_key}"] = _num(row.get(pe_key))
        records.append(record)
    return records


# ---------------------------------------------------------------------------
# estimate_fair_value (pure, no network)
# ---------------------------------------------------------------------------


def _estimate_fair_value_tool(
    metrics: dict[str, Any],
    symbol: str = "",
) -> dict[str, Any]:
    cfg, config_source, config_overrides = load_valuation_config(symbol)
    return estimate_fair_value(
        metrics,
        cfg=cfg,
        config_source=config_source,
        config_overrides=config_overrides,
    )


def _get_model_targets_tool(
    metrics: dict[str, Any],
    symbol: str = "",
) -> dict[str, Any]:
    cfg, _, _ = load_valuation_config(symbol)
    models_dir = cfg.get("model_models_dir") or "state/valuation_model"
    return model_targets_from_metrics(metrics, symbol=symbol, models_dir=models_dir)


# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

ToolFunc = Callable[..., Awaitable[dict[str, Any]] | dict[str, Any]]


TOOLS: dict[str, dict[str, Any]] = {
    "get_company_profile": {
        "description": "获取公司名称、行业归属、上市日期与主营业务等基础信息。",
        "cache_seconds": 6 * 3600,
        "func": _get_company_profile,
    },
    "get_financial_statements": {
        "description": "获取最近8期资产负债表、利润表与现金流量表（按报告期累计值）。",
        "cache_seconds": 12 * 3600,
        "func": _get_financial_statements,
    },
    "get_financial_indicators": {
        "description": "获取EPS、BPS、ROE、毛利率、净利率、资产负债率、营收/净利同比等财务指标。",
        "cache_seconds": 12 * 3600,
        "func": _get_financial_indicators,
    },
    "get_financial_abstract": {
        "description": "获取财务摘要主要指标与主营构成（交叉校验）。",
        "cache_seconds": 12 * 3600,
        "func": _get_financial_abstract,
    },
    "get_valuation_snapshot": {
        "description": "获取当前PE-TTM、PE静、PB、PS、PCF、总市值、股本与股息率等估值快照。",
        "cache_seconds": 30 * 60,
        "func": _get_valuation_snapshot,
    },
    "get_historical_valuation_percentile": {
        "description": "获取近三年PE/PB/PCF历史分位。",
        "cache_seconds": 6 * 3600,
        "func": _get_historical_valuation_percentile,
    },
    "get_industry_valuation_comparison": {
        "description": "获取个股估值相对行业的对比（PE/PB/PS中位数）。",
        "cache_seconds": 6 * 3600,
        "func": _get_industry_valuation_comparison,
    },
    "get_earnings_forecast": {
        "description": "获取研报一致预期与业绩预告，为增长假设提供依据。",
        "cache_seconds": 6 * 3600,
        "func": _get_earnings_forecast,
    },
    "estimate_fair_value": {
        "description": "纯计算工具：相对估值+DCF+股息折现综合估算合理股价区间。",
        "cache_seconds": 0,
        "func": _estimate_fair_value_tool,
    },
    "get_model_targets": {
        "description": "本地LightGBM模型推理：预测合理PE/PB/PS目标倍数（无网络）。",
        "cache_seconds": 0,
        "func": _get_model_targets_tool,
    },
}


async def run_tool(
    name: str,
    symbol: str,
    market_data: dict[str, Any],
    warnings: list[str],
    metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Dispatch a tool by name with cache and JSON-safe return values."""
    spec = TOOLS.get(name)
    if spec is None:
        raise ValueError(f"未知工具: {name}")
    func = spec["func"]
    cache_seconds = int(spec.get("cache_seconds", 0))
    period_key = f"{date.today().isoformat()}#v{CACHE_VERSION}"
    if cache_seconds > 0:
        cached = FUNDAMENTAL_CACHE.get(symbol, name, period_key, cache_seconds)
        if cached:
            result = json_loads(cached, {})
            if _cache_shape_valid(name, result):
                return result
    if name in ("estimate_fair_value", "get_model_targets"):
        result = func(metrics or {}, symbol)
    else:
        result = await func(symbol, market_data, warnings)
    if not isinstance(result, dict):
        raise RuntimeError(f"工具 {name} 返回了非字典结果")
    if cache_seconds > 0:
        FUNDAMENTAL_CACHE.put(symbol, name, period_key, json_dumps(result))
    return result


def _cache_shape_valid(name: str, result: Any) -> bool:
    """Reject cached payloads written by older tool versions."""
    if not isinstance(result, dict):
        return False
    if name == "get_industry_valuation_comparison":
        return bool(result.get("peers") or result.get("industry"))
    return True
