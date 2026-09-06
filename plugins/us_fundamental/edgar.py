"""SEC EDGAR CompanyFacts helpers for US fundamentals (official, non-China)."""

from __future__ import annotations

import os
from datetime import date, datetime
from typing import Any

import requests

from plugins.stock_common import disable_http_proxy
from plugins.us_cache import US_CACHE


COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"

REVENUE_TAGS = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
)
NET_INCOME_TAGS = ("NetIncomeLoss",)
EQUITY_TAGS = ("StockholdersEquity",)
OCF_TAGS = ("NetCashProvidedByUsedInOperatingActivities",)
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment",)
DIVIDEND_TAGS = ("PaymentsOfDividendsCommonStock", "PaymentsOfDividends")


def _headers() -> dict[str, str]:
    ua = os.getenv(
        "SEC_EDGAR_USER_AGENT",
        "StockAI Research contact@example.com",
    )
    return {"User-Agent": ua, "Accept-Encoding": "gzip, deflate"}


def _fetch_json(url: str, namespace: str, cache_key: str, max_age: int) -> Any:
    cached = US_CACHE.get(namespace, cache_key, max_age)
    if cached is not None:
        return cached
    with disable_http_proxy():
        response = requests.get(url, headers=_headers(), timeout=30.0)
        response.raise_for_status()
        data = response.json()
    US_CACHE.put(namespace, cache_key, data)
    return data


def ticker_to_cik(ticker: str) -> int:
    """Resolve a ticker against the official SEC company tickers list."""
    key = (ticker or "").strip().upper()
    if not key:
        raise ValueError("ticker is required")
    data = _fetch_json(
        COMPANY_TICKERS_URL,
        "sec",
        "company_tickers",
        max_age=7 * 86400,
    )
    if not isinstance(data, dict):
        raise RuntimeError("SEC company tickers unavailable")
    mapping: dict[str, int] = {}
    for item in data.values():
        if not isinstance(item, dict):
            continue
        raw_ticker = str(item.get("ticker") or "").upper()
        cik = item.get("cik_str")
        if raw_ticker and cik is not None:
            mapping[raw_ticker] = int(cik)
    for candidate in (key, key.replace("-", "."), key.replace(".", "-")):
        if candidate in mapping:
            return mapping[candidate]
    raise RuntimeError(
        f"ticker {key} not found in SEC company tickers (ADR/class shares may "
        "need a ticker alias)"
    )


def sec_company_title(ticker: str) -> str:
    """Official SEC company name from the tickers mapping ('' when unknown)."""
    key = (ticker or "").strip().upper()
    data = _fetch_json(
        COMPANY_TICKERS_URL,
        "sec",
        "company_tickers",
        max_age=7 * 86400,
    )
    if not isinstance(data, dict):
        return ""
    for candidate in (key, key.replace("-", "."), key.replace(".", "-")):
        for item in data.values():
            if not isinstance(item, dict):
                continue
            if str(item.get("ticker") or "").upper() == candidate:
                return str(item.get("title") or "")
    return ""


def company_facts(cik: int) -> dict[str, Any]:
    """Download the official CompanyFacts JSON for a CIK."""
    url = FACTS_URL.format(f"{int(cik):010d}")
    return _fetch_json(url, "sec", f"companyfacts_{int(cik)}", max_age=12 * 3600)


def _points(facts: dict[str, Any], tags: tuple[str, ...]) -> list[dict[str, Any]]:
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    points: list[dict[str, Any]] = []
    for tag in tags:
        for entry in (gaap.get(tag) or {}).get("units", {}).get("USD") or []:
            if not isinstance(entry, dict) or entry.get("val") is None:
                continue
            points.append({"tag": tag, **entry})
    return points


def _as_date(value: Any) -> date | None:
    if not value:
        return None
    try:
        parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d")
    except (TypeError, ValueError):
        return None
    return parsed.date()


def _record_kind(entry: dict[str, Any]) -> str | None:
    fp = str(entry.get("fp") or "").upper()
    if fp == "FY":
        return "annual"
    if fp in ("Q1", "Q2", "Q3", "Q4"):
        return "quarterly"
    form = str(entry.get("form") or "").upper()
    if form.startswith("10-K") or form.startswith("20-F"):
        return "annual"
    if form.startswith("10-Q"):
        return "quarterly"
    start = _as_date(entry.get("start"))
    end = _as_date(entry.get("end"))
    if start and end and end > start:
        days = (end - start).days
        if days >= 300:
            return "annual"
        if 40 <= days <= 200:
            return "quarterly"
    return None


def extract_financials(facts: dict[str, Any]) -> dict[str, Any]:
    """Build normalized annual/quarterly fact records from CompanyFacts."""
    revenue_points = sorted(
        _points(facts, REVENUE_TAGS),
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
    )
    income_points = sorted(
        _points(facts, NET_INCOME_TAGS),
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
    )
    equity_points = sorted(
        _points(facts, EQUITY_TAGS),
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
    )
    ocf_points = sorted(
        _points(facts, OCF_TAGS),
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
    )
    capex_points = sorted(
        _points(facts, CAPEX_TAGS),
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
    )
    dividend_points = sorted(
        _points(facts, DIVIDEND_TAGS),
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
    )

    annual: dict[str, dict[str, Any]] = {}
    quarterly: dict[str, dict[str, Any]] = {}
    equity_records: dict[str, dict[str, Any]] = {}

    def _put(
        bucket: dict[str, dict[str, Any]],
        end_key: str,
        tag_field: str,
        entry: dict[str, Any],
    ) -> None:
        existing = bucket.setdefault(
            end_key,
            {
                "end": end_key,
                "kind": bucket is annual and "annual" or "quarterly",
                "filed": str(entry.get("filed") or ""),
            },
        )
        if entry.get("val") is not None:
            existing[tag_field] = float(entry["val"])

    for entry in revenue_points:
        end = _as_date(entry.get("end"))
        kind = _record_kind(entry)
        if end is None or kind not in ("annual", "quarterly"):
            continue
        _put(annual if kind == "annual" else quarterly, end.isoformat(), "revenue", entry)
    for entry in income_points:
        end = _as_date(entry.get("end"))
        kind = _record_kind(entry)
        if end is None or kind not in ("annual", "quarterly"):
            continue
        _put(
            annual if kind == "annual" else quarterly,
            end.isoformat(),
            "net_income",
            entry,
        )
    for entry in ocf_points:
        end = _as_date(entry.get("end"))
        kind = _record_kind(entry)
        if end is None or kind not in ("annual", "quarterly"):
            continue
        _put(annual if kind == "annual" else quarterly, end.isoformat(), "ocf", entry)
    for entry in capex_points:
        end = _as_date(entry.get("end"))
        kind = _record_kind(entry)
        if end is None or kind not in ("annual", "quarterly"):
            continue
        _put(annual if kind == "annual" else quarterly, end.isoformat(), "capex", entry)
    for entry in dividend_points:
        end = _as_date(entry.get("end"))
        kind = _record_kind(entry)
        if end is None or kind not in ("annual", "quarterly"):
            continue
        _put(
            annual if kind == "annual" else quarterly,
            end.isoformat(),
            "dividends",
            entry,
        )
    for entry in equity_points:
        end = _as_date(entry.get("end"))
        if end is None:
            continue
        existing = equity_records.setdefault(
            end.isoformat(),
            {"end": end.isoformat(), "filed": str(entry.get("filed") or "")},
        )
        if entry.get("val") is not None:
            existing["equity"] = float(entry["val"])

    return {
        "annual": sorted(annual.values(), key=lambda item: item["end"]),
        "quarterly": sorted(quarterly.values(), key=lambda item: item["end"]),
        "equity": sorted(equity_records.values(), key=lambda item: item["end"]),
    }


def _quarterly_incremental(
    records: list[dict[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    """Convert YTD-cumulative quarterly facts into per-quarter values.

    SEC cash-flow/dividend facts inside 10-Qs are typically reported as
    year-to-date cumulative amounts while income-statement facts are per
    period.  For a given fiscal year we subtract each quarter's cumulative
    value from the previous quarter's cumulative value.
    """
    annual_ends = sorted(
        {
            str(item["end"])
            for item in records
            if item.get("kind") == "annual" and item.get("end")
        }
    )

    def _year_group(end: str) -> str | None:
        future = [candidate for candidate in annual_ends if candidate >= end]
        return min(future) if future else None

    groups: dict[str | None, list[dict[str, Any]]] = {}
    for item in sorted(
        [
            record
            for record in records
            if record.get("kind") == "quarterly"
            and record.get("end")
            and record.get(field) is not None
        ],
        key=lambda record: str(record["end"]),
    ):
        groups.setdefault(_year_group(str(item["end"])), []).append(item)
    converted: list[dict[str, Any]] = []
    for group_items in groups.values():
        previous: float | None = None
        for item in group_items:
            value = float(item[field])
            incremental = (
                value - previous
                if previous is not None and value >= previous
                else value
            )
            converted.append({**item, field: incremental})
            previous = value
    retained = [
        record
        for record in records
        if record.get("kind") != "quarterly"
        or record.get("end") is None
        or record.get(field) is None
    ]
    return retained + converted


def _ttm_flow(
    records: list[dict[str, Any]],
    field: str,
    cumulative: bool = False,
) -> float | None:
    """Trailing twelve months ending at the latest reported period.

    Preferred path: latest annual + quarters reported after the annual end
    minus the same fiscal quarters of the prior year. Falls back to the sum of
    the last four continuous quarters, then the latest annual.
    """
    if cumulative:
        records = _quarterly_incremental(records, field)
    annuals = sorted(
        [
            record
            for record in records
            if record.get("kind") == "annual"
            and record.get("end")
            and record.get(field) is not None
        ],
        key=lambda record: str(record["end"]),
    )
    quarterlies = sorted(
        [
            record
            for record in records
            if record.get("kind") == "quarterly"
            and record.get("end")
            and record.get(field) is not None
        ],
        key=lambda record: str(record["end"]),
    )
    if annuals:
        latest_annual = annuals[-1]
        annual_value = float(latest_annual[field])
        annual_end = date.fromisoformat(str(latest_annual["end"]))
        after = [
            item
            for item in quarterlies
            if date.fromisoformat(str(item["end"])) > annual_end
        ]
        if after:
            prior_sum = 0.0
            for item in after:
                current_end = date.fromisoformat(str(item["end"]))
                candidates = [
                    float(prior[field])
                    for prior in quarterlies
                    if 330 <= (current_end - date.fromisoformat(str(prior["end"]))).days <= 400
                ]
                if candidates:
                    prior_sum += max(candidates)
            return annual_value + sum(float(item[field]) for item in after) - prior_sum
        return annual_value
    quarterly = [
        record
        for record in records
        if record.get("kind") == "quarterly"
        and record.get(field) is not None
        and record.get("end")
    ]
    if len(quarterly) >= 4:
        last_four = quarterly[-4:]
        try:
            span_days = (
                date.fromisoformat(last_four[-1]["end"])
                - date.fromisoformat(last_four[0]["end"])
            ).days
        except (TypeError, ValueError):
            span_days = 0
        if 240 <= span_days <= 420:
            return float(sum(float(item[field]) for item in last_four))
    if quarterly:
        return float(sum(float(item[field]) for item in quarterly[-4:]))
    return None


def ttm_as_of(
    records: list[dict[str, Any]],
    field: str,
    as_of: str,
    cumulative: bool = False,
) -> float | None:
    """TTM value for a field as of a historical date."""
    if cumulative:
        records = _quarterly_incremental(records, field)
    annuals = sorted(
        [
            record
            for record in records
            if record.get("kind") == "annual"
            and record.get("end")
            and str(record["end"]) <= as_of
            and record.get(field) is not None
        ],
        key=lambda record: str(record["end"]),
    )
    quarterlies = sorted(
        [
            record
            for record in records
            if record.get("kind") == "quarterly"
            and record.get("end")
            and str(record["end"]) <= as_of
            and record.get(field) is not None
        ],
        key=lambda record: str(record["end"]),
    )
    if annuals:
        latest_annual = annuals[-1]
        annual_value = float(latest_annual[field])
        annual_end = date.fromisoformat(str(latest_annual["end"]))
        after = [
            item
            for item in quarterlies
            if date.fromisoformat(str(item["end"])) > annual_end
        ]
        if after:
            prior_sum = 0.0
            for item in after:
                current_end = date.fromisoformat(str(item["end"]))
                candidates = [
                    float(prior[field])
                    for prior in quarterlies
                    if 330
                    <= (current_end - date.fromisoformat(str(prior["end"]))).days
                    <= 400
                ]
                if candidates:
                    prior_sum += max(candidates)
            return annual_value + sum(float(item[field]) for item in after) - prior_sum
        return annual_value
    if len(quarterlies) >= 4:
        return float(sum(float(item[field]) for item in quarterlies[-4:]))
    return None


def financials_summary(records: dict[str, Any]) -> dict[str, Any]:
    """Compute TTM totals and the latest balance-sheet equity."""
    all_records = list(records.get("annual") or []) + list(records.get("quarterly") or [])
    equity = list(records.get("equity") or [])
    latest_equity = None
    if equity:
        latest_equity = float(equity[-1]["equity"])
    return {
        "revenue_ttm": _ttm_flow(all_records, "revenue"),
        "net_income_ttm": _ttm_flow(all_records, "net_income"),
        "ocf_ttm": _ttm_flow(all_records, "ocf", cumulative=True),
        "capex_ttm": _ttm_flow(all_records, "capex", cumulative=True),
        "dividends_ttm": _ttm_flow(all_records, "dividends", cumulative=True),
        "latest_equity": latest_equity,
        "latest_report_date": all_records[-1]["end"] if all_records else None,
    }


def latest_shares_outstanding(ticker: str) -> float | None:
    """Latest common-shares outstanding reported to the SEC (dei fact)."""
    facts = company_facts(ticker_to_cik(ticker))
    dei = ((facts.get("facts") or {}).get("dei") or {})
    concept = (dei.get("EntityCommonStockSharesOutstanding") or {}).get("units") or {}
    entries = concept.get("shares") or concept.get("USD") or []
    today = date.today().isoformat()
    valid = [
        entry
        for entry in entries
        if isinstance(entry, dict)
        and entry.get("val") is not None
        and str(entry.get("filed") or "") <= today
    ]
    if not valid:
        return None
    latest = max(valid, key=lambda entry: str(entry.get("filed") or ""))
    return float(latest["val"])


def sec_fundamentals(ticker: str) -> dict[str, Any]:
    """Official SEC data -> normalized financial summary for one ticker."""
    cik = ticker_to_cik(ticker)
    facts = company_facts(cik)
    records = extract_financials(facts)
    summary = financials_summary(records)
    if summary.get("revenue_ttm") is None and summary.get("net_income_ttm") is None:
        raise RuntimeError(f"SEC EDGAR has no usable financial facts for {ticker}")
    return {
        "cik": cik,
        "source": "sec_edgar",
        "annual": list(reversed((records.get("annual") or [])[-4:])),
        "quarters": list(reversed((records.get("quarterly") or [])[-8:])),
        "latest_equity": summary["latest_equity"],
        "latest_shares": latest_shares_outstanding(ticker),
        "revenue_ttm": summary["revenue_ttm"],
        "net_income_ttm": summary["net_income_ttm"],
        "ocf_ttm": summary["ocf_ttm"],
        "capex_ttm": summary["capex_ttm"],
        "dividends_ttm": summary["dividends_ttm"],
        "latest_report_date": summary["latest_report_date"],
        "record_count": len(all_records := list(records.get("annual") or []) + list(records.get("quarterly") or [])),
    }


__all__ = [
    "ticker_to_cik",
    "sec_company_title",
    "company_facts",
    "extract_financials",
    "financials_summary",
    "latest_shares_outstanding",
    "sec_fundamentals",
    "ttm_as_of",
]
