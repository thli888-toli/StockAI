from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.us_fundamental import edgar  # noqa: E402


def _facts_fixture() -> dict:
    def entry(start, end, val, fp, form, filed):
        return {"start": start, "end": end, "val": val, "fp": fp, "form": form, "filed": filed}

    gaap = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": {
            "units": {
                "USD": [
                    entry("2024-01-01", "2024-12-31", 100_000_000, "FY", "10-K", "2025-02-01"),
                    entry("2024-01-01", "2024-03-31", 24_000_000, "Q1", "10-Q", "2024-05-01"),
                    entry("2024-01-01", "2024-06-30", 26_000_000, "Q2", "10-Q", "2024-08-01"),
                    entry("2024-01-01", "2024-09-30", 28_000_000, "Q3", "10-Q", "2024-11-01"),
                    entry("2025-01-01", "2025-03-31", 30_000_000, "Q1", "10-Q", "2025-05-01"),
                    entry("2025-04-01", "2025-06-30", 32_000_000, "Q2", "10-Q", "2025-08-01"),
                    entry("2025-07-01", "2025-09-30", 35_000_000, "Q3", "10-Q", "2025-11-01"),
                ]
            }
        },
        "NetIncomeLoss": {
            "units": {
                "USD": [
                    entry("2024-01-01", "2024-12-31", 20_000_000, "FY", "10-K", "2025-02-01"),
                    entry("2024-01-01", "2024-03-31", 5_000_000, "Q1", "10-Q", "2024-05-01"),
                    entry("2024-01-01", "2024-06-30", 5_000_000, "Q2", "10-Q", "2024-08-01"),
                    entry("2024-01-01", "2024-09-30", 5_000_000, "Q3", "10-Q", "2024-11-01"),
                    entry("2025-01-01", "2025-03-31", 6_000_000, "Q1", "10-Q", "2025-05-01"),
                    entry("2025-04-01", "2025-06-30", 7_000_000, "Q2", "10-Q", "2025-08-01"),
                    entry("2025-07-01", "2025-09-30", 8_000_000, "Q3", "10-Q", "2025-11-01"),
                ]
            }
        },
        "StockholdersEquity": {
            "units": {
                "USD": [
                    {"end": "2024-12-31", "val": 50_000_000, "fp": "FY", "form": "10-K", "filed": "2025-02-01"},
                    {"end": "2025-09-30", "val": 60_000_000, "fp": "Q3", "form": "10-Q", "filed": "2025-11-01"},
                ]
            }
        },
        "NetCashProvidedByUsedInOperatingActivities": {
            "units": {
                "USD": [
                    entry("2024-01-01", "2024-12-31", 30_000_000, "FY", "10-K", "2025-02-01"),
                    entry("2024-01-01", "2024-03-31", 8_000_000, "Q1", "10-Q", "2024-05-01"),
                    entry("2024-01-01", "2024-06-30", 15_000_000, "Q2", "10-Q", "2024-08-01"),
                    entry("2024-01-01", "2024-09-30", 24_000_000, "Q3", "10-Q", "2024-11-01"),
                    entry("2025-01-01", "2025-03-31", 9_000_000, "Q1", "10-Q", "2025-05-01"),
                    entry("2025-01-01", "2025-06-30", 18_000_000, "Q2", "10-Q", "2025-08-01"),
                    entry("2025-01-01", "2025-09-30", 29_000_000, "Q3", "10-Q", "2025-11-01"),
                ]
            }
        },
    }
    dei = {
        "EntityCommonStockSharesOutstanding": {
            "units": {
                "shares": [
                    {"end": "2025-09-30", "val": 1_000_000, "filed": "2025-11-01"},
                ]
            }
        }
    }
    return {"facts": {"us-gaap": gaap, "dei": dei}}


def test_extract_financials_and_ttm(monkeypatch):
    fixture = _facts_fixture()
    monkeypatch.setattr(edgar, "ticker_to_cik", lambda ticker: 1)
    monkeypatch.setattr(edgar, "company_facts", lambda cik: fixture)
    summary = edgar.sec_fundamentals("TEST")
    assert summary["source"] == "sec_edgar"
    assert summary["revenue_ttm"] == 119_000_000
    assert summary["net_income_ttm"] == 26_000_000
    # OCF TTM = FY2024 30 + FY2025 Q1-Q3 incremental (9+9+11) - FY2024 Q1-Q3 (8+7+9)
    assert summary["ocf_ttm"] == 35_000_000
    assert summary["latest_equity"] == 60_000_000
    assert summary["latest_shares"] == 1_000_000
    assert summary["latest_report_date"] == "2025-09-30"


def test_missing_cash_flow_falls_back_to_annual(monkeypatch):
    fixture = _facts_fixture()
    del fixture["facts"]["us-gaap"]["NetCashProvidedByUsedInOperatingActivities"]["units"]["USD"][-3:]
    monkeypatch.setattr(edgar, "ticker_to_cik", lambda ticker: 1)
    monkeypatch.setattr(edgar, "company_facts", lambda cik: fixture)
    summary = edgar.sec_fundamentals("TEST")
    assert summary["ocf_ttm"] == 30_000_000


def test_latest_shares_filters_future_filings(monkeypatch):
    fixture = _facts_fixture()
    fixture["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"].append(
        {"end": "2099-01-01", "val": 999, "filed": "2099-01-01"}
    )
    monkeypatch.setattr(edgar, "ticker_to_cik", lambda ticker: 1)
    monkeypatch.setattr(edgar, "company_facts", lambda cik: fixture)
    assert edgar.latest_shares_outstanding("TEST") == 1_000_000
