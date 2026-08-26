from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.stock_fundamental.valuation import (  # noqa: E402
    dcf_valuation,
    ddm_valuation,
    estimate_fair_value,
    relative_valuation,
)


def _metrics(**overrides) -> dict:
    metrics = {
        "current_price": 20.0,
        "total_shares": 1_000_000_000,
        "eps_ttm": 2.0,
        "bps": 10.0,
        "sps_ttm": 5.0,
        "fcf": 2_000_000_000,
        "dps": 1.0,
        "roe": 0.15,
        "payout_ratio": 0.5,
        "revenue_growth_cagr": 0.10,
        "forecast_growth": 0.08,
        "valuation": {"pe_ttm": 15.0, "pb": 2.0, "ps": 4.0},
        "historical": {
            "pe_ttm": {"p50": 18.0, "percentile": 40.0},
            "pb": {"p50": 2.5, "percentile": 30.0},
        },
        "industry": {"pe": {"median": 20.0}, "pb": {"median": 2.2}},
    }
    metrics.update(overrides)
    return metrics


def test_relative_valuation_uses_historical_p50_when_available():
    result = relative_valuation(_metrics())
    assert result["available"] is True
    assert result["price"] == 30.5  # median(2.0*18, 10.0*2.5)
    implied = {item["metric"]: item["implied_price"] for item in result["detail"]}
    assert implied["pe_ttm"] == 36.0
    assert implied["pb"] == 25.0
    assert "ps" not in implied  # no historical PS and no industry PS median


def test_relative_valuation_falls_back_to_industry_median():
    metrics = _metrics()
    metrics["historical"] = {}
    result = relative_valuation(metrics)
    assert result["available"] is True
    assert result["detail"][0]["target_source"] == "行业中位数"


def test_relative_valuation_unavailable_without_multiples():
    result = relative_valuation(_metrics(historical={}, industry={}))
    assert result["available"] is False


def test_dcf_valuation_uses_weighted_growth_and_sensitivity():
    result = dcf_valuation(_metrics())
    assert result["available"] is True
    assert result["discount_rate"] == 0.10
    assert result["terminal_growth"] == 0.02
    assert result["growth"] == pytest.approx(0.092, abs=1e-6)
    assert result["low"] <= result["price"] <= result["high"]
    assert set(result["sensitivity"]) == {"8%", "12%"}


def test_dcf_valuation_unavailable_without_fcf():
    result = dcf_valuation(_metrics(fcf=None))
    assert result["available"] is False


def test_dcf_valuation_unavailable_with_negative_fcf():
    result = dcf_valuation(_metrics(fcf=-1.0))
    assert result["available"] is False
    assert "自由现金流" in result["notes"][0]


def test_ddm_valuation_uses_roe_times_retention():
    result = ddm_valuation(_metrics())
    assert result["available"] is True
    assert result["growth"] == pytest.approx(0.075, abs=1e-6)
    assert result["price"] == pytest.approx(43.0, abs=0.01)


def test_ddm_valuation_applicability_gates():
    assert ddm_valuation(_metrics(dps=None))["available"] is False
    assert ddm_valuation(_metrics(roe=None))["available"] is False
    assert ddm_valuation(_metrics(roe=-0.05))["available"] is False
    assert ddm_valuation(_metrics(payout_ratio=None))["available"] is False
    assert ddm_valuation(_metrics(current_price=200.0))["available"] is False


def test_estimate_fair_value_combines_methods_and_verdict():
    result = estimate_fair_value(_metrics())
    fair_value = result["fair_value_range"]
    assert fair_value["low"] <= fair_value["mid"] <= fair_value["high"]
    assert result["available_methods"] == ["relative", "dcf", "ddm"]
    assert result["verdict"]["label"] in ("低估", "合理", "高估")
    assert result["verdict"]["current_price"] == 20.0
    assert "不构成投资建议" in result["disclaimer"]


def test_estimate_fair_value_verdict_low_high_bands():
    assert estimate_fair_value(_metrics(current_price=10.0))["verdict"]["label"] == "低估"
    assert estimate_fair_value(_metrics(current_price=100.0))["verdict"]["label"] == "高估"
    assert estimate_fair_value(_metrics(current_price=31.0))["verdict"]["label"] == "合理"


def test_estimate_fair_value_verdict_missing_price():
    result = estimate_fair_value(_metrics(current_price=None))
    assert result["verdict"]["label"] == "数据不足"


def test_estimate_fair_value_raises_without_any_method():
    metrics = _metrics(
        eps_ttm=None,
        bps=None,
        sps_ttm=None,
        fcf=None,
        dps=None,
        roe=None,
    )
    with pytest.raises(ValueError):
        estimate_fair_value(metrics)


def test_estimate_fair_value_excludes_outlier_method():
    metrics = _metrics(
        fcf=1e7,
        revenue_growth_cagr=None,
        forecast_growth=None,
    )
    result = estimate_fair_value(metrics)
    assert result["available_methods"] == ["relative", "ddm"]
    assert result["excluded_methods"][0]["method"] == "dcf"
    assert result["fair_value_range"]["mid"] == 30.5
    assert result["fair_value_range"]["low"] <= result["fair_value_range"]["mid"]
