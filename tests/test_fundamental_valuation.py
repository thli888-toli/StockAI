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
        "forecast_growth": None,
        "forecast_eps": None,
        "forecast_year": None,
        "industry_name": "白酒",
        "valuation": {"pe_ttm": 15.0, "pb": 2.0, "ps": 4.0},
        "historical": {
            "pe_ttm": {"p25": 14.0, "p50": 18.0, "p75": 22.0, "percentile": 40.0},
            "pb": {"p25": 2.0, "p50": 2.5, "p75": 3.0, "percentile": 30.0},
        },
        "industry_peers": {
            "pe": {"median": 16.0, "mean": 18.0},
            "pb": {"median": 2.2, "mean": 2.5},
            "ps": {"median": 3.0, "mean": 3.5},
            "pe_forward": {"2026": 12.0},
            "peer_list": [
                {"code": "600519", "name": "贵州茅台", "pe_ttm": 20.0, "pb": 6.0, "ps": 9.0},
                {"code": "000858", "name": "五粮液", "pe_ttm": 18.0, "pb": 4.0, "ps": 8.0},
            ],
        },
        "industry_bench": {"pe": {"median": 20.0, "mean": 22.0}},
    }
    metrics.update(overrides)
    return metrics


def test_relative_valuation_prefers_peers_median():
    result = relative_valuation(_metrics())
    assert result["available"] is True
    assert result["price"] == pytest.approx(26.43, abs=0.01)  # trimmed mean incl. history blends
    implied = {item["metric"]: item for item in result["detail"]}
    assert implied["pe_ttm"]["implied_price"] == 32.0
    assert implied["pe_ttm"]["target_source"] == "类似公司中位数"
    assert implied["pb"]["implied_price"] == 22.0
    assert implied["ps"]["implied_price"] == 15.0
    assert result["basis"] == "类似公司中位数"
    assert result["peer_names"] == ["贵州茅台", "五粮液"]
    assert result["peer_count"] == 2


def test_relative_valuation_falls_back_to_historical_p50():
    metrics = _metrics(industry_peers={}, industry_bench={})
    result = relative_valuation(metrics)
    assert result["available"] is True
    implied = {item["metric"]: item for item in result["detail"]}
    assert implied["pe_ttm"]["target_multiple"] == 18.0
    assert implied["pe_ttm"]["target_source"] == "自身历史50分位"
    assert result["price"] == pytest.approx(30.5, abs=0.01)  # mean(2*18, 10*2.5)
    assert any("回退" in note for note in result["notes"])


def test_relative_valuation_falls_back_to_industry_median():
    metrics = _metrics(industry_peers={}, historical={})
    result = relative_valuation(metrics)
    assert result["available"] is True
    detail = result["detail"][0]
    assert detail["target_multiple"] == 20.0
    assert detail["target_source"] == "行业整体中位数"
    assert result["price"] == 40.0
    assert any("回退" in note for note in result["notes"])


def test_relative_valuation_adds_forward_pe_when_consensus_eps_exists():
    metrics = _metrics(forecast_eps=2.4, forecast_year=2026)
    result = relative_valuation(metrics)
    implied = {item["metric"]: item for item in result["detail"]}
    assert implied["pe_ttm_fwd"]["base"] == 2.4
    assert implied["pe_ttm_fwd"]["target_multiple"] == 12.0
    assert implied["pe_ttm_fwd"]["implied_price"] == 28.8
    assert implied["pe_ttm_fwd"]["target_source"] == "类似公司中位数(forward PE)"
    assert result["price"] == pytest.approx(27.05, abs=0.01)  # trimmed mean incl. history blends


def test_relative_valuation_skips_forward_pe_without_forward_multiple():
    metrics = _metrics(forecast_eps=2.4, forecast_year=2026)
    metrics["industry_peers"]["pe_forward"] = {}
    result = relative_valuation(metrics)
    assert all(item["metric"] != "pe_ttm_fwd" for item in result["detail"])
    assert any("未计算 forward" in note for note in result["notes"])


def test_relative_valuation_manual_peers_label():
    metrics = _metrics()
    metrics["industry_peers"]["source"] = "manual"
    result = relative_valuation(metrics)
    assert result["basis"] == "手动指定类似公司中位数"
    pe = next(item for item in result["detail"] if item["metric"] == "pe_ttm")
    assert pe["target_source"] == "手动指定类似公司中位数"
    assert any("手动指定类似公司" in note for note in result["notes"])


def test_relative_valuation_llm_peers_label():
    metrics = _metrics()
    metrics["industry_peers"]["source"] = "llm"
    metrics["industry_peers"]["reason"] = "已替换为更贴合的同类可比公司"
    result = relative_valuation(metrics)
    assert result["basis"] == "大模型建议类似公司中位数"
    pe = next(item for item in result["detail"] if item["metric"] == "pe_ttm")
    assert pe["target_source"] == "大模型建议类似公司中位数"
    assert any("大模型校验" in note and "已替换" in note for note in result["notes"])


def test_relative_valuation_keeps_metric_up_to_four_times_median():
    metrics = _metrics()
    metrics["industry_peers"]["pb"] = {"median": 6.0, "mean": 6.5}
    result = relative_valuation(metrics)
    implied = {item["metric"]: item for item in result["detail"]}
    assert implied["pb"]["implied_price"] == 60.0
    assert "pb" in implied  # ~1.9x median, no longer dropped
    assert not any("剔除" in note for note in result["notes"])


def test_relative_valuation_drops_extreme_metric_with_note():
    metrics = _metrics()
    metrics["valuation"]["ps"] = 15.0
    metrics["industry_peers"]["ps"] = {"median": 30.0, "mean": 35.0}
    result = relative_valuation(metrics)
    implied = {item["metric"]: item for item in result["detail"]}
    assert "ps" not in implied  # 150 > 4x median
    assert any("剔除异常放大口径" in note for note in result["notes"])


def test_relative_valuation_peg_calibrates_pe_target():
    metrics = _metrics(forecast_growth=0.20)
    result = relative_valuation(metrics)
    pe = next(item for item in result["detail"] if item["metric"] == "pe_ttm")
    assert pe["target_multiple"] == 20.0  # g% = 20, clamped to [9.6, 24]
    assert any("PEG" in note for note in result["notes"])


def test_relative_valuation_peg_skipped_for_low_growth():
    metrics = _metrics(forecast_growth=0.047)
    result = relative_valuation(metrics)
    pe = next(item for item in result["detail"] if item["metric"] == "pe_ttm")
    assert pe["target_multiple"] == 16.0  # PEG not applied below 15% growth
    assert any("PEG 校准不适用" in note for note in result["notes"])


def test_relative_valuation_downgrades_peers_outside_industry_band():
    metrics = _metrics()
    metrics["industry_peers"]["pe"] = {"median": 35.0, "mean": 38.0}
    result = relative_valuation(metrics)
    pe = next(item for item in result["detail"] if item["metric"] == "pe_ttm")
    assert pe["target_multiple"] == 20.0  # industry median used, peers downgraded
    assert pe["target_source"] == "行业整体中位数"
    assert any("偏离过大" in note for note in result["notes"])


def test_relative_valuation_history_capped_against_anchor():
    metrics = _metrics()
    metrics["historical"]["pe_ttm"]["p50"] = 100.0
    metrics["historical"]["pe_ttm"]["p25"] = 80.0
    metrics["historical"]["pe_ttm"]["p75"] = 120.0
    result = relative_valuation(metrics)
    hist = next(item for item in result["detail"] if item["metric"] == "pe_ttm_hist")
    assert hist["target_multiple"] == 48.0  # 3.0 * peers median 16
    assert any("已按上限" in note for note in result["notes"])


def test_relative_valuation_loss_making_anchors_on_history_pb():
    metrics = _metrics()
    metrics["eps_ttm"] = -0.07
    metrics["bps"] = 3.46
    metrics["sps_ttm"] = 6.19
    metrics["historical"]["pb"] = {"p25": 1.17, "p50": 1.97, "p75": 2.35}
    metrics["industry_peers"]["pb"] = {"median": 7.71, "mean": 8.39}
    metrics["industry_peers"]["ps"] = {"median": 2.53, "mean": 4.74}
    result = relative_valuation(metrics)
    assert result["price"] == 6.82  # 3.46 * 1.97 own-history PB
    assert result["basis"] == "自身历史50分位"
    detail = {item["metric"]: item for item in result["detail"]}
    assert detail["pb_hist"]["weight"] == 1.0
    assert detail["pb_peer"]["weight"] == 0
    assert detail["ps_peer"]["weight"] == 0
    assert any("自身历史 PB 为主锚" in note for note in result["notes"])
    assert not any("回退至自身近3年" in note for note in result["notes"])


def test_relative_valuation_skips_pe_for_cyclical_industries():
    metrics = _metrics(industry_name="证券")
    result = relative_valuation(metrics)
    assert all(item["metric"] != "pe_ttm" for item in result["detail"])
    assert any("周期" in note for note in result["notes"])
    assert result["price"] == 22.0  # weighted median incl. history blend


def test_relative_valuation_band_uses_historical_quartiles():
    result = relative_valuation(_metrics())
    assert result["low"] == pytest.approx(22.86, abs=0.01)
    assert result["high"] == pytest.approx(35.0, abs=0.01)
    assert result["low"] <= result["price"] <= result["high"]


def test_relative_valuation_band_falls_back_to_15_percent():
    metrics = _metrics(historical={})
    result = relative_valuation(metrics)
    for item in result["detail"]:
        assert item["implied_low"] <= item["implied_price"] <= item["implied_high"]


def test_relative_valuation_growth_leader_uses_own_history_median():
    metrics = _metrics(forecast_growth=0.35)
    result = relative_valuation(metrics)
    assert result["available"] is True
    assert result["basis"] == "自身历史50分位(高成长龙头)"
    # PE anchor: 2 * 18 = 36; PB anchor: 10 * 2.5 = 25 -> median 30.5
    assert result["price"] == pytest.approx(30.5, abs=0.01)
    assert result["low"] == pytest.approx(20.0, abs=0.01)
    assert result["high"] == pytest.approx(44.0, abs=0.01)
    detail = {item["metric"]: item for item in result["detail"]}
    assert detail["pe_ttm_hist_leader"]["target_multiple"] == 18.0
    assert detail["pb_hist_leader"]["target_multiple"] == 2.5
    assert detail["pe_ttm_peer"]["weight"] == 0
    assert result["peer_names"] == ["贵州茅台", "五粮液"]
    assert any("高成长龙头分支" in note for note in result["notes"])


def test_relative_valuation_growth_leader_uses_pb_when_pe_unreliable():
    metrics = _metrics(forecast_growth=0.35)
    metrics["historical"]["pe_ttm"] = {
        "p25": -1.0,
        "p50": 100.0,
        "p75": 200.0,
        "percentile": 80.0,
    }
    result = relative_valuation(metrics)
    assert result["available"] is True
    assert result["basis"] == "自身历史50分位(高成长龙头)"
    # Only the PB own-history anchor participates: 10 * 2.5 = 25
    assert result["price"] == pytest.approx(25.0, abs=0.01)
    detail = {item["metric"]: item for item in result["detail"]}
    assert "pe_ttm_hist_leader" not in detail
    assert detail["pb_hist_leader"]["target_multiple"] == 2.5
    assert any("PE 历史分位不可靠" in note for note in result["notes"])


def test_relative_valuation_marks_restructuring_in_progress():
    metrics = _metrics()
    metrics["current_price"] = 30.0
    metrics["eps_ttm"] = 0.2
    metrics["bps"] = 0.8
    metrics["sps_ttm"] = 0.7
    metrics["forecast_growth"] = 0.05
    result = relative_valuation(metrics)
    assert result["available"] is False
    assert result["restructuring_in_progress"] is True
    assert any("重组" in note for note in result["notes"])


def test_relative_valuation_does_not_flag_profitable_high_multiple_tech():
    metrics = _metrics(
        current_price=367.0,
        eps_ttm=4.4,
        bps=31.36,
        sps_ttm=14.74,
        forecast_growth=0.03,
    )
    result = relative_valuation(metrics)
    assert result["available"] is True
    assert not result.get("restructuring_in_progress")


def test_relative_valuation_drops_ps_for_low_margin_low_ps_company():
    metrics = _metrics()
    metrics["valuation"]["ps"] = 0.74
    metrics["sps_ttm"] = 115.0
    metrics["industry_peers"]["ps"] = {"median": 2.0, "mean": 3.0}
    result = relative_valuation(metrics)
    detail = {item["metric"]: item for item in result["detail"]}
    assert "ps" not in detail
    assert any("弃用PS" in note for note in result["notes"])


def test_relative_valuation_drops_ps_when_peer_diverges_without_low_ps():
    metrics = _metrics()
    metrics["valuation"]["ps"] = 4.42
    metrics["sps_ttm"] = 85.32
    metrics["industry_peers"]["ps"] = {"median": 11.1, "mean": 11.31}
    result = relative_valuation(metrics)
    detail = {item["metric"]: item for item in result["detail"]}
    assert "ps" not in detail
    assert any("弃用PS" in note for note in result["notes"])


def test_relative_valuation_drops_pe_on_cyclical_earnings_boom():
    metrics = _metrics(
        current_price=376.88,
        eps_ttm=27.89,
        bps=43.25,
        sps_ttm=85.32,
        forecast_growth=-0.05,
    )
    metrics["valuation"].update({"pe_ttm": 13.51, "pb": 8.71, "ps": 4.42})
    metrics["industry_peers"]["pe"] = {"median": 56.65, "mean": 54.31}
    metrics["industry_peers"]["pb"] = {"median": 8.57, "mean": 9.75}
    metrics["industry_peers"]["ps"] = {"median": 11.1, "mean": 11.31}
    metrics["historical"]["pe_ttm"] = {
        "p25": -480.39,
        "p50": 59.63,
        "p75": 110.33,
    }
    metrics["historical"]["pb"] = {"p25": 5.42, "p50": 6.67, "p75": 15.9}
    result = relative_valuation(metrics)
    detail = {item["metric"]: item for item in result["detail"]}
    assert "pe_ttm" not in detail
    assert "ps" not in detail
    assert any("周期" in note for note in result["notes"])
    # PB peer (43.25 * 8.57) + PB own-history blend (43.25 * 6.67, w0.8)
    assert result["price"] == pytest.approx(334.2, abs=0.1)


def test_relative_valuation_skips_pe_history_when_loss_periods():
    metrics = _metrics()
    metrics["valuation"]["pe_ttm"] = 60.0
    metrics["industry_peers"]["pe"] = {"median": 56.65, "mean": 54.31}
    metrics["historical"]["pe_ttm"] = {
        "p25": -480.39,
        "p50": 59.63,
        "p75": 110.33,
    }
    result = relative_valuation(metrics)
    detail = {item["metric"]: item for item in result["detail"]}
    assert "pe_ttm" in detail
    assert "pe_ttm_hist" not in detail
    assert any("亏损期" in note for note in result["notes"])


def test_relative_valuation_uses_own_history_when_peers_flagged_mismatched():
    metrics = _metrics()
    metrics["industry_peers"]["source"] = "llm"
    metrics["industry_peers"]["reason"] = (
        "候选公司与目标公司业务和规模不匹配；应替换为服务器可比公司。"
    )
    result = relative_valuation(metrics)
    assert result["basis"] == "自身历史50分位"
    detail = {item["metric"]: item for item in result["detail"]}
    assert detail["pe_ttm"]["target_source"] == "自身历史50分位"
    assert detail["pb"]["target_source"] == "自身历史50分位"
    assert any("自身历史分位为主锚" in note for note in result["notes"])


def test_relative_valuation_prefers_own_history_when_peer_pb_ps_understate():
    metrics = _metrics()
    metrics["eps_ttm"] = 2.52
    metrics["bps"] = 17.63
    metrics["sps_ttm"] = 5.05
    metrics["valuation"].update({"pe_ttm": 84.58, "pb": 12.08, "ps": 42.2})
    metrics["industry_peers"]["source"] = "llm"
    metrics["industry_peers"]["pe"] = {"median": 36.79, "mean": 58.37}
    metrics["industry_peers"]["pb"] = {"median": 5.14, "mean": 6.43}
    metrics["industry_peers"]["ps"] = {"median": 13.33, "mean": 12.05}
    metrics["industry_bench"]["pe"] = {"median": 82.32, "mean": 172.19}
    metrics["historical"]["pe_ttm"] = {"p25": 61.85, "p50": 68.44, "p75": 85.73}
    metrics["historical"]["pb"] = {"p25": 7.63, "p50": 10.16, "p75": 13.18}
    result = relative_valuation(metrics)
    detail = {item["metric"]: item for item in result["detail"]}
    assert detail["pb"]["target_multiple"] == 10.16
    assert detail["pb"]["target_source"] == "自身历史50分位"
    assert "ps" not in detail
    assert any("显著低于" in note for note in result["notes"])


def test_relative_valuation_keeps_validated_peers_above_industry_band():
    metrics = _metrics()
    metrics["eps_ttm"] = 4.42
    metrics["bps"] = 31.36
    metrics["sps_ttm"] = 14.74
    metrics["valuation"].update({"pe_ttm": 83.09, "pb": 11.7, "ps": 24.9})
    metrics["industry_peers"]["source"] = "llm"
    metrics["industry_peers"]["reason"] = "已替换为更贴合的可比公司"
    metrics["industry_peers"]["pe"] = {"median": 113.26, "mean": 347.05}
    metrics["industry_peers"]["pb"] = {"median": 11.13, "mean": 11.31}
    metrics["industry_peers"]["ps"] = {"median": 17.04, "mean": 17.66}
    metrics["industry_bench"]["pe"] = {"median": 51.26, "mean": 111.65}
    metrics["historical"]["pe_ttm"] = {"p25": 73.15, "p50": 86.09, "p75": 99.11}
    metrics["historical"]["pb"] = {"p25": 6.09, "p50": 7.14, "p75": 9.57}
    result = relative_valuation(metrics)
    pe = next(item for item in result["detail"] if item["metric"] == "pe_ttm")
    assert pe["target_multiple"] == 113.26
    assert pe["target_source"] == "大模型建议类似公司中位数"


def test_estimate_fair_value_restructuring_returns_placeholder():
    metrics = _metrics(
        current_price=30.0,
        eps_ttm=0.2,
        bps=0.8,
        sps_ttm=0.7,
        forecast_growth=0.05,
        fcf=None,
        dps=None,
    )
    result = estimate_fair_value(metrics)
    assert result["verdict"]["label"] == "重组/注入中"
    assert result["fair_value_range"]["mid"] is None
    assert result["available_methods"] == []
    assert any("重组" in item["reason"] for item in result["excluded_methods"])


def test_relative_valuation_growth_leader_uses_forward_pe_primary():
    metrics = _metrics(
        current_price=241.6,
        eps_ttm=5.11,
        bps=38.57,
        sps_ttm=22.27,
        forecast_growth=0.96,
        forecast_eps=10.03,
        forecast_year=2026,
    )
    metrics["industry_peers"]["pe_forward"] = {
        "2025": 34.3,
        "2026": 18.43,
        "2027": 12.02,
    }
    metrics["industry_peers"]["pb"] = {"median": 7.78, "mean": 8.34}
    metrics["historical"]["pe_ttm"] = {
        "p25": 48.03,
        "p50": 60.38,
        "p75": 70.88,
    }
    metrics["historical"]["pb"] = {
        "p25": 5.93,
        "p50": 13.94,
        "p75": 17.14,
    }
    result = relative_valuation(metrics)
    assert result["basis"] == "forward PE + PB(高成长龙头)"
    assert result["price"] == pytest.approx(213.23, abs=0.1)
    detail = {item["metric"]: item for item in result["detail"]}
    assert detail["pe_ttm_fwd_leader"]["target_multiple"] == 18.43
    assert detail["pb_peer_leader"]["target_multiple"] == pytest.approx(6.26, abs=0.01)
    assert any("forward PE" in note for note in result["notes"])


def test_relative_valuation_unavailable_without_multiples():
    result = relative_valuation(
        _metrics(industry_peers={}, industry_bench={}, historical={})
    )
    assert result["available"] is False


def test_dcf_valuation_uses_weighted_growth_and_sensitivity():
    result = dcf_valuation(_metrics())
    assert result["available"] is True
    assert result["discount_rate"] == 0.10
    assert result["terminal_growth"] == 0.02
    assert result["growth"] == pytest.approx(0.10, abs=1e-6)
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
    assert fair_value["mid"] == 35.5
    assert fair_value["low"] <= fair_value["mid"] <= fair_value["high"]
    assert result["available_methods"] == ["relative", "dcf", "ddm"]
    assert result["verdict"]["label"] in ("低估", "合理", "高估")
    assert result["verdict"]["current_price"] == 20.0
    assert "不构成投资建议" in result["disclaimer"]


def test_estimate_fair_value_verdict_low_high_bands():
    assert estimate_fair_value(_metrics(current_price=10.0))["verdict"]["label"] == "低估"
    assert estimate_fair_value(_metrics(current_price=100.0))["verdict"]["label"] == "高估"
    assert estimate_fair_value(_metrics(current_price=35.5))["verdict"]["label"] == "合理"


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
    assert result["fair_value_range"]["mid"] == pytest.approx(26.43, abs=0.01)
    assert result["fair_value_range"]["low"] <= result["fair_value_range"]["mid"]
