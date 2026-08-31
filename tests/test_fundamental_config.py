from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.stock_fundamental import config as valuation_config  # noqa: E402


def _default_payload() -> dict:
    return {
        "config_version": "1.0",
        "discount_rate": 0.10,
        "terminal_growth": 0.02,
        "forecast_years": 5,
        "target_percentile": 0.50,
        "fallback_growth": 0.05,
        "sensitivity_rates": [0.08, 0.12],
        "verdict_band": 0.10,
        "max_growth": 0.30,
        "min_dividend_yield": 0.01,
        "method_weights": {"relative": 1.0, "dcf": 0.8, "ddm": 0.5},
        "outlier_band": [0.2, 5.0],
        "cyclical_keywords": ["证券", "券商", "银行", "保险", "有色", "钢铁", "煤炭", "航运", "地产", "航空"],
        "peg_factor": 1.0,
        "min_peg_growth": 0.15,
        "target_band": 0.15,
        "metric_outlier_factor": 4.0,
        "history_weight": 0.8,
        "history_cap": 3.0,
        "peer_industry_band": [0.6, 1.6],
        "growth_leader_threshold": 0.30,
        "restructuring_pb_ratio": 10.0,
        "restructuring_ps_ratio": 10.0,
        "restructuring_max_eps": 0.5,
        "restructuring_min_bps": 1.0,
        "restructuring_min_sps": 1.0,
        "ps_peer_divergence_factor": 2.5,
        "cyclical_boom_pe_ratio": 0.5,
        "leader_history_weight": 0.5,
        "leader_history_cap_factor": 1.5,
        "peer_own_premium_factor": 2.0,
        "peer_mismatch_keywords": ["不匹配", "应替换为"],
        "model_enabled": True,
        "model_min_confidence": 0.5,
        "model_anchor_weight": 0.5,
        "model_models_dir": "state/valuation_model",
        "leader_primary_min_weight": 1.0,
        "growth_cagr_weight": 0.6,
        "peg_band": [0.6, 1.5],
        "ddm_growth_cap": 0.10,
    }


@pytest.fixture
def config_dir(tmp_path, monkeypatch):
    valuation_config.clear_config_cache()
    monkeypatch.setattr(valuation_config, "CONFIG_DIR", tmp_path)
    (tmp_path / "default.json").write_text(
        json.dumps(_default_payload(), ensure_ascii=False), encoding="utf-8"
    )
    yield tmp_path
    valuation_config.clear_config_cache()


def test_default_config_loads_all_known_keys(config_dir):
    cfg, source, overrides = valuation_config.load_valuation_config(None)
    assert source == "default"
    assert overrides == []
    assert cfg["discount_rate"] == 0.10
    assert cfg["method_weights"]["relative"] == 1.0
    assert cfg["cyclical_keywords"][0] == "证券"


def test_per_symbol_override_merges(config_dir):
    (config_dir / "600519.json").write_text(
        json.dumps(
            {
                "discount_rate": 0.08,
                "verdict_band": 0.2,
                "model_anchor_weight": 0.3,
                "manual_peers": [{"code": "000858", "name": "五粮液"}],
                "manual_fair_value": {"low": 15.0, "mid": 18.0, "high": 21.0},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg, source, overrides = valuation_config.load_valuation_config("600519")
    assert source == "600519.json"
    assert set(overrides) == {
        "discount_rate",
        "verdict_band",
        "model_anchor_weight",
        "manual_peers",
        "manual_fair_value",
    }
    assert cfg["discount_rate"] == 0.08
    assert cfg["terminal_growth"] == 0.02  # inherited from default
    assert cfg["verdict_band"] == 0.2
    assert cfg["model_anchor_weight"] == 0.3
    assert cfg["manual_peers"][0]["code"] == "000858"
    assert cfg["manual_fair_value"]["mid"] == 18.0


def test_invalid_values_fall_back_to_default(config_dir):
    (config_dir / "600519.json").write_text(
        json.dumps({"discount_rate": "oops", "forecast_years": "x", "verdict_band": 0.05}),
        encoding="utf-8",
    )
    cfg, source, overrides = valuation_config.load_valuation_config("600519")
    assert cfg["discount_rate"] == 0.10
    assert cfg["forecast_years"] == 5
    assert cfg["verdict_band"] == 0.05


def test_unknown_keys_are_preserved(config_dir):
    (config_dir / "600519.json").write_text(
        json.dumps({"future_param": 123, "discount_rate": 0.09}),
        encoding="utf-8",
    )
    cfg, _, overrides = valuation_config.load_valuation_config("600519")
    assert cfg["future_param"] == 123
    assert "future_param" in overrides
    assert cfg["discount_rate"] == 0.09


def test_manual_peers_priority_and_clear(config_dir):
    assert valuation_config.symbol_manual_peers_present("600519") is False
    assert valuation_config.symbol_manual_peers("600519") == []

    (config_dir / "600519.json").write_text(
        json.dumps({"manual_peers": ["000858", {"code": "600809", "name": "山西汾酒"}]}),
        encoding="utf-8",
    )
    valuation_config.clear_config_cache()
    assert valuation_config.symbol_manual_peers_present("600519") is True
    peers = valuation_config.symbol_manual_peers("600519")
    assert peers[0] == "000858"
    assert peers[1]["code"] == "600809"

    (config_dir / "600519.json").write_text(
        json.dumps({"manual_peers": []}),
        encoding="utf-8",
    )
    valuation_config.clear_config_cache()
    assert valuation_config.symbol_manual_peers_present("600519") is True
    assert valuation_config.symbol_manual_peers("600519") == []
