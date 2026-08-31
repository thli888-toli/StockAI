from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plugins.stock_fundamental import valuation_model as vm  # noqa: E402


def _fixture_yjbb(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    codes = [f"{i:06d}" for i in range(1, n + 1)]
    industries = ["白酒", "半导体", "软件", "医药"] * (n // 4)
    return pd.DataFrame(
        {
            "股票代码": codes,
            "每股收益": rng.uniform(0.2, 5, n),
            "净资产收益率": rng.uniform(5, 35, n),
            "每股净资产": rng.uniform(1, 30, n),
            "每股经营现金流量": rng.uniform(-1, 5, n),
            "销售毛利率": rng.uniform(15, 80, n),
            "营业总收入-同比增长": rng.uniform(-20, 80, n),
            "净利润-同比增长": rng.uniform(-30, 100, n),
            "营业总收入-营业总收入": rng.uniform(1e8, 5e10, n),
            "所处行业": industries,
        }
    )


def _fixture_spot(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    codes = [f"{i:06d}" for i in range(1, n + 1)]
    return pd.DataFrame(
        {
            "代码": codes,
            "最新价": rng.uniform(3, 300, n),
            "市盈率-动态": rng.uniform(5, 80, n),
            "市净率": rng.uniform(0.5, 15, n),
            "总市值": rng.uniform(1e9, 5e11, n),
        }
    )


def _fixture_metrics(**overrides) -> dict:
    metrics = {
        "symbol": "600519",
        "total_shares": 1.0e9,
        "current_price": 100.0,
        "eps_ttm": 3.0,
        "roe": 0.25,
        "bps": 15.0,
        "ocf_ttm": 5.0e9,
        "gross_margin": 0.6,
        "revenue_growth_yoy": 0.2,
        "net_profit_yoy": 0.3,
        "industry_name": "白酒",
    }
    metrics.update(overrides)
    return metrics


def test_build_dataset_merges_cross_section():
    dataset = vm.build_dataset(
        _fixture_yjbb(), vm._normalize_spot(_fixture_spot()), "20260630"
    )
    assert len(dataset) == 300
    assert "pe_ttm" in dataset.columns
    assert "pb" in dataset.columns
    assert "ps_ttm" in dataset.columns
    assert "log_mktcap" in dataset.columns
    assert dataset["industry"].notna().all()


def test_train_end_to_end_saves_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(vm, "_fetch_yjbb", lambda report_date: _fixture_yjbb())
    monkeypatch.setattr(
        vm, "_fetch_spot", lambda: vm._normalize_spot(_fixture_spot())
    )
    summary = vm.train(models_dir=tmp_path, report_date="20260630")
    assert summary["report_date"] == "20260630"
    assert set(summary["models"]) == {"pe", "pb", "ps"}
    assert (tmp_path / "pe_model.joblib").exists()
    assert (tmp_path / "pb_model.joblib").exists()
    assert (tmp_path / "ps_model.joblib").exists()
    assert (tmp_path / "training.csv").exists()
    meta = json.loads((tmp_path / "meta.json").read_text(encoding="utf-8"))
    assert meta["report_date"] == "20260630"
    assert meta["models"]["pe"]["n_samples"] == 300
    assert "features" in meta["models"]["pe"]


def test_train_fails_without_data(monkeypatch, tmp_path):
    monkeypatch.setattr(vm, "_fetch_yjbb", lambda report_date: pd.DataFrame())
    monkeypatch.setattr(vm, "_fetch_spot", lambda: _fixture_spot())
    with pytest.raises(RuntimeError, match="无全市场财务数据"):
        vm.train(models_dir=tmp_path, report_date="20260630")


def test_train_skips_future_and_failing_candidates(monkeypatch, tmp_path):
    monkeypatch.setattr(
        vm,
        "_recent_report_dates",
        lambda: ["20260930", "20260630", "20260331"],
    )
    calls: list[str] = []

    def fake_fetch(report_date: str) -> pd.DataFrame:
        calls.append(report_date)
        if report_date == "20260630":
            return _fixture_yjbb()
        return pd.DataFrame()

    monkeypatch.setattr(vm, "_fetch_yjbb", fake_fetch)
    monkeypatch.setattr(
        vm, "_fetch_spot", lambda: vm._normalize_spot(_fixture_spot())
    )
    summary = vm.train(models_dir=tmp_path)
    assert summary["report_date"] == "20260630"
    # The future period is skipped without being fetched, and the failed
    # period is skipped while the next valid period is used.
    assert "20260930" not in calls
    assert "20260630" in calls


def _train_fixture_models(tmp_path: Path) -> Path:
    dataset = vm.build_dataset(
        _fixture_yjbb(), vm._normalize_spot(_fixture_spot()), "20260630"
    )
    trained = {}
    for target in vm.MODEL_TARGETS:
        result = vm._train_target(dataset, target, "20260630")
        assert result is not None, target
        model, stats = result
        joblib.dump(model, tmp_path / vm.MODEL_FILES[target])
        trained[target] = stats
    meta = {
        "version": "1.0",
        "trained_at": "2026-08-31T00:00:00",
        "report_date": "20260630",
        "features": vm.FEATURE_COLUMNS,
        "models": trained,
    }
    (tmp_path / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return tmp_path


def test_model_targets_from_metrics_predicts(tmp_path):
    models_dir = _train_fixture_models(tmp_path)
    result = vm.model_targets_from_metrics(
        _fixture_metrics(), symbol="600519", models_dir=models_dir
    )
    assert result["available"] is True
    assert result["pe"] > 0
    assert result["pb"] > 0
    assert result["ps"] > 0
    assert result["confidence"] == 1.0
    assert result["model_version"] == "1.0"
    assert result["report_date"] == "20260630"
    assert "eps" in result["features_used"]


def test_model_targets_confidence_drops_when_features_missing(tmp_path):
    models_dir = _train_fixture_models(tmp_path)
    metrics = _fixture_metrics(
        total_shares=None,
        current_price=None,
        ocf_ttm=None,
        gross_margin=None,
        revenue_growth_yoy=None,
        net_profit_yoy=None,
        industry_name="",
    )
    result = vm.model_targets_from_metrics(metrics, models_dir=models_dir)
    assert result["available"] is True
    assert result["confidence"] < 1.0
    assert result["pe"] > 0


def test_model_targets_missing_artifacts_returns_unavailable(tmp_path):
    result = vm.model_targets_from_metrics(_fixture_metrics(), models_dir=tmp_path)
    assert result["available"] is False
    assert "未训练" in result["reason"]


def test_normalize_spot_tencent_columns():
    tx = pd.DataFrame(
        {
            "code": ["sh600519", "sz000858", "bj430047"],
            "zxj": [1297.40, 120.0, 8.0],
            "pe_ttm": [19.92, 15.0, 20.0],
            "pn": [6.46, 3.0, 2.0],
            "zsz": [16218.56, 2000.0, 50.0],
        }
    )
    normalized = vm._normalize_spot(tx)
    assert list(normalized["symbol"]) == ["600519", "000858", "430047"]
    assert normalized["total_market_cap"].iloc[0] == pytest.approx(16218.56e8)
    assert normalized["pe_ttm"].iloc[0] == pytest.approx(19.92)
    assert normalized["pb"].iloc[0] == pytest.approx(6.46)


def test_fetch_spot_falls_back_to_tencent(monkeypatch):
    em_frame = pd.DataFrame()
    tx_frame = pd.DataFrame(
        {
            "code": ["sh600519"],
            "zxj": [100.0],
            "pe_ttm": [20.0],
            "pn": [5.0],
            "zsz": [1000.0],
        }
    )

    def fake_retry(func, *args, **kwargs):
        if func.__name__ == "stock_zh_a_spot_em":
            raise ConnectionError("push2 down")
        return tx_frame

    monkeypatch.setattr(vm, "_fetch_with_retry", fake_retry)
    result = vm._fetch_spot()
    assert result["symbol"].iloc[0] == "600519"
    assert result["total_market_cap"].iloc[0] == pytest.approx(1000.0e8)
