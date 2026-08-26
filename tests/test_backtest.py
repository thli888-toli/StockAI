from __future__ import annotations

import asyncio
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import backtest as bt  # noqa: E402


def _bars(rows: int = 220) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype="float64")
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 1.0,
            "high": close + 1.0,
            "low": close - 2.0,
            "close": close,
            "volume": 1000000.0,
            "amount": 20000000.0,
            "turnover": 1.0,
            "pct_change": 0.5,
        }
    )


def test_sample_dates_respects_cadence_and_bounds():
    bars = _bars()
    sampled = bt.sample_dates(
        bars,
        date(2024, 2, 1),
        date(2024, 3, 1),
        cadence=5,
    )
    assert len(sampled) > 0
    assert all(date(2024, 2, 1) <= item.date() <= date(2024, 3, 1) for item in sampled)
    assert all((sampled.iloc[i + 1] - sampled.iloc[i]).days >= 3 for i in range(len(sampled) - 1))


def test_sample_dates_non_overlap():
    bars = _bars(240)
    sampled = bt.sample_dates(
        bars,
        date(2024, 2, 1),
        date(2024, 12, 1),
        cadence=5,
        non_overlap=True,
    )
    positions = bars.index[bars["date"].isin(sampled)].tolist()
    assert len(positions) > 0
    for index in range(1, len(positions)):
        assert positions[index] - positions[index - 1] >= bt.MIN_GAP_TRADING_DAYS


def test_features_truncation_has_no_future_dates():
    bars = _bars()
    sim_date = bars["date"].iloc[150]
    features = bt.prepare_daily_features(bars.iloc[:151])
    assert features["date"].max() <= sim_date


def test_evaluate_direction():
    assert bt.evaluate_direction("up", True) is True
    assert bt.evaluate_direction("up", False) is False
    assert bt.evaluate_direction("down", False) is True
    assert bt.evaluate_direction("flat", True) is None


def test_compute_metrics_ignores_flat():
    rows = [
        {"5d_hit": True, "summary_hit": True},
        {"5d_hit": None, "summary_hit": False},
        {"5d_hit": False, "summary_hit": True},
    ]
    metrics = bt.compute_metrics(rows)
    assert metrics["5d"]["samples"] == 3
    assert metrics["5d"]["valid"] == 2
    assert metrics["5d"]["correct"] == 1
    assert metrics["5d"]["flat"] == 1
    assert metrics["5d"]["accuracy"] == 0.5


def test_wilson_interval_bounds():
    low, high = bt.wilson_interval(10, 10)
    assert low is not None and high is not None
    assert 0.0 <= low <= high <= 1.0


def test_compute_metrics_includes_always_up_baseline():
    rows = [
        {"5d_hit": True, "5d_actual": "up"},
        {"5d_hit": False, "5d_actual": "down"},
        {"5d_hit": None, "5d_actual": "up"},
    ]
    metrics = bt.compute_metrics(rows)
    assert metrics["5d"]["baseline_always_up"] == 0.6667
    assert metrics["5d"]["baseline_always_down"] == 0.3333


def test_compute_metrics_significance_fields():
    rows = [
        {"5d_hit": index < 24, "5d_actual": "up" if index < 18 else "down"}
        for index in range(30)
    ]
    metrics = bt.compute_metrics(rows)
    assert metrics["5d"]["dominant_baseline"] == 0.6
    assert metrics["5d"]["p_value_vs_baseline"] is not None
    assert metrics["5d"]["significant"] in (True, False)


def test_write_outputs_handles_numpy_scalars(tmp_path, monkeypatch):
    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)
    labels = ["5d", "15d", "1w", "1mo", "summary"]
    metrics = {
        label: {
            "samples": 1,
            "valid": 1,
            "correct": 1,
            "flat": 0,
            "accuracy": np.float64(1.0),
            "ci_low": None,
            "ci_high": None,
            "baseline_always_up": np.float64(0.5),
            "baseline_always_down": np.float64(0.5),
            "dominant_baseline": np.float64(0.5),
            "p_value_vs_baseline": None,
            "significant": np.bool_(True),
        }
        for label in labels
    }
    outputs = bt.write_outputs([], {"600988": metrics}, {})
    json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))


@pytest.mark.asyncio
async def test_backtest_symbol_and_outputs(monkeypatch, tmp_path):
    def fake_quant(symbol, records, as_of):
        return json.dumps(
            {
                "horizons": {
                    horizon: {"direction": "up", "up_probability": 0.6, "confidence": 0.2}
                    for horizon, _ in bt.HORIZONS
                },
                "backtest": {},
            }
        )

    async def fake_summary(market_data, quant, use_llm=True):
        return {"overall": "bullish", "text": "偏多"}

    monkeypatch.setattr(bt, "_build_quant_payload", fake_quant)
    monkeypatch.setattr(bt, "generate_llm_summary", fake_summary)
    monkeypatch.setattr(bt, "RESULTS_DIR", tmp_path)

    bars = _bars()
    rows = await bt.backtest_symbol(
        "600988",
        bars,
        date(2024, 10, 1),
        date(2024, 12, 1),
        cadence=5,
        use_llm=False,
    )
    assert rows
    for row in rows:
        assert row["5d_hit"] is True
        assert row["summary_hit"] is True

    metrics = bt.compute_metrics(rows)
    assert metrics["5d"]["correct"] == metrics["5d"]["valid"]
    outputs = bt.write_outputs(rows, {"600988": metrics}, {"symbols": ["600988"]})
    assert Path(outputs["csv"]).exists()
    assert Path(outputs["json"]).exists()
