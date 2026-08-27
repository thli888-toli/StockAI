from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import GraphManifest, TaskRequest  # noqa: E402
from plugins.stock_analyst.service import (  # noqa: E402
    StockAnalystHandler,
    cross_validate,
)
from plugins.stock_common import (  # noqa: E402
    compute_macd,
    confidence_from_probability,
    direction_from_probability,
    prepare_daily_features,
    resample_ohlcv,
    validate_symbol,
)
from plugins.stock_news.service import _cross_validate, _telegraph_records  # noqa: E402
from plugins.stock_quant.service import _build_quant_payload  # noqa: E402
from plugins.stock_data.service import StockDataHandler  # noqa: E402


def _daily_frame(rows: int = 180) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = pd.Series(range(100, 100 + rows), dtype="float64")
    frame = pd.DataFrame(
        {
            "日期": dates,
            "开盘": close - 1.0,
            "收盘": close,
            "最高": close + 1.0,
            "最低": close - 2.0,
            "成交量": 1000000.0,
            "成交额": 20000000.0,
            "换手率": 1.0,
        }
    )
    return frame


def test_validate_symbol_accepts_six_digits_and_rejects_bad_input():
    assert validate_symbol("600519") == "600519"
    with pytest.raises(ValueError):
        validate_symbol("AAPL")


def test_macd_has_expected_columns_and_trend_value():
    close = pd.Series(range(1, 101), dtype="float64")
    macd = compute_macd(close)
    assert list(macd.columns) == ["macd", "signal", "histogram"]
    assert len(macd) == len(close)
    assert macd["histogram"].iloc[-1] > 0


def test_weekly_and_monthly_resampling_reduces_rows_and_preserves_ohlcv():
    daily = _daily_frame()
    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(daily["日期"]),
            "open": daily["开盘"],
            "high": daily["最高"],
            "low": daily["最低"],
            "close": daily["收盘"],
            "volume": daily["成交量"],
            "amount": daily["成交额"],
            "turnover": daily["换手率"],
        }
    )
    weekly = resample_ohlcv(normalized, "W-FRI")
    monthly = resample_ohlcv(normalized, "ME")
    assert len(weekly) < len(normalized)
    assert len(monthly) < len(weekly)
    assert {"open", "high", "low", "close", "volume"}.issubset(weekly.columns)


def test_prepare_daily_features_has_required_columns_without_nan():
    features = prepare_daily_features(_daily_frame())
    core = {
        "macd",
        "macd_signal",
        "macd_histogram",
        "rsi14",
        "ma20",
        "ma66",
        "volatility20",
        "volume_ratio",
        "return_1d",
        "return_5d",
        "return_20d",
        "close_ma20_ratio",
        "close_ma66_ratio",
        "atr14",
        "bollinger_bandwidth",
        "bollinger_pctb",
        "close_high20_ratio",
        "close_low20_ratio",
        "amount_ratio",
    }
    sparse = {"ma154", "ma250", "close_ma154_ratio", "close_ma250_ratio"}
    assert core.issubset(features.columns)
    assert sparse.issubset(features.columns)
    assert not features[list(core)].isna().any().any()
    # Short series leave long-window MA columns as NaN (not zero-filled), so
    # training code can drop them instead of learning from constants.
    assert features["close_ma250_ratio"].isna().all()


def test_news_cross_validation_matches_only_when_title_and_date_align():
    eastmoney = [
        {
            "title": "关于公司发布重大合同的公告",
            "published_at": "2026-08-20 09:00:00",
            "url": "https://eastmoney.example/1",
            "source": "东方财富",
        }
    ]
    cninfo = [
        {
            "title": "公司发布重大合同的公告",
            "published_at": "2026-08-21 10:00:00",
            "url": "https://cninfo.example/1",
            "source": "巨潮资讯",
        },
        {
            "title": "无关的季度业绩说明",
            "published_at": "2026-08-21 10:00:00",
            "url": "https://cninfo.example/2",
            "source": "巨潮资讯",
        },
    ]
    matched, _ = _cross_validate(eastmoney, cninfo)
    assert len(matched) == 1
    assert matched[0]["confidence"] == "high"


def test_telegraph_records_filters_by_company_and_industry_keywords():
    frame = pd.DataFrame(
        {
            "标题": ["赤峰黄金披露重大合同", "某公司业绩预增", "黄金价格创阶段新高"],
            "内容": [
                "赤峰黄金签订重大合同",
                "与目标公司无关的内容",
                "有色金属板块走强，黄金价格新高",
            ],
            "发布时间": ["2026-08-22 10:00:00", "2026-08-22 11:00:00", "2026-08-22 12:00:00"],
            "链接": ["https://example.com/1", "https://example.com/2", "https://example.com/3"],
        }
    )
    records = _telegraph_records(frame, "财联社", "cls_news", ["赤峰黄金", "黄金"], limit=10)
    assert len(records) == 2
    assert {r["title"] for r in records} == {"赤峰黄金披露重大合同", "黄金价格创阶段新高"}
    assert records[0]["source"] == "财联社"
    assert records[0]["source_type"] == "cls_news"
    assert records[0]["url"].startswith("https://")


def test_direction_and_confidence_thresholds():
    assert direction_from_probability(0.53) == "up"
    assert direction_from_probability(0.47) == "down"
    assert direction_from_probability(0.51) == "flat"
    assert confidence_from_probability(0.75) == 0.5


def test_quant_payload_is_neutral_when_history_is_insufficient():
    payload = _build_quant_payload("600519", [{"close": 1.0} for _ in range(50)], "2026-08-22")
    data = __import__("json").loads(payload)
    assert data["horizons"]["5d"]["direction"] == "flat"
    assert data["horizons"]["5d"]["up_probability"] == 0.5
    assert data["warnings"]


def test_quant_payload_ensembles_lstm_and_lightgbm(monkeypatch):
    from plugins.stock_quant import service as quant_service

    monkeypatch.setattr(
        quant_service,
        "_lightgbm_probability",
        lambda df, target, min_train: 0.4,
    )
    monkeypatch.setattr(
        quant_service,
        "_lstm_probability",
        lambda df, target, min_train, seq_len, epochs: 0.4,
    )
    monkeypatch.setattr(
        quant_service,
        "_lightgbm_walk_forward",
        lambda df, target, min_train, validation: (0.6, len(df), [], []),
    )
    monkeypatch.setattr(
        quant_service,
        "_lstm_walk_forward",
        lambda df, target, min_train, validation, seq_len, epochs: (0.6, len(df), [], []),
    )
    monkeypatch.setattr(quant_service, "_fit_calibration", lambda y_true, y_pred: None)
    monkeypatch.setattr(quant_service, "_period_features", lambda records, freq: pd.DataFrame())
    monkeypatch.setattr(quant_service, "_period_ohlcv", lambda records, freq: pd.DataFrame())

    records = [{"close": float(index + 1)} for index in range(150)]
    payload = quant_service._build_quant_payload("600519", records, "2026-08-22")
    data = __import__("json").loads(payload)

    assert data["model"] == "LSTM+LightGBM"
    assert "models" not in data
    assert set(data["horizons"]) == {"5d", "15d", "1w", "1mo"}
    assert data["horizons"]["5d"]["direction"] == "down"
    assert data["backtest"]["walk_forward_auc"] == 0.6
    assert data["weekly_backtest"]["sample_count"] == 0


def test_monthly_signal_combines_macd_and_ma():
    from plugins.stock_quant import service as quant_service

    up_frame = pd.DataFrame({"close": [float(value) for value in range(100, 130)]})
    up_signal, up_count = quant_service._monthly_signal(up_frame)
    assert up_count == 30
    assert up_signal == {"up_probability": 0.7, "direction": "up", "confidence": 0.4}

    down_frame = pd.DataFrame({"close": [float(value) for value in range(130, 100, -1)]})
    down_signal, _ = quant_service._monthly_signal(down_frame)
    assert down_signal == {"up_probability": 0.3, "direction": "down", "confidence": 0.4}

    tiny_frame = pd.DataFrame({"close": [float(value) for value in range(10)]})
    tiny_signal, tiny_count = quant_service._monthly_signal(tiny_frame)
    assert tiny_count == 10
    assert tiny_signal["direction"] == "flat"


def test_quant_uses_monthly_model_when_enough_monthly_history(monkeypatch):
    from plugins.stock_quant import service as quant_service

    monkeypatch.setattr(
        quant_service,
        "_lightgbm_probability",
        lambda df, target, min_train: 0.65,
    )
    monkeypatch.setattr(
        quant_service,
        "_lstm_probability",
        lambda df, target, min_train, seq_len, epochs: 0.55,
    )
    monkeypatch.setattr(
        quant_service,
        "_lightgbm_walk_forward",
        lambda df, target, min_train, validation: (0.62, len(df), [], []),
    )
    monkeypatch.setattr(
        quant_service,
        "_lstm_walk_forward",
        lambda df, target, min_train, validation, seq_len, epochs: (0.58, len(df), [], []),
    )
    monkeypatch.setattr(quant_service, "_fit_calibration", lambda y_true, y_pred: None)

    daily = [{"close": float(index + 1)} for index in range(150)]
    monthly = []
    for index in range(90):
        close = 10.0 + index
        monthly.append(
            {
                "date": f"20{20 + index // 12:02d}-{index % 12 + 1:02d}-28",
                "open": close - 1.0,
                "high": close + 1.0,
                "low": close - 2.0,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    payload = json.loads(
        quant_service._build_quant_payload("600519", daily, "2026-08-22", monthly)
    )
    assert payload["horizons"]["1mo"]["direction"] == "up"
    assert payload["monthly_backtest"]["walk_forward_auc"] == 0.6
    assert payload["monthly_backtest"]["sample_count"] == 71


def test_quant_monthly_gate_uses_usable_feature_rows(monkeypatch):
    from plugins.stock_quant import service as quant_service

    def boom_prob(df, target, *args, **kwargs):
        if target == "target_1mo":
            raise AssertionError("monthly model should not train below usable threshold")
        return 0.5

    def boom_wf(df, target, *args, **kwargs):
        if target == "target_1mo":
            raise AssertionError("monthly model should not train below usable threshold")
        return (0.5, len(df), [], [])

    monkeypatch.setattr(quant_service, "_lightgbm_probability", boom_prob)
    monkeypatch.setattr(quant_service, "_lstm_probability", boom_prob)
    monkeypatch.setattr(quant_service, "_lightgbm_walk_forward", boom_wf)
    monkeypatch.setattr(quant_service, "_lstm_walk_forward", boom_wf)
    monkeypatch.setattr(quant_service, "_fit_calibration", lambda y_true, y_pred: None)
    daily = [{"close": float(index + 1)} for index in range(150)]
    # 73 raw monthly bars -> 54 feature rows -> 53 usable < 60 -> rule fallback
    monthly = [
        {
            "date": f"20{20 + index // 12:02d}-{index % 12 + 1:02d}-28",
            "open": 10.0 + index,
            "high": 12.0 + index,
            "low": 9.0 + index,
            "close": 11.0 + index,
            "volume": 1_000_000.0,
        }
        for index in range(73)
    ]
    payload = json.loads(
        quant_service._build_quant_payload("600519", daily, "2026-08-22", monthly)
    )
    assert payload["monthly_backtest"]["walk_forward_auc"] is None
    assert payload["monthly_backtest"]["sample_count"] == 73
    assert any("insufficient monthly history" in warning for warning in payload["warnings"])


def test_quant_falls_back_to_monthly_rule_when_insufficient_history(monkeypatch):
    from plugins.stock_quant import service as quant_service

    monkeypatch.setattr(
        quant_service,
        "_lightgbm_probability",
        lambda df, target, min_train: 0.65,
    )
    monkeypatch.setattr(
        quant_service,
        "_lstm_probability",
        lambda df, target, min_train, seq_len, epochs: 0.55,
    )
    daily = [{"close": float(index + 1)} for index in range(150)]
    monthly = [
        {
            "date": f"20{20 + index // 12:02d}-{index % 12 + 1:02d}-28",
            "open": 10.0 + index,
            "high": 12.0 + index,
            "low": 9.0 + index,
            "close": 11.0 + index,
            "volume": 1_000_000.0,
        }
        for index in range(30)
    ]
    payload = json.loads(
        quant_service._build_quant_payload("600519", daily, "2026-08-22", monthly)
    )
    assert payload["horizons"]["1mo"]["direction"] == "up"
    assert payload["horizons"]["1mo"]["up_probability"] == 0.7
    assert payload["monthly_backtest"]["sample_count"] == 30
    assert any("insufficient monthly history" in warning for warning in payload["warnings"])


def test_usable_feature_columns_drops_sparse_long_ma():
    from plugins.stock_quant import service as quant_service

    features = prepare_daily_features(_daily_frame(rows=150))
    columns = quant_service._usable_feature_columns(features)
    assert "close_ma20_ratio" in columns
    assert "atr14" in columns
    assert "amount_ratio" in columns
    assert "close_ma154_ratio" not in columns
    assert "close_ma250_ratio" not in columns


def test_run_blocking_timeout_has_readable_message():
    import asyncio
    import time as _time

    from plugins.stock_common import run_blocking

    async def go():
        await run_blocking(lambda: _time.sleep(1.0), timeout=0.05, retries=0)

    with pytest.raises(TimeoutError) as exc_info:
        asyncio.run(go())
    assert "秒" in str(exc_info.value)


def test_quant_cache_store_roundtrip(tmp_path):
    from plugins.stock_quant.quant_cache import QuantCacheStore

    store = QuantCacheStore(tmp_path / "quant.db")
    assert store.get("600519", "hash1") is None
    store.put("600519", "hash1", '{"model":"LSTM"}')
    assert store.get("600519", "hash1") == '{"model":"LSTM"}'


def test_lstm_model_builds_sequences():
    import numpy as np

    from plugins.stock_quant import lstm_model

    features = np.array(
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0], [6.0, 7.0]],
        dtype=float,
    )
    labels = np.array([0, 1, 0, 1, 0, 1], dtype=float)
    standardized, mean, std = lstm_model.standardize(features)
    sequences, aligned = lstm_model.build_sequences(standardized, labels, seq_len=3)
    assert standardized.shape == features.shape
    assert sequences.shape == (4, 3, 2)
    assert aligned.shape == (4,)


def test_cross_validate_llm_and_quant_conviction():
    quant = {
        "horizons": {
            "5d": {"direction": "up", "confidence": 0.7},
            "15d": {"direction": "up", "confidence": 0.7},
            "1w": {"direction": "up", "confidence": 0.7},
            "1mo": {"direction": "up", "confidence": 0.7},
        }
    }
    llm_signal = {
        "5d": {"direction": "up"},
        "15d": {"direction": "up"},
        "1w": {"direction": "up"},
        "1mo": {"direction": "up"},
    }
    agree = cross_validate(llm_signal, quant)
    assert agree["overall"] == "high"

    llm_signal["1mo"] = {"direction": "down"}
    disagree = cross_validate(llm_signal, quant)
    assert disagree["overall"] == "low"


def test_stock_graph_manifest_is_valid():
    manifest_data = yaml.safe_load((ROOT / "config" / "orchestration.yaml").read_text(encoding="utf-8"))
    manifest = GraphManifest.model_validate(manifest_data)
    assert manifest.entry == "stock_data"
    assert set(manifest.nodes) == {
        "stock_data",
        "stock_news",
        "stock_quant",
        "stock_fundamental",
        "stock_analyst",
    }
    assert manifest.nodes["stock_analyst"].input["quant"] == "{quant}"
    assert manifest.nodes["stock_analyst"].input["fundamental"] == "{fundamental}"
    assert manifest.nodes["stock_fundamental"].input["market_data"] == "{market_data}"
    edges = {(edge.source, edge.to) for edge in manifest.edges}
    assert ("stock_data", "stock_news") in edges
    assert ("stock_data", "stock_quant") in edges
    assert ("stock_data", "stock_fundamental") in edges
    assert ("stock_fundamental", "stock_analyst") in edges
    assert ("stock_news", "stock_analyst") in edges
    assert ("stock_quant", "stock_analyst") in edges


@pytest.mark.asyncio
async def test_analyst_deterministic_fallback_report(monkeypatch):
    monkeypatch.setattr("plugins.stock_analyst.service.llm_configured", lambda: False)
    request = TaskRequest(
        query="600519",
        inputs={
            "market_data": '{"symbol":"600519","company_name":"测试","industry":"白酒","macd":{"daily":{"trend":"bullish"}},"latest":{},"as_of":"2026-08-22"}',
            "news": '{"source_counts":{"eastmoney":1,"cninfo":1,"cross_validated":1},"warnings":[]}',
            "quant": '{"horizons":{"1d":{"direction":"up","up_probability":0.6,"confidence":0.2},"5d":{"direction":"up","up_probability":0.6,"confidence":0.2},"20d":{"direction":"up","up_probability":0.6,"confidence":0.2}},"backtest":{}}',
        },
    )
    report = await StockAnalystHandler().run(request)
    assert "免责声明" in report
    assert "600519" in report
    assert "## 基本面与估值" in report
    assert "基本面数据暂不可用" in report


@pytest.mark.asyncio
async def test_analyst_fallback_report_embeds_fundamental_section(monkeypatch):
    monkeypatch.setattr("plugins.stock_analyst.service.llm_configured", lambda: False)
    fundamental = {
        "report_section": (
            "## 基本面与估值（测试 600519）\n"
            "- 合理股价估算（多方法综合）：区间 1000.00–1250.00 元，中枢 1100.00 元。\n"
            "- 判断：低估。"
        ),
        "summary": {
            "valuation_verdict": "低估",
            "fair_value_range": {"low": 1000.0, "mid": 1100.0, "high": 1250.0},
        },
    }
    request = TaskRequest(
        query="600519",
        inputs={
            "market_data": '{"symbol":"600519","company_name":"测试","industry":"白酒","macd":{"daily":{"trend":"bullish"}},"latest":{},"as_of":"2026-08-22"}',
            "news": '{"source_counts":{"eastmoney":1,"cninfo":1,"cross_validated":1},"warnings":[]}',
            "quant": '{"horizons":{"5d":{"direction":"up","up_probability":0.6,"confidence":0.2}},"backtest":{}}',
            "fundamental": json.dumps(fundamental),
        },
    )
    report = await StockAnalystHandler().run(request)
    assert "## 基本面与估值（测试 600519）" in report
    assert "中枢 1100.00 元" in report
    assert "基本面数据暂不可用" not in report


def test_xq_symbol_uses_uppercase_market_prefix():
    assert StockDataHandler._xq_symbol("600988") == "SH600988"
    assert StockDataHandler._xq_symbol("000001") == "SZ000001"


def test_extract_affiliate_industry_handles_dict_and_string_forms():
    assert StockDataHandler._extract_affiliate_industry({"ind_name": "贵金属"}) == "贵金属"
    assert StockDataHandler._extract_affiliate_industry("{'ind_name': '贵金属'}") == "贵金属"
    assert StockDataHandler._extract_affiliate_industry("") == ""


def test_company_info_cninfo_parses_name_and_industry():
    frame = pd.DataFrame(
        {
            "A股简称": ["赤峰黄金"],
            "公司名称": ["赤峰吉隆黄金矿业集团股份有限公司"],
            "所属行业": ["有色金属矿采选业"],
        }
    )

    class FakeAk:
        @staticmethod
        def stock_profile_cninfo(symbol):
            assert symbol == "600988"
            return frame

    info = StockDataHandler._company_info_from_cninfo(FakeAk, "600988")
    assert info == {"company_name": "赤峰黄金", "industry": "有色金属矿采选业"}


def test_fetch_company_info_falls_back_when_eastmoney_fails():
    cninfo_frame = pd.DataFrame(
        {"A股简称": ["赤峰黄金"], "所属行业": ["有色金属矿采选业"]}
    )

    class FakeAk:
        @staticmethod
        def stock_individual_info_em(symbol):
            raise RuntimeError("eastmoney unavailable")

        @staticmethod
        def stock_profile_cninfo(symbol):
            return cninfo_frame

        @staticmethod
        def stock_individual_basic_info_xq(symbol):
            raise AssertionError("should not be reached")

    info = StockDataHandler._fetch_company_info("600988")
    assert info == {"company_name": "赤峰黄金", "industry": "有色金属矿采选业"}


@pytest.mark.asyncio
async def test_analyst_uses_fallback_when_llm_call_raises(monkeypatch):
    async def boom(system, user, max_tokens=2000):
        raise RuntimeError("llm unavailable")

    monkeypatch.setattr("plugins.stock_analyst.service.llm_configured", lambda: True)
    monkeypatch.setattr("plugins.stock_analyst.service.llm_reply", boom)
    request = TaskRequest(
        query="600519",
        inputs={
            "market_data": '{"symbol":"600519","company_name":"测试","industry":"白酒","macd":{"daily":{"trend":"bullish"}},"latest":{},"as_of":"2026-08-22"}',
            "news": '{"source_counts":{"eastmoney":1,"cninfo":1,"cross_validated":1},"warnings":[]}',
            "quant": '{"horizons":{"1d":{"direction":"up","up_probability":0.6,"confidence":0.2},"5d":{"direction":"up","up_probability":0.6,"confidence":0.2},"20d":{"direction":"up","up_probability":0.6,"confidence":0.2}},"backtest":{}}',
        },
    )
    report = await StockAnalystHandler().run(request)
    assert "A股市场观察报告" in report


@pytest.mark.asyncio
async def test_analyst_returns_json_report_and_summary(monkeypatch):
    monkeypatch.setattr("plugins.stock_analyst.service.llm_configured", lambda: False)
    request = TaskRequest(
        query="600519",
        inputs={
            "market_data": '{"symbol":"600519","company_name":"测试","industry":"白酒","macd":{"daily":{"trend":"bullish"}},"latest":{},"as_of":"2026-08-22"}',
            "news": '{"source_counts":{"eastmoney":1,"cninfo":1,"cross_validated":1},"warnings":[]}',
            "quant": '{"horizons":{"5d":{"direction":"up","up_probability":0.6,"confidence":0.2},"15d":{"direction":"up","up_probability":0.6,"confidence":0.2},"1w":{"direction":"down","up_probability":0.4,"confidence":0.2},"1mo":{"direction":"up","up_probability":0.7,"confidence":0.4}},"backtest":{}}',
        },
    )
    result = await StockAnalystHandler().run(request)
    data = __import__("json").loads(result)
    assert set(data) == {"report", "summary"}
    assert data["summary"]["overall"] in ("bullish", "bearish", "neutral")
    assert data["summary"]["text"]
    assert "免责声明" in data["report"]


def test_analyst_system_prompt_requires_auc_discount():
    prompt = StockAnalystHandler._system_prompt()
    assert "AUC" in prompt
    assert "折价" in prompt


def test_analyst_build_prompt_includes_compact_market_and_news():
    market_data = {
        "symbol": "600519",
        "company_name": "贵州茅台",
        "industry": "白酒",
        "as_of": "2026-08-22",
        "latest": {"close": 1500.0, "pct_change": 1.5},
        "macd": {"daily": {"trend": "bullish"}},
        "stats": {"pct_change_20d": 5.0},
        "recent_daily": [
            {
                "date": "2026-08-21",
                "open": 1490.0,
                "high": 1510.0,
                "low": 1485.0,
                "close": 1500.0,
                "volume": 100000,
                "pct_change": 1.5,
            }
        ]
        * 6,
        "daily_features": [
            {
                "rsi14": 60.0,
                "ma20": 1480.0,
                "ma66": 1460.0,
                "ma154": 1440.0,
                "ma250": 1400.0,
                "volatility20": 0.02,
                "return_1d": 0.01,
                "macd": 0.5,
                "macd_signal": 0.4,
                "macd_histogram": 0.1,
            }
        ],
    }
    news = {
        "source_counts": {},
        "cross_validated": [],
        "eastmoney_news": [
            {
                "title": "标题",
                "published_at": "2026-08-22 10:00:00",
                "summary": "摘要",
                "url": "https://example.com/1",
                "source": "东方财富",
            }
        ],
        "cninfo_disclosures": [],
        "secondary_media_news": [],
        "secondary_media_source": "",
        "warnings": [],
    }

    prompt = StockAnalystHandler._build_prompt("600519", market_data, news, {"horizons": {}})

    assert "recent_daily" in prompt
    assert "indicators" in prompt
    assert '"rsi14"' in prompt
    assert '"ma66"' in prompt
    assert '"url"' not in prompt
    assert '"source"' not in prompt
