from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.schemas import TaskRequest  # noqa: E402
from plugins.us_validator import service as validator_service  # noqa: E402


def _fundamental(report="## 基本面与估值（美股）\n- 合理股价估算：中枢 100 美元。") -> str:
    analysis = {
        "symbol": "TEST",
        "company_name": "Test Corp",
        "currency": "USD",
        "metrics": {
            "current_price": 95.0,
            "eps_ttm": 5.0,
            "bps": 20.0,
            "total_shares": 1_000_000_000.0,
            "valuation": {"pe_ttm": 19.0, "pb": 4.75},
            "statements": {"latest_report_date": "2026-03-31"},
        },
        "valuation": {
            "fair_value_range": {"low": 80.0, "mid": 100.0, "high": 120.0},
            "available_methods": ["relative", "dcf"],
            "verdict": {"label": "低估"},
        },
        "warnings": [],
        "data_quality": {"missing": []},
    }
    return json.dumps(
        {
            "analysis": analysis,
            "report_section": report,
            "summary": {"valuation_verdict": "低估"},
        }
    )


def test_validator_deterministic_fallback(monkeypatch):
    monkeypatch.setattr(validator_service, "llm_configured", lambda: False)
    payload = json.loads(
        asyncio.run(
            validator_service.UsValidatorHandler().run(
                TaskRequest(
                    query="TEST",
                    inputs={
                        "market_data": "{}",
                        "fundamental": _fundamental(),
                    },
                )
            )
        )
    )
    assert payload["validation"]["llm_used"] is False
    assert "确定性估值校验" in payload["report"]
    assert payload["summary"]["overall"] == "bullish"
    assert payload["market"] == "us"


def test_validator_llm_review(monkeypatch):
    monkeypatch.setattr(validator_service, "llm_configured", lambda: True)

    async def fake_reply(system, prompt, max_tokens=700):
        return (
            '{"confidence": "high", "issues": [{"severity": "info", "item": "ok"}], '
            '"notes": "假设自洽", "verdict": "低估"}'
        )

    monkeypatch.setattr(validator_service, "llm_reply", fake_reply)
    payload = json.loads(
        asyncio.run(
            validator_service.UsValidatorHandler().run(
                TaskRequest(
                    query="TEST",
                    inputs={
                        "market_data": "{}",
                        "fundamental": _fundamental(),
                    },
                )
            )
        )
    )
    validation = payload["validation"]
    assert validation["llm_used"] is True
    assert validation["confidence"] == "high"
    assert any(issue.get("item") == "ok" for issue in validation["issues"])
    assert "LLM 估值校验" in payload["report"]


def test_validator_critical_missing_fair_value(monkeypatch):
    monkeypatch.setattr(validator_service, "llm_configured", lambda: False)
    missing = json.loads(_fundamental())
    missing["analysis"]["valuation"]["fair_value_range"] = {}
    payload = json.loads(
        asyncio.run(
            validator_service.UsValidatorHandler().run(
                TaskRequest(
                    query="TEST",
                    inputs={
                        "market_data": "{}",
                        "fundamental": json.dumps(missing),
                    },
                )
            )
        )
    )
    assert any(
        issue.get("severity") == "critical" and "估值中枢" in issue.get("item", "")
        for issue in payload["validation"]["issues"]
    )


def test_validator_rejects_malformed_fundamental():
    with pytest.raises(RuntimeError, match="格式不正确"):
        asyncio.run(
            validator_service.UsValidatorHandler().run(
                TaskRequest(
                    query="TEST",
                    inputs={"market_data": "{}", "fundamental": '{"analysis": []}'},
                )
            )
        )
