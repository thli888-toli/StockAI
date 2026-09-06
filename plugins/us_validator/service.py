"""US valuation validator agent (LLM review + deterministic fallback)."""

from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import date
from typing import Any

from framework.llm import llm_configured, llm_reply
from framework.schemas import TaskRequest
from plugins.stock_common import json_dumps, json_loads
from plugins.us_common import validate_us_symbol


LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "150"))
VERDICT_TO_OVERALL = {
    "低估": "bullish",
    "合理": "neutral",
    "高估": "bearish",
}


def _extract_json(text: str) -> dict[str, Any] | None:
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text or "", re.DOTALL)
    candidate = match.group(1) if match else (text or "")
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(candidate[start : end + 1])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None


def _deterministic_checks(analysis: dict[str, Any]) -> list[dict[str, str]]:
    issues: list[dict[str, str]] = []
    metrics = analysis.get("metrics") or {}
    valuation = analysis.get("valuation") or {}
    statements = metrics.get("statements") or {}
    report_date = statements.get("latest_report_date")
    if report_date:
        try:
            age_days = (date.today() - date.fromisoformat(str(report_date))).days
            if age_days > 400:
                issues.append(
                    {"severity": "warning", "item": f"最新财报期 {report_date} 距今已超过 400 天"}
                )
        except ValueError:
            pass
    fair = valuation.get("fair_value_range") or {}
    mid = fair.get("mid")
    current = metrics.get("current_price")
    if mid is None or mid <= 0:
        issues.append({"severity": "critical", "item": "未能给出估值中枢"})
    elif current is not None and current > 0:
        deviation = (float(current) / float(mid) - 1.0)
        if abs(deviation) > 0.30:
            issues.append(
                {
                    "severity": "warning",
                    "item": f"现价相对中枢偏离 {deviation * 100:.1f}%，请关注方法假设",
                }
            )
    for key in ("eps_ttm", "bps", "current_price", "total_shares"):
        if metrics.get(key) is None:
            issues.append({"severity": "warning", "item": f"缺少 {key}"})
    if not metrics.get("industry_peers") and not metrics.get("historical"):
        issues.append(
            {"severity": "warning", "item": "缺少可比公司/历史分位锚，相对估值可能失真"}
        )
    available = valuation.get("available_methods") or []
    if not available and not fair:
        issues.append({"severity": "critical", "item": "所有估值方法均不可用"})
    return issues


async def _llm_review(analysis: dict[str, Any], checks: list[dict[str, str]]) -> dict[str, Any]:
    payload = {
        "symbol": analysis.get("symbol"),
        "company_name": analysis.get("company_name"),
        "industry": analysis.get("industry"),
        "currency": analysis.get("currency"),
        "as_of": analysis.get("as_of"),
        "metrics": analysis.get("metrics"),
        "valuation": analysis.get("valuation"),
        "warnings": analysis.get("warnings"),
        "data_quality": analysis.get("data_quality"),
        "deterministic_checks": checks,
    }
    prompt = (
        "你是美股估值审阅专家。请只基于 DATA 中的数字与来源（SEC EDGAR / Yahoo Finance）审阅，"
        "不得编造或篡改任何数值。重点：1) 数据时效与来源；2) 方法适用性（亏损/周期/增长龙头）；"
        "3) 关键假设是否自洽（增速 vs 现价隐含、折现率、同行倍数、目标价交叉校验）；"
        "4) 估值区间是否合理，判断低估/合理/高估并给出可信度。"
        '只返回 JSON：{"confidence": "high|medium|low", '
        '"issues": [{"severity":"info|warning|critical","item":"..."}], '
        '"notes": "一段中文说明", "verdict": "低估|合理|高估|数据不足"}。\n\n'
        f"DATA: {json.dumps(payload, ensure_ascii=False, default=str)[:15000]}"
    )
    try:
        response = await asyncio.wait_for(
            llm_reply(
                "你只输出严格的 JSON 对象。",
                prompt,
                max_tokens=900,
            ),
            timeout=LLM_TIMEOUT_SECONDS,
        )
    except Exception:
        return {}
    parsed = _extract_json(response)
    if not parsed:
        return {}
    confidence = str(parsed.get("confidence") or "low")
    if confidence not in ("high", "medium", "low"):
        confidence = "low"
    issues = parsed.get("issues")
    if not isinstance(issues, list):
        issues = []
    return {
        "llm_used": True,
        "confidence": confidence,
        "issues": issues,
        "notes": str(parsed.get("notes") or ""),
        "verdict": str(parsed.get("verdict") or ""),
    }


def _validation_section(
    validation: dict[str, Any],
    checks: list[dict[str, str]],
) -> str:
    lines = ["## LLM 估值校验" if validation.get("llm_used") else "## 确定性估值校验"]
    if validation.get("llm_used"):
        confidence = {"high": "高", "medium": "中", "low": "低"}.get(
            str(validation.get("confidence") or "low"), "低"
        )
        lines.append(f"- LLM 校验可信度：{confidence}。")
        if validation.get("notes"):
            lines.append(f"- 结论说明：{validation['notes']}")
    else:
        lines.append("- 未配置 LLM，以下为确定性规则校验。")
    if checks:
        lines.append("- 校验项：")
        for item in checks:
            severity = item.get("severity") or "info"
            label = {"critical": "严重", "warning": "提示", "info": "说明"}.get(severity, "说明")
            lines.append(f"  - [{label}] {item.get('item')}")
    if validation.get("issues"):
        lines.append("- LLM 提示：")
        for issue in validation.get("issues", []):
            if not isinstance(issue, dict):
                continue
            severity = issue.get("severity") or "info"
            label = {"critical": "严重", "warning": "提示", "info": "说明"}.get(severity, "说明")
            lines.append(f"  - [{label}] {issue.get('item')}")
    lines.append("- 校验只复核假设与数据质量，不修改估值数字。")
    return "\n".join(lines)


class UsValidatorHandler:
    async def run(self, request: TaskRequest) -> str:
        ticker = validate_us_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        fundamental = json_loads(request.inputs.get("fundamental", ""), {})
        analysis = fundamental.get("analysis") if isinstance(fundamental, dict) else None
        report_section = (
            fundamental.get("report_section") if isinstance(fundamental, dict) else None
        )
        if not isinstance(analysis, dict) or not isinstance(report_section, str):
            raise RuntimeError("fundamental 输出格式不正确")

        checks = _deterministic_checks(analysis)
        validation: dict[str, Any] = {"llm_used": False, "issues": checks}
        if llm_configured():
            review = await _llm_review(analysis, checks)
            if review.get("llm_used"):
                validation = review
                validation["deterministic_checks"] = checks
                validation.setdefault("issues", checks)

        section = _validation_section(validation, checks)
        report = (report_section + "\n\n" + section).strip()
        fair = (analysis.get("valuation") or {}).get("fair_value_range") or {}
        metrics = analysis.get("metrics") or {}
        valuation = analysis.get("valuation") or {}
        engine_label = (valuation.get("verdict") or {}).get("label", "")
        llm_verdict = validation.get("verdict") or engine_label
        overall = VERDICT_TO_OVERALL.get(str(llm_verdict or engine_label), "neutral")
        company = str(
            analysis.get("company_name")
            or market_data.get("company_name")
            or ticker
        )
        price_text = f"当前价 {metrics.get('current_price')} 美元"
        mid_text = f"估值中枢 {fair.get('mid')} 美元" if fair.get("mid") else "暂无估值中枢"
        summary = {
            "overall": overall,
            "text": f"{company}：{price_text}，{mid_text}，LLM 校验结论 {llm_verdict or '待人工复核'}。",
        }
        artifact = {
            "report": report,
            "summary": summary,
            "validation": validation,
            "market": "us",
        }
        return json_dumps(artifact)


def build_agent() -> UsValidatorHandler:
    return UsValidatorHandler()
