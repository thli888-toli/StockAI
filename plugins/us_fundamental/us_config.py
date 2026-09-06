"""Per-ticker valuation configuration for US stocks.

Defaults live in ``config/valuation/us_default.json``; per-ticker overrides in
``config/valuation/us/{TICKER}.json``. Unknown keys are preserved so future
parameters can be introduced without breaking existing files.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "valuation"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "us_default.json"
SYMBOL_DIR = CONFIG_DIR / "us"

_INT_KEYS = {"forecast_years"}
_FLOAT_KEYS = {
    "discount_rate",
    "terminal_growth",
    "target_percentile",
    "fallback_growth",
    "verdict_band",
    "max_growth",
    "min_dividend_yield",
    "peg_factor",
    "min_peg_growth",
    "target_band",
    "metric_outlier_factor",
    "history_weight",
    "history_cap",
    "growth_leader_threshold",
    "restructuring_pb_ratio",
    "restructuring_ps_ratio",
    "restructuring_max_eps",
    "restructuring_min_bps",
    "restructuring_min_sps",
    "ps_peer_divergence_factor",
    "cyclical_boom_pe_ratio",
    "leader_history_weight",
    "leader_history_cap_factor",
    "peer_own_premium_factor",
    "model_min_confidence",
    "model_anchor_weight",
    "leader_primary_min_weight",
    "growth_cagr_weight",
    "ddm_growth_cap",
}
_BOOL_KEYS = {"model_enabled", "skip_llm_peer_validation"}
_STR_KEYS = {
    "config_version",
    "model_models_dir",
    "disclaimer",
    "combine_mode",
}
_LIST_KEYS = {
    "sensitivity_rates",
    "outlier_band",
    "peer_industry_band",
    "cyclical_keywords",
    "peer_mismatch_keywords",
    "manual_peers",
    "peg_band",
}
_DICT_KEYS = {"method_weights", "manual_fair_value"}

_cache: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _coerce(key: str, value: Any) -> Any:
    try:
        if key in _INT_KEYS:
            return int(value)
        if key in _FLOAT_KEYS:
            number = float(value)
            return number if number == number and abs(number) != float("inf") else None
        if key in _BOOL_KEYS:
            if isinstance(value, bool):
                return value
            return str(value).strip().lower() in ("1", "true", "yes", "on")
        if key in _STR_KEYS:
            return str(value) if value is not None else None
        if key in _LIST_KEYS:
            return list(value) if isinstance(value, list) else None
        if key in _DICT_KEYS:
            return dict(value) if isinstance(value, dict) else None
    except (TypeError, ValueError):
        return None
    return value


def _read(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法读取美股估值配置 %s: %s", path, exc)
        return {}
    if not isinstance(raw, dict):
        logger.warning("美股估值配置 %s 不是 JSON 对象，已忽略", path)
        return {}
    cleaned: dict[str, Any] = {}
    for key, value in raw.items():
        coerced = _coerce(key, value)
        if coerced is None and value is not None:
            logger.warning("美股估值配置 %s 的键 %s 类型非法，已回退默认值", path, key)
            continue
        cleaned[key] = coerced
    return cleaned


def load_us_valuation_config(
    ticker: str | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Return (merged_cfg, source_name, overridden_keys) for a US ticker."""
    with _lock:
        if "default" not in _cache:
            _cache["default"] = _read(DEFAULT_CONFIG_PATH)
        defaults = dict(_cache["default"])
    if not ticker:
        return defaults, "us_default.json", []
    key = ticker.upper()
    with _lock:
        if key in _cache:
            cached = _cache[key]
            return dict(cached["cfg"]), cached["source"], list(cached["overrides"])
    path = SYMBOL_DIR / f"{key}.json"
    symbol_cfg = _read(path)
    overrides = [item for item in symbol_cfg if item != "config_version"]
    if not symbol_cfg:
        merged, source = defaults, "us_default.json"
    else:
        merged = {**defaults, **symbol_cfg}
        source = f"us/{path.name}"
    with _lock:
        _cache[key] = {"cfg": dict(merged), "source": source, "overrides": list(overrides)}
    return dict(merged), source, list(overrides)


def us_manual_peers(ticker: str) -> tuple[list[Any], bool]:
    """Return (peers, present) for a US ticker from JSON config only."""
    cfg, _, overrides = load_us_valuation_config(ticker)
    present = "manual_peers" in overrides or "manual_peers" in cfg
    return list(cfg.get("manual_peers") or []), present


def clear_us_config_cache() -> None:
    with _lock:
        _cache.clear()
