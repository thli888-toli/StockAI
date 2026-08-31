"""Valuation parameter configuration with per-symbol JSON overrides.

The base parameter set lives in ``config/valuation/default.json``.  A symbol
may ship its own ``config/valuation/{symbol}.json`` containing only the keys it
wants to override; missing keys fall back to the default file.  Unknown keys
are preserved so future parameters can be added without breaking existing
override files.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config" / "valuation"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "default.json"

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
_BOOL_KEYS = {"model_enabled"}
_STR_KEYS = {"config_version", "model_models_dir", "disclaimer"}
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


def _coerce_value(key: str, value: Any) -> Any:
    """Coerce a raw JSON value to the expected type; return None when invalid."""
    if key in _INT_KEYS:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
    if key in _FLOAT_KEYS:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number or number in (float("inf"), float("-inf")):
            return None
        return number
    if key in _BOOL_KEYS:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return None
    if key in _STR_KEYS:
        return str(value) if value is not None else None
    if key in _LIST_KEYS:
        if isinstance(value, list):
            return list(value)
        return None
    if key in _DICT_KEYS:
        if isinstance(value, dict):
            return dict(value)
        return None
    # Unknown key: preserved as-is for forward compatibility.
    return value


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("无法读取估值配置 %s: %s", path, exc)
        return {}
    if not isinstance(data, dict):
        logger.warning("估值配置 %s 不是 JSON 对象，已忽略", path)
        return {}
    cleaned: dict[str, Any] = {}
    for key, raw in data.items():
        coerced = _coerce_value(key, raw)
        if coerced is None and raw is not None:
            logger.warning("估值配置 %s 的键 %s 类型非法（%r），已回退默认值", path, key, raw)
            continue
        cleaned[key] = coerced
    return cleaned


def load_default_config() -> dict[str, Any]:
    """Load and cache the default valuation parameter set."""
    with _lock:
        if "default" not in _cache:
            _cache["default"] = _read_config(DEFAULT_CONFIG_PATH)
        return dict(_cache["default"])


def load_valuation_config(
    symbol: str | None = None,
) -> tuple[dict[str, Any], str, list[str]]:
    """Return ``(merged_cfg, config_source, overridden_keys)`` for a symbol.

    ``config_source`` is ``"default"`` when no per-symbol file exists, otherwise
    the per-symbol file name (e.g. ``600519.json``).  ``overridden_keys`` lists
    the keys explicitly set by the per-symbol file.
    """
    defaults = load_default_config()
    if not symbol:
        return defaults, "default", []
    cache_key = f"symbol:{symbol}"
    with _lock:
        if cache_key in _cache:
            cached = _cache[cache_key]
            return dict(cached["cfg"]), cached["source"], list(cached["overrides"])
    symbol_path = CONFIG_DIR / f"{symbol}.json"
    symbol_cfg = _read_config(symbol_path)
    overrides = [key for key in symbol_cfg if key != "config_version"]
    if not symbol_cfg:
        merged = defaults
        source = "default"
    else:
        merged = {**defaults, **symbol_cfg}
        source = symbol_path.name
    with _lock:
        _cache[cache_key] = {
            "cfg": dict(merged),
            "source": source,
            "overrides": list(overrides),
        }
    return dict(merged), source, list(overrides)


def clear_config_cache() -> None:
    """Drop cached configs (used by tests)."""
    with _lock:
        _cache.clear()


def symbol_manual_peers(symbol: str) -> list[Any]:
    """Manual peer list for a symbol from JSON config only.

    Returns ``[]`` both when the key is absent and when it is explicitly empty;
    callers decide fallback behaviour by checking ``symbol_manual_peers_present``.
    """
    cfg, _, overrides = load_valuation_config(symbol)
    if "manual_peers" in overrides:
        return list(cfg.get("manual_peers") or [])
    if "manual_peers" in cfg:
        return list(cfg.get("manual_peers") or [])
    return []


def symbol_manual_peers_present(symbol: str) -> bool:
    """True when the symbol's JSON config explicitly sets ``manual_peers``."""
    _, _, overrides = load_valuation_config(symbol)
    return "manual_peers" in overrides
