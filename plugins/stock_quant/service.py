from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np
import pandas as pd

from framework.schemas import TaskRequest
from plugins.stock_common import (
    TTLCache,
    compute_macd,
    confidence_from_probability,
    direction_from_probability,
    json_dumps,
    json_loads,
    prepare_daily_features,
    resample_ohlcv,
    run_blocking,
    validate_symbol,
)
from plugins.stock_quant import lstm_model
from plugins.stock_quant.quant_cache import QuantCacheStore


QUANT_CACHE = TTLCache(ttl_seconds=900)
PERSISTENT_CACHE = QuantCacheStore()

DAILY_HORIZONS = (5, 15)

FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_20d",
    "macd",
    "macd_signal",
    "macd_histogram",
    "rsi14",
    "volatility20",
    "volume_ratio",
    "close_ma20_ratio",
    "close_ma66_ratio",
    "close_ma154_ratio",
    "close_ma250_ratio",
    "turnover",
]

DAILY_SEQ_LEN = 15
DAILY_MIN_TRAIN = 120
DAILY_VALIDATION = 180
DAILY_EPOCHS = 10

WEEKLY_SEQ_LEN = 10
WEEKLY_MIN_TRAIN = 60
WEEKLY_VALIDATION = 30
WEEKLY_EPOCHS = 8

MONTHLY_MIN_BARS = 20


def _neutral_horizon() -> dict[str, Any]:
    return {
        "up_probability": 0.5,
        "direction": "flat",
        "confidence": 0.0,
    }


def _prepare_dataset(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty:
        return df
    df = df.copy()
    for column in FEATURE_COLUMNS + ["close"]:
        if column not in df.columns:
            df[column] = 0.0
    numeric_columns = FEATURE_COLUMNS + ["close"]
    df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df.reset_index(drop=True)


def _add_shift_target(df: pd.DataFrame, target_column: str, shift: int = 1) -> pd.DataFrame:
    df[target_column] = (df["close"].shift(-shift) > df["close"]).astype(float)
    return df


def _add_daily_targets(df: pd.DataFrame) -> pd.DataFrame:
    for horizon in DAILY_HORIZONS:
        df[f"target_{horizon}d"] = (df["close"].shift(-horizon) > df["close"]).astype(float)
    return df


def _records_to_ohlcv(records: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(records)
    if df.empty or not {"date", "open", "high", "low", "close", "volume"}.issubset(df.columns):
        return pd.DataFrame()
    columns = ["date", "open", "high", "low", "close", "volume"]
    for column in ("amount", "turnover"):
        columns.append(column)
    df = df[columns].copy()
    for column in ("amount", "turnover"):
        if column not in df.columns:
            df[column] = 0.0
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume", "amount", "turnover"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)


def _period_ohlcv(records: list[dict[str, Any]], freq: str) -> pd.DataFrame:
    daily = _records_to_ohlcv(records)
    if daily.empty:
        return daily
    return resample_ohlcv(daily, freq)


def _period_features(records: list[dict[str, Any]], freq: str) -> pd.DataFrame:
    frame = _period_ohlcv(records, freq)
    if frame.empty:
        return frame
    return prepare_daily_features(frame)


def _lstm_signal(
    df: pd.DataFrame,
    target_column: str,
    min_train: int,
    seq_len: int,
    epochs: int,
) -> dict[str, Any]:
    usable = df.dropna(subset=[target_column])
    if len(usable) < min_train:
        return _neutral_horizon()

    features_all = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    usable_features = usable[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    usable_labels = usable[target_column].to_numpy(dtype=np.float64)

    standardized, mean, std = lstm_model.standardize(features_all)
    usable_standardized = (usable_features - mean) / std
    probability = lstm_model.predict_latest(
        standardized,
        usable_standardized,
        usable_labels,
        seq_len=seq_len,
        epochs=epochs,
    )
    return {
        "up_probability": round(float(probability), 6),
        "direction": direction_from_probability(probability),
        "confidence": confidence_from_probability(probability),
    }


def _lstm_walk_forward(
    df: pd.DataFrame,
    target_column: str,
    min_train: int,
    validation_size: int,
    seq_len: int,
    epochs: int,
) -> tuple[float, int]:
    usable = df.dropna(subset=[target_column])
    if len(usable) < min_train:
        return 0.0, len(usable)
    features = usable[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    labels = usable[target_column].to_numpy(dtype=np.float64)
    standardized, _, _ = lstm_model.standardize(features)
    return lstm_model.walk_forward_auc(
        standardized,
        labels,
        min_train=min_train,
        validation_size=validation_size,
        seq_len=seq_len,
        epochs=epochs,
    )


def _monthly_signal(frame: pd.DataFrame) -> tuple[dict[str, Any], int]:
    if len(frame) < MONTHLY_MIN_BARS:
        return _neutral_horizon(), len(frame)
    close = frame["close"]
    macd = compute_macd(close, 12, 26, 9)
    ma20 = close.rolling(20).mean()
    histogram = macd["histogram"].iloc[-1]
    ma_value = ma20.iloc[-1]
    if pd.isna(histogram) or pd.isna(ma_value):
        return _neutral_horizon(), len(frame)
    macd_bullish = float(histogram) > 0
    ma_bullish = float(close.iloc[-1]) > float(ma_value)
    if macd_bullish and ma_bullish:
        return {"up_probability": 0.7, "direction": "up", "confidence": 0.4}, len(frame)
    if not macd_bullish and not ma_bullish:
        return {"up_probability": 0.3, "direction": "down", "confidence": 0.4}, len(frame)
    return _neutral_horizon(), len(frame)


def _build_quant_payload(symbol: str, records: list[dict[str, Any]], as_of: str) -> str:
    df = _prepare_dataset(records)
    if len(df) < 120:
        payload = {
            "symbol": symbol,
            "as_of": as_of,
            "model": "LSTM",
            "horizons": {key: _neutral_horizon() for key in ("5d", "15d", "1w", "1mo")},
            "backtest": {"walk_forward_auc": None, "sample_count": len(df)},
            "weekly_backtest": {"walk_forward_auc": None, "sample_count": 0},
            "warnings": ["insufficient daily history for LSTM training"],
        }
        return json_dumps(payload)

    df = _add_daily_targets(df)
    horizons: dict[str, Any] = {}
    daily_aucs: list[float] = []
    warnings: list[str] = []

    for horizon in DAILY_HORIZONS:
        target = f"target_{horizon}d"
        horizons[f"{horizon}d"] = _lstm_signal(
            df, target, DAILY_MIN_TRAIN, DAILY_SEQ_LEN, DAILY_EPOCHS
        )
        auc, sample_count = _lstm_walk_forward(
            df, target, DAILY_MIN_TRAIN, DAILY_VALIDATION, DAILY_SEQ_LEN, DAILY_EPOCHS
        )
        daily_aucs.append(auc)
        if sample_count < DAILY_MIN_TRAIN:
            warnings.append(f"{horizon}d horizon has fewer than {DAILY_MIN_TRAIN} usable rows")

    daily_auc = round(float(sum(daily_aucs) / len(daily_aucs)), 6)

    weekly = _period_features(records, "W-FRI")
    if len(weekly) >= WEEKLY_MIN_TRAIN:
        weekly = _add_shift_target(weekly, "target_1w", 1)
        horizons["1w"] = _lstm_signal(
            weekly, "target_1w", WEEKLY_MIN_TRAIN, WEEKLY_SEQ_LEN, WEEKLY_EPOCHS
        )
        weekly_auc, weekly_samples = _lstm_walk_forward(
            weekly,
            "target_1w",
            WEEKLY_MIN_TRAIN,
            WEEKLY_VALIDATION,
            WEEKLY_SEQ_LEN,
            WEEKLY_EPOCHS,
        )
        weekly_backtest = {
            "walk_forward_auc": weekly_auc,
            "sample_count": int(len(weekly)),
        }
    else:
        horizons["1w"] = _neutral_horizon()
        weekly_backtest = {"walk_forward_auc": None, "sample_count": int(len(weekly))}
        warnings.append("insufficient weekly history for LSTM training")

    monthly = _period_ohlcv(records, "ME")
    monthly_signal, monthly_samples = _monthly_signal(monthly)
    horizons["1mo"] = monthly_signal
    if monthly_samples < MONTHLY_MIN_BARS:
        warnings.append("insufficient monthly history for technical trend")

    payload = {
        "symbol": symbol,
        "as_of": as_of,
        "model": "LSTM",
        "horizons": horizons,
        "backtest": {"walk_forward_auc": daily_auc, "sample_count": int(len(df))},
        "weekly_backtest": weekly_backtest,
        "warnings": warnings,
    }
    return json_dumps(payload)


class StockQuantHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        records = market_data.get("daily_features") or []
        as_of = str(market_data.get("as_of") or "")
        records_dump = json.dumps(records, sort_keys=True, default=str)
        feature_hash = hashlib.sha1(records_dump.encode("utf-8")).hexdigest()
        cache_key = f"stock_quant:{symbol}:{as_of}:{feature_hash}"

        cached = QUANT_CACHE.get(cache_key)
        if cached:
            return cached
        cached = PERSISTENT_CACHE.get(symbol, feature_hash)
        if cached:
            QUANT_CACHE.set(cache_key, cached)
            return cached

        result = await run_blocking(_build_quant_payload, symbol, records, as_of, timeout=150.0, retries=0)
        QUANT_CACHE.set(cache_key, result)
        PERSISTENT_CACHE.put(symbol, feature_hash, result)
        return result


def build_agent() -> StockQuantHandler:
    return StockQuantHandler()
