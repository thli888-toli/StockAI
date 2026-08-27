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
    "atr14",
    "bollinger_bandwidth",
    "bollinger_pctb",
    "close_high20_ratio",
    "close_low20_ratio",
    "amount_ratio",
]

LIGHTGBM_PARAMS = {
    "num_leaves": 31,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "min_child_samples": 20,
    "objective": "binary",
    "random_state": 42,
    "verbosity": -1,
}

DAILY_SEQ_LEN = 15
DAILY_MIN_TRAIN = 120
DAILY_VALIDATION = 180
DAILY_EPOCHS = 10

WEEKLY_SEQ_LEN = 10
WEEKLY_MIN_TRAIN = 60
WEEKLY_VALIDATION = 30
WEEKLY_EPOCHS = 8

MONTHLY_MIN_BARS = 20
MONTHLY_MIN_TRAIN = 60
MONTHLY_VALIDATION = 30
MONTHLY_SEQ_LEN = 10
MONTHLY_EPOCHS = 8
QUANT_BUILD_TIMEOUT_SECONDS = 900.0


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
    if "close" not in df.columns:
        df["close"] = 0.0
    numeric_columns = FEATURE_COLUMNS + ["close"]
    existing = [column for column in numeric_columns if column in df.columns]
    df[existing] = df[existing].apply(pd.to_numeric, errors="coerce")
    return df.reset_index(drop=True)


def _usable_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return feature columns with enough valid values (>=50%) to train on."""
    result: list[str] = []
    for column in FEATURE_COLUMNS:
        if column not in df.columns:
            continue
        if df[column].notna().mean() >= 0.5:
            result.append(column)
    return result


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
    existing = [column for column in columns if column in df.columns]
    df = df[existing].copy()
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


def _new_model():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(**LIGHTGBM_PARAMS)


def _lstm_probability(
    df: pd.DataFrame,
    target_column: str,
    min_train: int,
    seq_len: int,
    epochs: int,
) -> float:
    usable = df.dropna(subset=[target_column])
    if len(usable) < min_train:
        return 0.5
    features = _usable_feature_columns(df)
    if not features:
        return 0.5
    features_all = df[features].fillna(0.0).to_numpy(dtype=np.float64)
    usable_features = usable[features].fillna(0.0).to_numpy(dtype=np.float64)
    usable_labels = usable[target_column].to_numpy(dtype=np.float64)
    standardized, mean, std = lstm_model.standardize(features_all)
    usable_standardized = (usable_features - mean) / std
    return float(
        lstm_model.predict_latest(
            standardized,
            usable_standardized,
            usable_labels,
            seq_len=seq_len,
            epochs=epochs,
        )
    )


def _lightgbm_probability(
    df: pd.DataFrame,
    target_column: str,
    min_train: int,
) -> float:
    usable = df.dropna(subset=[target_column])
    if len(usable) < min_train:
        return 0.5
    features = _usable_feature_columns(df)
    if not features:
        return 0.5
    model = _new_model()
    model.fit(usable[features], usable[target_column])
    latest = df.iloc[[-1]][features].fillna(0.0)
    return float(model.predict_proba(latest)[0, 1])


def _lstm_walk_forward(
    df: pd.DataFrame,
    target_column: str,
    min_train: int,
    validation_size: int,
    seq_len: int,
    epochs: int,
) -> tuple[float, int, list[float], list[float]]:
    from sklearn.metrics import roc_auc_score

    usable = df.dropna(subset=[target_column])
    if len(usable) < min_train:
        return 0.0, len(usable), [], []

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    end = min_train
    while end < len(usable):
        val_end = min(end + validation_size, len(usable))
        train_x, train_y = _lstm_sequences(usable, 0, end, seq_len, target_column)
        val_x, val_y = _lstm_sequences(usable, end, val_end, seq_len, target_column)
        if len(train_x) == 0 or len(val_x) == 0:
            end = val_end
            continue
        if len(set(train_y.tolist())) < 2 or len(set(val_y.tolist())) < 2:
            end = val_end
            continue
        model = lstm_model._train_model(train_x, train_y, epochs)
        predictions = lstm_model.predict_proba(model, val_x)
        y_true_all.extend(val_y.tolist())
        y_pred_all.extend(predictions.tolist())
        end = val_end

    if not y_true_all or len(set(y_true_all)) < 2:
        return 0.0, len(usable), y_true_all, y_pred_all
    auc = float(roc_auc_score(y_true_all, y_pred_all))
    return round(auc, 6), len(usable), y_true_all, y_pred_all


def _lstm_sequences(
    usable: pd.DataFrame,
    start: int,
    end: int,
    seq_len: int,
    target_column: str,
):
    feature_columns = _usable_feature_columns(usable)
    features = usable.iloc[start:end][feature_columns].fillna(0.0).to_numpy(dtype=np.float64)
    labels = usable.iloc[start:end][target_column].to_numpy(dtype=np.float64)
    standardized, _, _ = lstm_model.standardize(features)
    return lstm_model.build_sequences(standardized, labels, seq_len)


def _lightgbm_walk_forward(
    df: pd.DataFrame,
    target_column: str,
    min_train: int,
    validation_size: int,
) -> tuple[float, int, list[float], list[float]]:
    from sklearn.metrics import roc_auc_score

    usable = df.dropna(subset=[target_column])
    if len(usable) < min_train:
        return 0.0, len(usable), [], []
    features = _usable_feature_columns(df)
    if not features:
        return 0.0, len(usable), [], []

    y_true_all: list[float] = []
    y_pred_all: list[float] = []
    end = min_train
    while end < len(usable):
        val_end = min(end + validation_size, len(usable))
        train = usable.iloc[:end]
        val = usable.iloc[end:val_end]
        if val.empty:
            break
        if train[target_column].nunique() < 2 or val[target_column].nunique() < 2:
            end = val_end
            continue
        model = _new_model()
        model.fit(train[features], train[target_column])
        y_true_all.extend(val[target_column].tolist())
        y_pred_all.extend(model.predict_proba(val[features])[:, 1].tolist())
        end = val_end

    if not y_true_all or len(set(y_true_all)) < 2:
        return 0.0, len(usable), y_true_all, y_pred_all
    auc = float(roc_auc_score(y_true_all, y_pred_all))
    return round(auc, 6), len(usable), y_true_all, y_pred_all


def _fit_calibration(
    y_true: list[float],
    y_pred: list[float],
    bins: int = 10,
    prior: float = 5.0,
) -> pd.DataFrame | None:
    if len(y_true) < 30 or len(set(y_true)) < 2:
        return None
    frame = pd.DataFrame(
        {
            "y": [float(value) for value in y_true],
            "p": [float(value) for value in y_pred],
        }
    )
    try:
        frame["bin"] = pd.qcut(
            frame["p"],
            q=min(bins, max(2, frame["p"].nunique())),
            duplicates="drop",
        )
    except Exception:
        return None
    mapping = (
        frame.groupby("bin", observed=True)
        .agg(p_mean=("p", "mean"), y_mean=("y", "mean"), count=("y", "size"))
        .reset_index(drop=True)
        .sort_values("p_mean")
    )
    mapping["calibrated"] = (mapping["count"] * mapping["y_mean"] + prior * 0.5) / (
        mapping["count"] + prior
    )
    return mapping


def _apply_calibration(probability: float, mapping: pd.DataFrame | None) -> float:
    if mapping is None or mapping.empty:
        return probability
    index = (mapping["p_mean"] - probability).abs().idxmin()
    return float(mapping.loc[index, "calibrated"])


def _ensemble_signal(
    lgb_probability: float,
    lstm_probability: float,
    lgb_auc: float,
    lstm_auc: float,
    lgb_calibration: pd.DataFrame | None = None,
    lstm_calibration: pd.DataFrame | None = None,
) -> dict[str, Any]:
    lgb_weight = max(0.0, lgb_auc - 0.5)
    lstm_weight = max(0.0, lstm_auc - 0.5)
    total_weight = lgb_weight + lstm_weight
    if total_weight <= 0:
        return _neutral_horizon()

    lgb_probability = (
        _apply_calibration(lgb_probability, lgb_calibration)
        if lgb_auc > 0.5
        else lgb_probability
    )
    lstm_probability = (
        _apply_calibration(lstm_probability, lstm_calibration)
        if lstm_auc > 0.5
        else lstm_probability
    )
    probability = (
        lgb_probability * lgb_weight + lstm_probability * lstm_weight
    ) / total_weight
    return {
        "up_probability": round(probability, 6),
        "direction": direction_from_probability(probability),
        "confidence": confidence_from_probability(probability),
    }


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


def _build_quant_payload(
    symbol: str,
    records: list[dict[str, Any]],
    as_of: str,
    monthly_records: list[dict[str, Any]] | None = None,
) -> str:
    df = _prepare_dataset(records)
    if len(df) < 120:
        payload = {
            "symbol": symbol,
            "as_of": as_of,
            "model": "LSTM+LightGBM",
            "horizons": {key: _neutral_horizon() for key in ("5d", "15d", "1w", "1mo")},
            "backtest": {"walk_forward_auc": None, "sample_count": len(df)},
            "weekly_backtest": {"walk_forward_auc": None, "sample_count": 0},
            "warnings": ["insufficient daily history for LSTM+LightGBM training"],
        }
        return json_dumps(payload)

    df = _add_daily_targets(df)
    horizons: dict[str, Any] = {}
    daily_aucs: list[float] = []
    warnings: list[str] = []

    for horizon in DAILY_HORIZONS:
        target = f"target_{horizon}d"
        lgb_probability = _lightgbm_probability(df, target, DAILY_MIN_TRAIN)
        lstm_probability = _lstm_probability(
            df, target, DAILY_MIN_TRAIN, DAILY_SEQ_LEN, DAILY_EPOCHS
        )
        lgb_auc, lgb_n, lgb_y_true, lgb_y_pred = _lightgbm_walk_forward(
            df, target, DAILY_MIN_TRAIN, DAILY_VALIDATION
        )
        lstm_auc, lstm_n, lstm_y_true, lstm_y_pred = _lstm_walk_forward(
            df, target, DAILY_MIN_TRAIN, DAILY_VALIDATION, DAILY_SEQ_LEN, DAILY_EPOCHS
        )
        lgb_calibration = _fit_calibration(lgb_y_true, lgb_y_pred)
        lstm_calibration = _fit_calibration(lstm_y_true, lstm_y_pred)
        horizons[f"{horizon}d"] = _ensemble_signal(
            lgb_probability,
            lstm_probability,
            lgb_auc,
            lstm_auc,
            lgb_calibration,
            lstm_calibration,
        )
        daily_aucs.append(round((lgb_auc + lstm_auc) / 2, 6))
        if lgb_n < DAILY_MIN_TRAIN or lstm_n < DAILY_MIN_TRAIN:
            warnings.append(f"{horizon}d horizon has fewer than {DAILY_MIN_TRAIN} usable rows")

    daily_auc = round(float(sum(daily_aucs) / len(daily_aucs)), 6)

    weekly = _period_features(records, "W-FRI")
    if len(weekly) >= WEEKLY_MIN_TRAIN:
        weekly = _add_shift_target(weekly, "target_1w", 1)
        lgb_probability = _lightgbm_probability(weekly, "target_1w", WEEKLY_MIN_TRAIN)
        lstm_probability = _lstm_probability(
            weekly, "target_1w", WEEKLY_MIN_TRAIN, WEEKLY_SEQ_LEN, WEEKLY_EPOCHS
        )
        lgb_auc, _, lgb_y_true, lgb_y_pred = _lightgbm_walk_forward(
            weekly, "target_1w", WEEKLY_MIN_TRAIN, WEEKLY_VALIDATION
        )
        lstm_auc, weekly_samples, lstm_y_true, lstm_y_pred = _lstm_walk_forward(
            weekly, "target_1w", WEEKLY_MIN_TRAIN, WEEKLY_VALIDATION, WEEKLY_SEQ_LEN, WEEKLY_EPOCHS
        )
        lgb_calibration = _fit_calibration(lgb_y_true, lgb_y_pred)
        lstm_calibration = _fit_calibration(lstm_y_true, lstm_y_pred)
        horizons["1w"] = _ensemble_signal(
            lgb_probability,
            lstm_probability,
            lgb_auc,
            lstm_auc,
            lgb_calibration,
            lstm_calibration,
        )
        weekly_backtest = {
            "walk_forward_auc": round((lgb_auc + lstm_auc) / 2, 6),
            "sample_count": int(len(weekly)),
        }
    else:
        horizons["1w"] = _neutral_horizon()
        weekly_backtest = {"walk_forward_auc": None, "sample_count": int(len(weekly))}
        warnings.append("insufficient weekly history for LSTM+LightGBM training")

    if monthly_records:
        monthly = _records_to_ohlcv(monthly_records)
    else:
        monthly = _period_ohlcv(records, "ME")
    monthly_features = prepare_daily_features(monthly)
    if not monthly_features.empty:
        monthly_features = _add_shift_target(monthly_features, "target_1mo", 1)
    monthly_usable = (
        len(monthly_features.dropna(subset=["target_1mo"]))
        if "target_1mo" in monthly_features.columns
        else 0
    )
    if monthly_usable >= MONTHLY_MIN_TRAIN:
        lgb_probability = _lightgbm_probability(
            monthly_features, "target_1mo", MONTHLY_MIN_TRAIN
        )
        lstm_probability = _lstm_probability(
            monthly_features,
            "target_1mo",
            MONTHLY_MIN_TRAIN,
            MONTHLY_SEQ_LEN,
            MONTHLY_EPOCHS,
        )
        lgb_auc, _, lgb_y_true, lgb_y_pred = _lightgbm_walk_forward(
            monthly_features, "target_1mo", MONTHLY_MIN_TRAIN, MONTHLY_VALIDATION
        )
        lstm_auc, monthly_samples, lstm_y_true, lstm_y_pred = _lstm_walk_forward(
            monthly_features,
            "target_1mo",
            MONTHLY_MIN_TRAIN,
            MONTHLY_VALIDATION,
            MONTHLY_SEQ_LEN,
            MONTHLY_EPOCHS,
        )
        lgb_calibration = _fit_calibration(lgb_y_true, lgb_y_pred)
        lstm_calibration = _fit_calibration(lstm_y_true, lstm_y_pred)
        horizons["1mo"] = _ensemble_signal(
            lgb_probability,
            lstm_probability,
            lgb_auc,
            lstm_auc,
            lgb_calibration,
            lstm_calibration,
        )
        monthly_backtest = {
            "walk_forward_auc": round((lgb_auc + lstm_auc) / 2, 6),
            "sample_count": int(len(monthly_features)),
        }
    else:
        monthly_signal, monthly_samples = _monthly_signal(monthly)
        horizons["1mo"] = monthly_signal
        monthly_backtest = {
            "walk_forward_auc": None,
            "sample_count": int(len(monthly)),
        }
        if monthly_samples < MONTHLY_MIN_BARS:
            warnings.append("insufficient monthly history for technical trend")
        warnings.append("insufficient monthly history for LSTM+LightGBM training")

    payload = {
        "symbol": symbol,
        "as_of": as_of,
        "model": "LSTM+LightGBM",
        "horizons": horizons,
        "backtest": {"walk_forward_auc": daily_auc, "sample_count": int(len(df))},
        "weekly_backtest": weekly_backtest,
        "monthly_backtest": monthly_backtest,
        "warnings": warnings,
    }
    return json_dumps(payload)


class StockQuantHandler:
    async def run(self, request: TaskRequest) -> str:
        symbol = validate_symbol(request.query)
        market_data = json_loads(request.inputs.get("market_data", ""), {})
        records = market_data.get("daily_features") or []
        monthly_records = market_data.get("monthly_history") or []
        as_of = str(market_data.get("as_of") or "")
        records_dump = json.dumps(records, sort_keys=True, default=str)
        monthly_dump = json.dumps(monthly_records, sort_keys=True, default=str)
        feature_hash = hashlib.sha1(
            f"{records_dump}|{monthly_dump}".encode("utf-8")
        ).hexdigest()
        cache_key = f"stock_quant:{symbol}:{as_of}:{feature_hash}"

        cached = QUANT_CACHE.get(cache_key)
        if cached:
            return cached
        cached = PERSISTENT_CACHE.get(symbol, feature_hash)
        if cached:
            QUANT_CACHE.set(cache_key, cached)
            return cached

        result = await run_blocking(
            _build_quant_payload,
            symbol,
            records,
            as_of,
            monthly_records,
            timeout=QUANT_BUILD_TIMEOUT_SECONDS,
            retries=0,
        )
        QUANT_CACHE.set(cache_key, result)
        PERSISTENT_CACHE.put(symbol, feature_hash, result)
        return result


def build_agent() -> StockQuantHandler:
    return StockQuantHandler()
