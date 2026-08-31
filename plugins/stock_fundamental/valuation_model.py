"""Local LightGBM valuation-multiple models (train + inference).

Three small LightGBM regressors predict fair PE-TTM / PB / PS-TTM target
multiples from a cross-section of A-share fundamentals (latest report period).
Artifacts live under ``state/valuation_model/`` (gitignored) and are built by
``python -m main train-valuation-model``.  Inference is pure CPU and takes
milliseconds, so the models are loaded lazily once and cached in memory.
"""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from plugins.stock_common import disable_http_proxy


logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS_DIR = ROOT / "state" / "valuation_model"

FEATURE_COLUMNS = [
    "eps",
    "roe",
    "bps",
    "ocf_ps",
    "gross_margin",
    "revenue_yoy",
    "profit_yoy",
    "log_mktcap",
    "industry",
]
MODEL_TARGETS = ("pe", "pb", "ps")
MODEL_FILES = {
    "pe": "pe_model.joblib",
    "pb": "pb_model.joblib",
    "ps": "ps_model.joblib",
}

_YJBB_COLUMNS = {
    "每股收益": "eps",
    "净资产收益率": "roe_pct",
    "每股净资产": "bps",
    "每股经营现金流量": "ocf_ps",
    "销售毛利率": "gross_margin_pct",
    "营业总收入-同比增长": "revenue_yoy_pct",
    "净利润-同比增长": "profit_yoy_pct",
    "营业总收入-营业总收入": "revenue",
    "所处行业": "industry",
}
_LGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 30,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
}

_MODEL_CACHE: dict[tuple[Path, str], tuple[Any, dict[str, Any]]] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def _num(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _resolve_models_dir(models_dir: str | Path | None) -> Path:
    if models_dir is None:
        return DEFAULT_MODELS_DIR
    path = Path(models_dir)
    if not path.is_absolute():
        path = ROOT / path
    return path


def _recent_report_dates(limit: int = 8) -> list[str]:
    """Quarter-end report dates (YYYYMMDD) from the current period backwards."""
    ends = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
    today = date.today()
    quarter_index = (today.month - 1) // 3
    year = today.year
    dates: list[str] = []
    for _ in range(limit):
        month, day = ends[quarter_index + 1]
        dates.append(f"{year}{month:02d}{day:02d}")
        quarter_index -= 1
        if quarter_index < 0:
            quarter_index = 3
            year -= 1
    return dates


def _annualize_factor(report_date: str) -> float:
    month = int(report_date[4:6])
    if month == 3:
        return 4.0
    if month == 6:
        return 2.0
    if month == 9:
        return 4.0 / 3.0
    return 1.0


def _fetch_with_retry(
    func,
    *args,
    retries: int = 3,
    **kwargs,
):
    """Call an AkShare fetch with proxy bypass and bounded retries."""
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with disable_http_proxy():
                return func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                time.sleep(0.8 * (2**attempt))
    assert last_error is not None
    raise last_error


def _fetch_yjbb(report_date: str) -> pd.DataFrame:
    import akshare as ak

    try:
        frame = _fetch_with_retry(ak.stock_yjbb_em, date=report_date)
    except Exception as exc:  # noqa: BLE001
        logger.warning("业绩报表 %s 获取失败（%s），视为无数据", report_date, exc)
        return pd.DataFrame()
    if frame is None or frame.empty:
        return pd.DataFrame()
    return frame


def _normalize_spot(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize East Money / Tencent spot snapshots to a common schema."""
    out = pd.DataFrame()
    if "代码" in frame.columns and "最新价" in frame.columns:
        # East Money: 总市值 in 元
        out["symbol"] = (
            frame["代码"].map(lambda value: str(value).split(".")[0]).str.zfill(6)
        )
        out["close"] = pd.to_numeric(frame["最新价"], errors="coerce")
        out["pe_ttm"] = pd.to_numeric(frame.get("市盈率-动态"), errors="coerce")
        out["pb"] = pd.to_numeric(frame.get("市净率"), errors="coerce")
        out["total_market_cap"] = pd.to_numeric(
            frame.get("总市值"), errors="coerce"
        )
    elif "code" in frame.columns and "zxj" in frame.columns:
        # Tencent: zsz in 亿元 -> 元
        out["symbol"] = (
            frame["code"]
            .map(lambda value: str(value)[2:].split(".")[0])
            .str.zfill(6)
        )
        out["close"] = pd.to_numeric(frame["zxj"], errors="coerce")
        out["pe_ttm"] = pd.to_numeric(frame.get("pe_ttm"), errors="coerce")
        out["pb"] = pd.to_numeric(frame.get("pn"), errors="coerce")
        out["total_market_cap"] = (
            pd.to_numeric(frame.get("zsz"), errors="coerce") * 1e8
        )
    else:
        raise RuntimeError("无法识别的行情快照列结构")
    return out


def _fetch_spot() -> pd.DataFrame:
    import akshare as ak

    frame = pd.DataFrame()
    try:
        frame = _fetch_with_retry(ak.stock_zh_a_spot_em)
    except Exception as exc:  # noqa: BLE001
        logger.warning("东财全市场行情获取失败（%s），改用腾讯行情兜底", exc)
    if frame is None or frame.empty:
        frame = _fetch_with_retry(ak.stock_zh_a_spot_tx)
    if frame is None or frame.empty:
        return pd.DataFrame()
    return _normalize_spot(frame)


def build_dataset(
    yjbb: pd.DataFrame,
    spot: pd.DataFrame,
    report_date: str,
) -> pd.DataFrame:
    """Merge the latest report-period cross-section with current multiples."""
    renamed = yjbb.rename(columns=_YJBB_COLUMNS)
    if "股票代码" not in renamed.columns:
        raise RuntimeError("业绩报表缺少股票代码列")
    financial = renamed[
        ["股票代码"]
        + [column for column in _YJBB_COLUMNS.values() if column in renamed.columns]
    ].copy()
    financial["symbol"] = (
        financial["股票代码"].map(lambda value: str(value).split(".")[0]).str.zfill(6)
    )
    for column in (
        "roe_pct",
        "gross_margin_pct",
        "revenue_yoy_pct",
        "profit_yoy_pct",
    ):
        if column in financial.columns:
            financial[column] = pd.to_numeric(financial[column], errors="coerce") / 100.0
    for column in ("eps", "bps", "ocf_ps", "revenue"):
        if column in financial.columns:
            financial[column] = pd.to_numeric(financial[column], errors="coerce")

    spot_rows = spot.copy()
    if spot_rows.empty or "symbol" not in spot_rows.columns:
        raise RuntimeError("行情快照缺少代码/估值列")
    spot_rows["symbol"] = (
        spot_rows["symbol"].map(lambda value: str(value).split(".")[0]).str.zfill(6)
    )

    merged = financial.merge(spot_rows, on="symbol", how="inner")
    merged["report_date"] = report_date
    annualize = _annualize_factor(report_date)
    if "revenue" in merged.columns:
        merged["revenue_annualized"] = merged["revenue"].astype(float) * annualize
    if "total_market_cap" in merged.columns:
        merged["log_mktcap"] = np.log(merged["total_market_cap"].astype(float))
    if "revenue" in merged.columns and "total_market_cap" in merged.columns:
        merged["ps_ttm"] = (
            merged["total_market_cap"].astype(float) / merged["revenue_annualized"]
        )
    for source, target in (
        ("roe_pct", "roe"),
        ("gross_margin_pct", "gross_margin"),
        ("revenue_yoy_pct", "revenue_yoy"),
        ("profit_yoy_pct", "profit_yoy"),
    ):
        if source in merged.columns:
            merged[target] = merged[source]
    merged["industry"] = (
        merged["industry"].astype(str)
        if "industry" in merged.columns
        else ""
    )
    keep = ["symbol", "report_date"] + FEATURE_COLUMNS + [
        "pe_ttm",
        "pb",
        "ps_ttm",
    ]
    return merged[[column for column in keep if column in merged.columns]].copy()


def _train_target(
    dataset: pd.DataFrame,
    target: str,
    report_date: str,
) -> tuple[Any, dict[str, Any]] | None:
    from lightgbm import LGBMRegressor

    label = {"pe": "pe_ttm", "pb": "pb", "ps": "ps_ttm"}[target]
    if label not in dataset.columns:
        return None
    numeric_features = [
        column for column in FEATURE_COLUMNS
        if column != "industry" and column in dataset.columns
    ]
    rows = dataset.dropna(subset=[label]).copy()
    if target == "pe":
        if "eps" in rows.columns:
            rows = rows[rows["eps"] > 0]
        rows = rows[(rows[label] > 0) & (rows[label] < 500)]
    elif target == "pb":
        if "bps" in rows.columns:
            rows = rows[rows["bps"] > 0]
        rows = rows[(rows[label] > 0) & (rows[label] < 200)]
    else:
        if "revenue" in rows.columns:
            rows = rows[rows["revenue"] > 0]
        rows = rows[(rows[label] > 0) & (rows[label] < 100)]
    rows = rows.replace([np.inf, -np.inf], np.nan)
    rows = rows.dropna(subset=numeric_features)
    if len(rows) < 200:
        logger.warning("%s 模型样本不足（%d），跳过训练", target, len(rows))
        return None
    features = FEATURE_COLUMNS.copy()
    params = dict(_LGBM_PARAMS)
    industry_categories: list[Any] | None = None
    if "industry" in rows.columns and rows["industry"].nunique() >= 2:
        rows["industry"] = rows["industry"].astype("category")
        industry_categories = list(rows["industry"].cat.categories)
    else:
        features = [column for column in features if column != "industry"]
    features = [column for column in features if column in rows.columns]
    if not features:
        logger.warning("%s 模型没有可用特征，跳过训练", target)
        return None
    y = np.log(rows[label].astype(float))
    model = LGBMRegressor(**params)
    model.fit(rows[features], y)
    predicted = np.exp(model.predict(rows[features]))
    actual = rows[label].astype(float).to_numpy()
    mape = float(np.mean(np.abs(predicted - actual) / np.maximum(actual, 1e-9)))
    mse = float(np.mean((np.log(predicted) - np.log(actual)) ** 2))
    return model, {
        "n_samples": int(len(rows)),
        "mape": round(mape, 4),
        "mse": round(mse, 4),
        "report_date": report_date,
        "features": features,
        "industry_categories": industry_categories,
    }


def train(
    models_dir: str | Path | None = None,
    report_date: str | None = None,
) -> dict[str, Any]:
    """Fetch the latest cross-section, train three models, and save artifacts."""
    target_dir = _resolve_models_dir(models_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if report_date is None:
        selected_date: str | None = None
        yjbb = pd.DataFrame()
        today = date.today().strftime("%Y%m%d")
        for candidate in _recent_report_dates():
            if candidate > today:
                continue
            frame = _fetch_yjbb(candidate)
            if frame is not None and not frame.empty:
                selected_date = candidate
                yjbb = frame
                break
        if selected_date is None:
            raise RuntimeError("无法获取任何最近报告期的全市场财务数据")
    else:
        selected_date = report_date
        yjbb = _fetch_yjbb(report_date)
        if yjbb is None or yjbb.empty:
            raise RuntimeError(f"报告期 {report_date} 无全市场财务数据")

    try:
        spot = _fetch_spot()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法获取全市场实时行情快照: {exc}") from exc
    if spot is None or spot.empty:
        raise RuntimeError("无法获取全市场实时行情快照")
    dataset = build_dataset(yjbb, spot, selected_date)
    if dataset.empty:
        raise RuntimeError("训练数据集为空，无法训练")

    trained: dict[str, Any] = {}
    for target in MODEL_TARGETS:
        result = _train_target(dataset, target, selected_date)
        if result is None:
            continue
        model, stats = result
        joblib.dump(model, target_dir / MODEL_FILES[target])
        trained[target] = stats
    if not trained:
        raise RuntimeError("所有目标倍数模型均因样本不足而跳过，未生成任何模型")

    dataset.to_csv(target_dir / "training.csv", index=False)
    meta = {
        "version": "1.0",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "report_date": selected_date,
        "features": FEATURE_COLUMNS,
        "models": trained,
    }
    (target_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "models_dir": str(target_dir),
        "report_date": selected_date,
        "models": trained,
    }


def _load_models(
    models_dir: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load (models, meta), raising FileNotFoundError when artifacts are absent."""
    target_dir = _resolve_models_dir(models_dir)
    meta_path = target_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"估值模型未训练（{meta_path} 不存在），请先运行 python -m main train-valuation-model"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get((target_dir, meta.get("version", "")))
        if cached is not None:
            return cached
    models: dict[str, Any] = {}
    for target in MODEL_TARGETS:
        model_path = target_dir / MODEL_FILES[target]
        if model_path.exists():
            models[target] = joblib.load(model_path)
    if not models:
        raise FileNotFoundError(f"{target_dir} 下没有可用的模型文件")
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE[(target_dir, meta.get("version", ""))] = (models, meta)
    return models, meta


def _features_from_metrics(
    metrics: dict[str, Any],
) -> tuple[pd.DataFrame, float, list[str]]:
    """Assemble the model feature row from fundamental metrics."""
    total_shares = _num(metrics.get("total_shares"))
    current_price = _num(metrics.get("current_price"))
    ocf_ttm = _num(metrics.get("ocf_ttm"))
    market_cap = (
        current_price * total_shares
        if current_price is not None and total_shares is not None and current_price > 0 and total_shares > 0
        else None
    )
    industry = str(metrics.get("industry_name") or "").strip()
    row: dict[str, Any] = {
        "eps": _num(metrics.get("eps_ttm")),
        "roe": _num(metrics.get("roe")),
        "bps": _num(metrics.get("bps")),
        "ocf_ps": (
            ocf_ttm / total_shares
            if ocf_ttm is not None and total_shares is not None and total_shares > 0
            else None
        ),
        "gross_margin": _num(metrics.get("gross_margin")),
        "revenue_yoy": _num(metrics.get("revenue_growth_yoy")),
        "profit_yoy": _num(metrics.get("net_profit_yoy")),
        "log_mktcap": math.log(market_cap) if market_cap is not None and market_cap > 0 else None,
        "industry": industry,
    }
    present = [
        column
        for column in FEATURE_COLUMNS
        if (column == "industry" and row["industry"])
        or (column != "industry" and row[column] is not None)
    ]
    confidence = round(len(present) / len(FEATURE_COLUMNS), 4)
    frame = pd.DataFrame([row])
    for column in FEATURE_COLUMNS:
        if column != "industry":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame, confidence, present


def model_targets_from_metrics(
    metrics: dict[str, Any],
    symbol: str = "",
    models_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Predict fair PE/PB/PS target multiples for a single stock's metrics."""
    try:
        models, meta = _load_models(models_dir)
    except FileNotFoundError as exc:
        return {
            "available": False,
            "reason": str(exc),
            "symbol": symbol,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "reason": f"估值模型加载失败: {exc}",
            "symbol": symbol,
        }

    row_df, confidence, present = _features_from_metrics(metrics)
    result: dict[str, Any] = {
        "available": True,
        "symbol": symbol,
        "confidence": confidence,
        "model_version": meta.get("version", "?"),
        "trained_on": meta.get("trained_at"),
        "report_date": meta.get("report_date"),
        "features_used": present,
        "warnings": [],
    }
    for target in MODEL_TARGETS:
        model = models.get(target)
        if model is None:
            continue
        model_features = (
            (meta.get("models") or {}).get(target, {}).get("features")
            or FEATURE_COLUMNS
        )
        model_features = [
            column for column in model_features if column in row_df.columns
        ]
        if not model_features:
            result["warnings"].append(f"{target} 无可用特征，跳过推理")
            continue
        predict_frame = row_df[model_features]
        if "industry" in model_features:
            predict_frame = predict_frame.copy()
            categories = (
                (meta.get("models") or {}).get(target, {}).get("industry_categories")
            )
            industry_value = str(row_df.iloc[0].get("industry") or "")
            if categories:
                if industry_value not in categories:
                    industry_value = np.nan
                predict_frame["industry"] = pd.Categorical(
                    [industry_value],
                    categories=list(categories),
                )
            else:
                predict_frame["industry"] = predict_frame["industry"].astype("category")
        try:
            value = float(np.exp(model.predict(predict_frame)[0]))
        except Exception as exc:  # noqa: BLE001
            result["warnings"].append(f"{target} 推理失败: {exc}")
            continue
        if math.isfinite(value) and 0 < value < 1e5:
            result[target] = round(value, 4)
    return result
