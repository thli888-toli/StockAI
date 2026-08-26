"""Offline regression/backtest for the A-share analysis pipeline.

For each sampled historical date the tool simulates an analysis generated
"as of" that date (using only data up to that date), then compares the
predicted directions against the actual future price movement.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from scipy.stats import binomtest

from framework.config import STOCK_CACHE_DB
from plugins.stock_analyst.service import generate_llm_summary
from plugins.stock_cache import StockHistoryStore
from plugins.stock_common import normalize_akshare_frame, prepare_daily_features
from plugins.stock_quant.service import _build_quant_payload


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config" / "backtest.yaml"
RESULTS_DIR = ROOT / "state" / "backtest_results"

HORIZONS: list[tuple[str, int]] = [("5d", 5), ("15d", 15), ("1w", 5), ("1mo", 20)]
SUMMARY_HORIZON_DAYS = 20
MIN_LOOKBACK = 120
MIN_FUTURE = 20
MIN_GAP_TRADING_DAYS = 20


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return data or {}


def ensure_history(symbol: str) -> pd.DataFrame:
    store = StockHistoryStore(STOCK_CACHE_DB)
    frame = store.load(symbol, "qfq")
    if not frame.empty:
        return frame

    from plugins.stock_data.service import StockDataHandler

    end = date.today()
    start = end - timedelta(days=365 * 4)
    raw = StockDataHandler._fetch_daily(
        symbol,
        start.strftime("%Y%m%d"),
        end.strftime("%Y%m%d"),
    )
    normalized = normalize_akshare_frame(raw)
    if normalized.empty:
        raise RuntimeError(f"no history available for {symbol}")
    store.merge(symbol, "qfq", normalized)
    return store.load(symbol, "qfq")


def sample_dates(
    bars: pd.DataFrame,
    start: date,
    end: date,
    cadence: int,
    non_overlap: bool = False,
) -> pd.Series:
    mask = (bars["date"] >= pd.Timestamp(start)) & (bars["date"] <= pd.Timestamp(end))
    if not non_overlap:
        return bars.loc[mask, "date"].iloc[::cadence].reset_index(drop=True)
    min_gap = max(cadence, MIN_GAP_TRADING_DAYS)
    positions = bars.loc[mask].index.tolist()
    selected: list[int] = []
    last_index: int | None = None
    for index in positions:
        if last_index is None or (index - last_index) >= min_gap:
            selected.append(index)
            last_index = index
    return bars.loc[selected, "date"].reset_index(drop=True)


def evaluate_direction(pred: str, actual_up: bool) -> bool | None:
    if pred == "up":
        return actual_up
    if pred == "down":
        return not actual_up
    return None


def _market_data_from_bars(
    symbol: str,
    as_of: str,
    features: pd.DataFrame,
) -> dict[str, Any]:
    recent: list[dict[str, Any]] = []
    for _, row in features.tail(5).iterrows():
        recent.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": float(row["volume"]),
                "pct_change": float(row.get("pct_change", 0.0)),
            }
        )
    last = features.iloc[-1]
    indicator_keys = [
        "rsi14",
        "ma20",
        "ma66",
        "ma154",
        "ma250",
        "volatility20",
        "return_1d",
        "return_5d",
        "return_20d",
        "volume_ratio",
        "close_ma20_ratio",
        "close_ma66_ratio",
        "close_ma154_ratio",
        "close_ma250_ratio",
        "macd",
        "macd_signal",
        "macd_histogram",
    ]
    indicators = {
        key: (None if pd.isna(last.get(key)) else float(last[key]))
        for key in indicator_keys
        if key in features.columns
    }
    return {
        "symbol": symbol,
        "as_of": as_of,
        "latest": recent[-1] if recent else {},
        "macd": {},
        "recent_daily": recent,
        "indicators": indicators,
    }


async def backtest_symbol(
    symbol: str,
    bars: pd.DataFrame,
    start: date,
    end: date,
    cadence: int,
    use_llm: bool,
    non_overlap: bool = False,
) -> list[dict[str, Any]]:
    close = bars["close"].tolist()
    sim_dates = sample_dates(bars, start, end, cadence, non_overlap)
    rows: list[dict[str, Any]] = []

    for sim_date in sim_dates:
        index = bars.index[bars["date"] == sim_date]
        if len(index) == 0:
            continue
        idx = int(index[0])
        if idx < MIN_LOOKBACK or idx + MIN_FUTURE >= len(bars):
            continue

        history = bars.iloc[: idx + 1]
        features = prepare_daily_features(history)
        if len(features) < MIN_LOOKBACK:
            continue

        as_of = sim_date.strftime("%Y-%m-%d")
        records = features.to_dict(orient="records")
        quant_text = _build_quant_payload(symbol, records, as_of)
        quant = json.loads(quant_text)
        market_data = _market_data_from_bars(symbol, as_of, features)
        summary = await generate_llm_summary(market_data, quant, use_llm=use_llm)

        row: dict[str, Any] = {"symbol": symbol, "date": as_of}
        for horizon, offset in HORIZONS:
            pred = str(quant.get("horizons", {}).get(horizon, {}).get("direction", "flat"))
            actual_up = close[idx + offset] > close[idx]
            row[f"{horizon}_pred"] = pred
            row[f"{horizon}_actual"] = "up" if actual_up else "down"
            row[f"{horizon}_hit"] = evaluate_direction(pred, actual_up)

        summary_pred = str(summary.get("overall", "neutral"))
        actual_up = close[idx + SUMMARY_HORIZON_DAYS] > close[idx]
        row["summary_pred"] = summary_pred
        row["summary_actual"] = "up" if actual_up else "down"
        row["summary_hit"] = (
            evaluate_direction(
                "up" if summary_pred == "bullish" else "down" if summary_pred == "bearish" else "flat",
                actual_up,
            )
        )
        rows.append(row)

    return rows


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if total <= 0:
        return None, None
    p = correct / total
    denominator = 1 + z * z / total
    center = p + z * z / (2 * total)
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total)
    return (center - margin) / denominator, (center + margin) / denominator


def _py(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _metric_label_from_hit_key(key: str) -> str:
    return key.replace("_hit", "")


def _actual_key_for_label(label: str) -> str:
    return "summary_actual" if label == "summary" else f"{label}_actual"


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [f"{h}_hit" for h, _ in HORIZONS] + ["summary_hit"]
    metrics: dict[str, Any] = {}
    for key in keys:
        label = _metric_label_from_hit_key(key)
        actual_key = _actual_key_for_label(label)
        valid = [row for row in rows if row.get(key) is not None]
        correct = sum(1 for row in valid if row[key])
        total = len(rows)
        ci_low, ci_high = wilson_interval(correct, len(valid))
        accuracy = round(correct / len(valid), 4) if valid else None
        actual_up_count = sum(1 for row in rows if row.get(actual_key) == "up")
        baseline_up = round(actual_up_count / total, 4) if total else None
        baseline_down = round(1 - baseline_up, 4) if baseline_up is not None else None
        dominant = max(baseline_up, baseline_down) if baseline_up is not None else None
        p_value = None
        significant = False
        if valid and dominant is not None:
            try:
                p_value = round(
                    binomtest(correct, len(valid), p=dominant, alternative="greater").pvalue,
                    6,
                )
            except Exception:
                p_value = None
            significant = (
                accuracy is not None
                and accuracy > dominant
                and p_value is not None
                and p_value < 0.05
            )
        metrics[label] = {
            "samples": total,
            "valid": len(valid),
            "correct": correct,
            "flat": total - len(valid),
            "accuracy": accuracy,
            "ci_low": round(ci_low, 4) if ci_low is not None else None,
            "ci_high": round(ci_high, 4) if ci_high is not None else None,
            "baseline_always_up": baseline_up,
            "baseline_always_down": baseline_down,
            "dominant_baseline": dominant,
            "p_value_vs_baseline": p_value,
            "significant": significant,
        }
        metrics[label] = {key: _py(value) for key, value in metrics[label].items()}
    return metrics


def aggregate_metrics(symbols_metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    aggregated: dict[str, Any] = {}
    keys = [f"{h}_hit" for h, _ in HORIZONS] + ["summary_hit"]
    for key in keys:
        label = _metric_label_from_hit_key(key)
        samples = sum(item[label]["samples"] for item in symbols_metrics.values())
        valid = sum(item[label]["valid"] for item in symbols_metrics.values())
        correct = sum(item[label]["correct"] for item in symbols_metrics.values())
        flat = sum(item[label]["flat"] for item in symbols_metrics.values())
        ci_low, ci_high = wilson_interval(correct, valid)
        actual_up_count = sum(
            item[label]["baseline_always_up"] * item[label]["samples"]
            for item in symbols_metrics.values()
            if item[label]["baseline_always_up"] is not None
        )
        baseline_up = round(actual_up_count / samples, 4) if samples else None
        baseline_down = round(1 - baseline_up, 4) if baseline_up is not None else None
        dominant = max(baseline_up, baseline_down) if baseline_up is not None else None
        accuracy = round(correct / valid, 4) if valid else None
        p_value = None
        significant = False
        if valid and dominant is not None:
            try:
                p_value = round(
                    binomtest(correct, valid, p=dominant, alternative="greater").pvalue,
                    6,
                )
            except Exception:
                p_value = None
            significant = (
                accuracy is not None
                and accuracy > dominant
                and p_value is not None
                and p_value < 0.05
            )
        aggregated[label] = {
            "samples": samples,
            "valid": valid,
            "correct": correct,
            "flat": flat,
            "accuracy": accuracy,
            "ci_low": round(ci_low, 4) if ci_low is not None else None,
            "ci_high": round(ci_high, 4) if ci_high is not None else None,
            "baseline_always_up": baseline_up,
            "baseline_always_down": baseline_down,
            "dominant_baseline": dominant,
            "p_value_vs_baseline": p_value,
            "significant": significant,
        }
        aggregated[label] = {key: _py(value) for key, value in aggregated[label].items()}
    return aggregated


def write_outputs(
    rows: list[dict[str, Any]],
    symbols_metrics: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, str]:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"{timestamp}.csv"
    json_path = RESULTS_DIR / f"{timestamp}.json"

    fieldnames = ["symbol", "date"]
    for horizon, _ in HORIZONS:
        fieldnames += [f"{horizon}_pred", f"{horizon}_actual", f"{horizon}_hit"]
    fieldnames += ["summary_pred", "summary_actual", "summary_hit"]

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    payload = {
        "generated_at": datetime.now().isoformat(),
        "config": config,
        "symbols": symbols_metrics,
        "aggregate": aggregate_metrics(symbols_metrics),
    }
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "json": str(json_path)}


def print_report(symbols_metrics: dict[str, dict[str, Any]]) -> None:
    for symbol, metrics in symbols_metrics.items():
        print(f"\n== {symbol} ==")
        for label, item in metrics.items():
            accuracy = f"{item['accuracy']:.2%}" if item["accuracy"] is not None else "n/a"
            ci = ""
            if item["ci_low"] is not None and item["ci_high"] is not None:
                ci = f" CI[{item['ci_low']:.2%},{item['ci_high']:.2%}]"
            baseline_up = f"{item['baseline_always_up']:.2%}" if item["baseline_always_up"] is not None else "n/a"
            baseline_down = f"{item['baseline_always_down']:.2%}" if item["baseline_always_down"] is not None else "n/a"
            p_value = f"{item['p_value_vs_baseline']:.4f}" if item["p_value_vs_baseline"] is not None else "n/a"
            marker = " *" if item["significant"] else ""
            print(
                f"  {label:>8}: samples={item['samples']} valid={item['valid']} "
                f"correct={item['correct']} flat={item['flat']} accuracy={accuracy}{ci} "
                f"baseline_up={baseline_up} baseline_down={baseline_down} "
                f"p_vs_baseline={p_value}{marker}"
            )


async def run_backtest(config: dict[str, Any], cli: dict[str, Any]) -> dict[str, str]:
    symbols = cli.get("symbols") or config.get("symbols") or ["600988"]
    start = date.fromisoformat(cli.get("start") or config.get("start_date") or "2024-01-01")
    end = date.fromisoformat(cli.get("end") or config.get("end_date") or date.today().isoformat())
    cadence = int(cli.get("cadence") or config.get("cadence_trading_days") or 5)
    use_llm = not cli.get("no_llm", False) and bool(config.get("include_llm", True))
    non_overlap = bool(cli.get("non_overlap") or config.get("non_overlap", False))

    all_rows: list[dict[str, Any]] = []
    symbols_metrics: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        bars = ensure_history(symbol)
        if bars.empty:
            print(f"skip {symbol}: no history")
            continue
        print(f"backtesting {symbol}: {len(bars)} bars, cadence {cadence}, use_llm={use_llm}")
        rows = await backtest_symbol(symbol, bars, start, end, cadence, use_llm, non_overlap)
        symbols_metrics[symbol] = compute_metrics(rows)
        all_rows.extend(rows)
        print(f"  {symbol}: {len(rows)} samples")

    print_report(symbols_metrics)
    return write_outputs(all_rows, symbols_metrics, config)


def main() -> int:
    parser = argparse.ArgumentParser(description="A-share analysis regression/backtest")
    parser.add_argument("--symbols", help="comma separated 6-digit codes, overrides config")
    parser.add_argument("--start", help="YYYY-MM-DD")
    parser.add_argument("--end", help="YYYY-MM-DD")
    parser.add_argument("--cadence", type=int, help="trading-day interval")
    parser.add_argument("--no-llm", action="store_true", help="skip LLM summary calls")
    parser.add_argument("--non-overlap", action="store_true", help="non-overlapping sample dates")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="config yaml path")
    args = parser.parse_args()

    config = load_config(args.config)
    cli = {
        "symbols": args.symbols.split(",") if args.symbols else None,
        "start": args.start,
        "end": args.end,
        "cadence": args.cadence,
        "no_llm": args.no_llm,
        "non_overlap": args.non_overlap,
    }
    outputs = asyncio.run(run_backtest(config, cli))
    print("\noutputs:")
    for key, path in outputs.items():
        print(f"  {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
