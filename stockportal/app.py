"""FastAPI backend for the dedicated stock analysis portal."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from framework.config import ORCHESTRATOR_URL, STOCK_PORTAL_DB
from plugins.stock_common import compute_macd
from stockportal.auth import AuthStore
from stockportal.store import WatchlistStore


class RunCreatePayload(BaseModel):
    query: str
    context: str | None = None
    manifest_name: str | None = None


class WatchlistAddPayload(BaseModel):
    query: str


class LoginPayload(BaseModel):
    openid: str | None = None
    nickname: str | None = None


def _validate_symbol(symbol: str) -> str:
    symbol = symbol.strip()
    if not re.fullmatch(r"\d{6}", symbol):
        raise HTTPException(status_code=422, detail="symbol must be a 6-digit A-share code")
    return symbol


def _is_today(iso_value: str | None) -> bool:
    if not iso_value:
        return False
    try:
        value = datetime.fromisoformat(iso_value)
    except Exception:
        return False
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    shanghai = timezone(timedelta(hours=8))
    return value.astimezone(shanghai).date() == datetime.now(shanghai).date()


def _market_data_from_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    market_data = outputs.get("market_data")
    if isinstance(market_data, str):
        try:
            return json.loads(market_data)
        except Exception:
            return {}
    if isinstance(market_data, dict):
        return market_data
    return {}


def _quant_from_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    quant = outputs.get("quant")
    if isinstance(quant, str):
        try:
            return json.loads(quant)
        except Exception:
            return {}
    if isinstance(quant, dict):
        return quant
    return {}


def _fibonacci_levels(
    frame: pd.DataFrame,
    n: int = 2,
    lookback: int = 20,
) -> tuple[float, float]:
    highs = frame["high"].tolist()
    lows = frame["low"].tolist()
    swing_highs: list[float] = []
    swing_lows: list[float] = []
    for index in range(n, len(frame) - n):
        if highs[index] == max(highs[index - n : index + n + 1]):
            swing_highs.append(highs[index])
        if lows[index] == min(lows[index - n : index + n + 1]):
            swing_lows.append(lows[index])

    recent_high = max(highs[-lookback:]) if len(highs) >= lookback else max(highs)
    recent_low = min(lows[-lookback:]) if len(lows) >= lookback else min(lows)
    swing_high = swing_highs[-1] if swing_highs else recent_high
    swing_low = swing_lows[-1] if swing_lows else recent_low
    if swing_high <= swing_low:
        swing_high, swing_low = recent_high, recent_low
    if swing_high <= swing_low:
        swing_high = swing_low + 1.0

    price_range = swing_high - swing_low
    ratios = (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)
    levels = [swing_low + price_range * ratio for ratio in ratios]

    last_close = float(frame["close"].iloc[-1])
    supports = [value for value in levels if value <= last_close]
    resistances = [value for value in levels if value >= last_close]
    support = max(supports) if supports else min(levels)
    resistance = min(resistances) if resistances else max(levels)
    return round(support, 4), round(resistance, 4)


def _summary_from_report(report: Any) -> dict[str, Any] | None:
    if not isinstance(report, str):
        return None
    try:
        data = json.loads(report)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    summary = data.get("summary")
    if not isinstance(summary, dict):
        return None
    overall = str(summary.get("overall") or "").strip().lower()
    if overall not in ("bullish", "bearish", "neutral"):
        return None
    return {"overall": overall, "text": str(summary.get("text") or "")}


def _build_chart_payload(
    symbol: str,
    market_data: dict[str, Any],
    period: str,
    quant: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    features = market_data.get("daily_features") or []
    if not features:
        return None

    df = pd.DataFrame(features)
    required = ["date", "open", "high", "low", "close", "volume"]
    if not set(required).issubset(df.columns):
        return None

    df = df[required].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for column in ("open", "high", "low", "close", "volume"):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=required).sort_values("date").reset_index(drop=True)
    if df.empty:
        return None

    if period == "weekly":
        frame = (
            df.set_index("date")
            .resample("W-FRI")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
    elif period == "monthly":
        frame = (
            df.set_index("date")
            .resample("ME")
            .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
    else:
        frame = df.copy()

    weekly_frame = (
        df.set_index("date")
        .resample("W-FRI")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )
    monthly_frame = (
        df.set_index("date")
        .resample("ME")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )

    close = frame["close"]
    macd = compute_macd(close, 6, 13, 5)

    ma: dict[str, list[dict[str, Any]]] = {}
    for window in (20, 66, 154, 250):
        values = close.rolling(window).mean()
        ma[f"ma{window}"] = [
            {"time": date_value.strftime("%Y-%m-%d"), "value": round(float(value), 4)}
            for date_value, value in zip(frame["date"], values)
            if pd.notna(value)
        ]

    dif: list[dict[str, Any]] = []
    dea: list[dict[str, Any]] = []
    histogram: list[dict[str, Any]] = []
    for date_value, dif_value, dea_value, hist_value in zip(
        frame["date"], macd["macd"], macd["signal"], macd["histogram"]
    ):
        if pd.notna(dif_value):
            time_value = date_value.strftime("%Y-%m-%d")
            dif.append({"time": time_value, "value": round(float(dif_value), 6)})
            dea.append({"time": time_value, "value": round(float(dea_value), 6)})
            histogram.append({"time": time_value, "value": round(float(hist_value), 6)})

    candles = [
        {
            "time": row["date"].strftime("%Y-%m-%d"),
            "open": round(float(row["open"]), 4),
            "high": round(float(row["high"]), 4),
            "low": round(float(row["low"]), 4),
            "close": round(float(row["close"]), 4),
            "volume": float(row["volume"]),
        }
        for _, row in frame.iterrows()
    ]

    horizons = (quant or {}).get("horizons", {}) or {}
    signals = {
        horizon: str((horizons.get(horizon, {}) or {}).get("direction", "flat")).lower()
        for horizon in ("5d", "15d", "1w", "1mo")
    }
    daily_support, daily_resistance = _fibonacci_levels(df)
    weekly_support, weekly_resistance = _fibonacci_levels(weekly_frame)
    monthly_support, monthly_resistance = _fibonacci_levels(monthly_frame)
    levels = {
        "daily": {"support": daily_support, "resistance": daily_resistance},
        "weekly": {"support": weekly_support, "resistance": weekly_resistance},
        "monthly": {"support": monthly_support, "resistance": monthly_resistance},
    }

    return {
        "symbol": symbol,
        "period": period,
        "candles": candles,
        "ma": ma,
        "macd": {"dif": dif, "dea": dea, "histogram": histogram},
        "signals": signals,
        "levels": levels,
    }


def _create_run(orchestrator_url: str, symbol: str) -> dict[str, Any]:
    base = orchestrator_url.rstrip("/")
    try:
        response = httpx.post(
            f"{base}/runs",
            json={"query": symbol},
            timeout=10.0,
        )
    except httpx.HTTPError as exc:
        return {
            "run_id": None,
            "status": "failed",
            "error": f"orchestrator run creation failed: {exc}",
            "outputs": {},
        }
    if response.status_code >= 400:
        try:
            detail = response.json()
        except Exception:
            detail = response.text
        return {
            "run_id": None,
            "status": "failed",
            "error": f"orchestrator rejected run: {detail}",
            "outputs": {},
        }
    try:
        run = response.json()
    except Exception as exc:
        return {
            "run_id": None,
            "status": "failed",
            "error": f"invalid orchestrator response: {exc}",
            "outputs": {},
        }
    outputs = run.get("outputs") or {}
    if not isinstance(outputs, dict):
        outputs = {}
    return {
        "run_id": run.get("run_id"),
        "status": run.get("status") or "running",
        "error": run.get("error"),
        "outputs": outputs,
    }


def _metadata_from_outputs(outputs: dict[str, Any]) -> tuple[str, str]:
    if not isinstance(outputs, dict):
        return "", ""
    market_data = outputs.get("market_data")
    data: dict[str, Any] | None = None
    if isinstance(market_data, str):
        try:
            data = json.loads(market_data)
        except Exception:
            data = None
    elif isinstance(market_data, dict):
        data = market_data
    if not data:
        return "", ""
    return str(data.get("company_name") or ""), str(data.get("industry") or "")


def _sync_watchlist(store: WatchlistStore, user_id: str, orchestrator_url: str) -> None:
    base = orchestrator_url.rstrip("/")
    try:
        response = httpx.get(f"{base}/runs", timeout=5.0)
        response.raise_for_status()
        runs = response.json()
    except (httpx.HTTPError, ValueError):
        return
    if not isinstance(runs, list):
        return
    runs_by_id = {
        run["run_id"]: run
        for run in runs
        if isinstance(run, dict) and run.get("run_id")
    }
    for item in store.all_items(user_id):
        run_id = item.get("run_id")
        run = runs_by_id.get(run_id) if run_id else None
        if run:
            outputs = run.get("outputs") or {}
            if not isinstance(outputs, dict):
                outputs = {}
            company_name, industry = _metadata_from_outputs(outputs)
            store.upsert(
                user_id,
                item["symbol"],
                run_id=run_id,
                status=run.get("status", "failed"),
                error=run.get("error"),
                outputs=outputs,
                company_name=company_name or item.get("company_name") or "",
                industry=industry or item.get("industry") or "",
            )
        elif item.get("status") == "running":
            store.upsert(
                user_id,
                item["symbol"],
                run_id=run_id,
                status="failed",
                error="run no longer available (orchestrator may have restarted)",
                outputs=item.get("outputs") or {},
                company_name=item.get("company_name") or "",
                industry=item.get("industry") or "",
            )


def create_stock_portal_app(
    orchestrator_url: str = ORCHESTRATOR_URL,
    db_path: str | Path = STOCK_PORTAL_DB,
) -> FastAPI:
    store = WatchlistStore(db_path)
    auth_store = AuthStore(db_path)

    def get_current_user(authorization: str | None = Header(default=None)) -> dict[str, Any]:
        token = None
        if authorization and authorization.lower().startswith("bearer "):
            token = authorization[7:].strip()
        user = auth_store.get_user_by_token(token)
        if user is None:
            raise HTTPException(status_code=401, detail="未登录")
        return user

    app = FastAPI(title="StockAI Portal", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "portal": "stock"}

    @app.post("/api/login")
    def login(payload: LoginPayload):
        return auth_store.login(payload.openid, payload.nickname)

    @app.get("/api/me")
    def me(user: dict[str, Any] = Depends(get_current_user)):
        return user

    @app.post("/api/runs")
    def create_run(payload: RunCreatePayload):
        try:
            response = httpx.post(
                f"{orchestrator_url.rstrip('/')}/runs",
                json=payload.model_dump(),
                timeout=10.0,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise HTTPException(status_code=response.status_code, detail=detail)
            return response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"orchestrator run creation failed: {exc}",
            ) from exc

    @app.get("/api/runs/{run_id}")
    def get_run(run_id: str):
        try:
            response = httpx.get(
                f"{orchestrator_url.rstrip('/')}/runs/{run_id}",
                timeout=10.0,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise HTTPException(status_code=response.status_code, detail=detail)
            return response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"orchestrator run lookup failed: {exc}",
            ) from exc

    @app.get("/api/watchlist")
    def list_watchlist(user: dict[str, Any] = Depends(get_current_user)):
        _sync_watchlist(store, user["user_id"], orchestrator_url)
        return store.all_items(user["user_id"])

    @app.post("/api/watchlist")
    def add_to_watchlist(
        payload: WatchlistAddPayload,
        user: dict[str, Any] = Depends(get_current_user),
    ):
        symbol = _validate_symbol(payload.query)
        if store.get(user["user_id"], symbol) is not None:
            return {"already_exists": True, "message": "股票已经在股票池中"}
        run = _create_run(orchestrator_url, symbol)
        return store.upsert(
            user["user_id"],
            symbol,
            run_id=run["run_id"],
            status=run["status"],
            error=run["error"],
            outputs=run["outputs"],
        )

    @app.post("/api/watchlist/{symbol}/refresh")
    def refresh_watchlist(
        symbol: str,
        user: dict[str, Any] = Depends(get_current_user),
    ):
        symbol = _validate_symbol(symbol)
        existing = store.get(user["user_id"], symbol)
        if existing is None:
            raise HTTPException(status_code=404, detail="watchlist symbol not found")
        if existing.get("status") == "completed" and _is_today(existing.get("updated_at")):
            return {
                "already_generated": True,
                "message": "今日股票分析已经生成，无需重复刷新",
            }
        run = _create_run(orchestrator_url, symbol)
        return store.upsert(
            user["user_id"],
            symbol,
            run_id=run["run_id"],
            status=run["status"],
            error=run["error"],
            outputs=run["outputs"],
            company_name=existing.get("company_name") or "",
            industry=existing.get("industry") or "",
        )

    @app.delete("/api/watchlist/{symbol}")
    def delete_watchlist(
        symbol: str,
        user: dict[str, Any] = Depends(get_current_user),
    ):
        symbol = _validate_symbol(symbol)
        if not store.delete(user["user_id"], symbol):
            raise HTTPException(status_code=404, detail="watchlist symbol not found")
        return {"deleted": True}

    @app.get("/api/watchlist/{symbol}/chart")
    def chart(
        symbol: str,
        period: str = "daily",
        user: dict[str, Any] = Depends(get_current_user),
    ):
        symbol = _validate_symbol(symbol)
        if period not in ("daily", "weekly", "monthly"):
            raise HTTPException(status_code=422, detail="period must be daily, weekly, or monthly")
        item = store.get(user["user_id"], symbol)
        if item is None:
            raise HTTPException(status_code=404, detail="watchlist symbol not found")
        market_data = _market_data_from_outputs(item.get("outputs") or {})
        quant = _quant_from_outputs(item.get("outputs") or {})
        payload = _build_chart_payload(symbol, market_data, period, quant)
        if payload is None:
            raise HTTPException(status_code=404, detail="market data not available yet")
        payload["llm_summary"] = _summary_from_report((item.get("outputs") or {}).get("report"))
        return payload

    dist_dir = Path(__file__).resolve().parent / "ui" / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="stockportal")

    return app
