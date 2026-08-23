"""Portal backend FastAPI application."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from framework.config import ORCHESTRATOR_URL, PORTAL_DB, REGISTRY_URL
from portal_backend.aggregator import PortalAggregator
from portal_backend.store import PortalStore


class LogIngestPayload(BaseModel):
    records: list[dict[str, Any]]


class MetricIngestPayload(BaseModel):
    service: str
    metrics: dict[str, Any]


class GraphConfigApplyPayload(BaseModel):
    name: str


def create_portal_app(
    db_path: str = PORTAL_DB,
    registry_url: str = REGISTRY_URL,
    orchestrator_url: str = ORCHESTRATOR_URL,
) -> FastAPI:
    store = PortalStore(db_path)
    aggregator = PortalAggregator(store, registry_url, orchestrator_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        task = asyncio.create_task(aggregator.run_forever())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="Agent Management Portal", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/agents")
    def agents():
        return store.get_agents()

    @app.get("/api/graph")
    def graph():
        try:
            response = httpx.get(f"{orchestrator_url.rstrip('/')}/graph", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return store.get_graph()

    @app.get("/api/graph-configs")
    def graph_configs():
        try:
            response = httpx.get(f"{orchestrator_url.rstrip('/')}/graph-configs", timeout=3.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError:
            return store.get_graph_configs()

    @app.post("/api/graph-configs/apply")
    def apply_graph_config(payload: GraphConfigApplyPayload):
        try:
            response = httpx.post(
                f"{orchestrator_url.rstrip('/')}/graph-configs/apply",
                json={"name": payload.name},
                timeout=5.0,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise HTTPException(status_code=response.status_code, detail=detail)
            return response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"orchestrator apply failed: {exc}") from exc

    @app.get("/api/runs")
    def runs(
        limit: int = Query(default=50, ge=1, le=200),
        graph_config: str | None = None,
    ):
        return store.list_runs(limit, graph_config)

    @app.get("/api/runs/{run_id}")
    def run(run_id: str):
        item = store.get_run(run_id)
        if item is None:
            raise HTTPException(status_code=404, detail="run not found")
        return item

    @app.post("/api/runs/{run_id}/cancel")
    def cancel_run(run_id: str):
        try:
            response = httpx.post(
                f"{orchestrator_url.rstrip('/')}/runs/{run_id}/cancel",
                timeout=5.0,
            )
            if response.status_code >= 400:
                try:
                    detail = response.json()
                except Exception:
                    detail = response.text
                raise HTTPException(status_code=response.status_code, detail=detail)
            return response.json()
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=502, detail=f"orchestrator cancel failed: {exc}") from exc

    @app.get("/api/agents/{agent_name}/metrics")
    def agent_metrics(agent_name: str, limit: int = Query(default=200, ge=1, le=1000)):
        return store.query_metrics(agent_name, limit)

    @app.get("/api/agents/{agent_name}/logs")
    def agent_logs(
        agent_name: str,
        level: str | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        event: str | None = None,
        q: str | None = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        return store.query_logs(
            agent=agent_name,
            level=level,
            run_id=run_id,
            node_id=node_id,
            event=event,
            q=q,
            limit=limit,
        )

    @app.get("/api/logs")
    def all_logs(
        level: str | None = None,
        run_id: str | None = None,
        node_id: str | None = None,
        event: str | None = None,
        q: str | None = None,
        limit: int = Query(default=200, ge=1, le=1000),
    ):
        return store.query_logs(
            agent=None,
            level=level,
            run_id=run_id,
            node_id=node_id,
            event=event,
            q=q,
            limit=limit,
        )

    @app.post("/logs/ingest")
    def ingest_logs(payload: LogIngestPayload):
        store.insert_logs(payload.records)
        return {"accepted": len(payload.records)}

    @app.post("/metrics/ingest")
    def ingest_metrics(payload: MetricIngestPayload):
        store.insert_metric(payload.service, payload.metrics)
        return {"accepted": 1}

    dist_dir = Path(__file__).resolve().parents[1] / "portal-ui" / "dist"
    if dist_dir.exists():
        app.mount("/", StaticFiles(directory=dist_dir, html=True), name="portal")

    return app
