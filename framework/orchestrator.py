"""LangGraph-backed orchestrator with run API and SQLite checkpointing."""

from __future__ import annotations

import asyncio
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from langgraph.graph import StateGraph
from pydantic import BaseModel

from framework.agent_client import RemoteAgentClient
from framework.checkpoint import create_shared_checkpointer
from framework.config import (
    CHECKPOINT_DB,
    ORCHESTRATOR_MAX_RUNNING_RUNS,
    ORCHESTRATOR_QUEUE_CHECK_INTERVAL,
    PORTAL_BACKEND_URL,
    REGISTRY_URL,
    RUN_QUEUE_DB,
)
from framework.graph_compiler import GraphCompiler
from framework.logging import configure_service_logger, log_event
from framework.metrics import MetricsRegistry
from framework.registry_client import RegistryClient
from framework.run_queue import SQLiteRunQueue
from framework.schemas import GraphManifest, OrchestrationState, RunSummary, utcnow


class GraphConfigApply(BaseModel):
    name: str


def load_graph_manifest(path: str | Path) -> GraphManifest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return GraphManifest.model_validate(data)


class Orchestrator:
    """Loads the latest graph manifest on each run and executes it."""

    def __init__(
        self,
        manifest_path: str | Path,
        registry_url: str = REGISTRY_URL,
        checkpoint_db: str | Path = CHECKPOINT_DB,
        queue_db: str | Path = RUN_QUEUE_DB,
        run_timeout: float | None = None,
    ) -> None:
        manifest_path = Path(manifest_path)
        self.manifest_dir = manifest_path.resolve().parent
        self.active_manifest = manifest_path.name
        self.registry_url = registry_url
        self.checkpoint_db = checkpoint_db
        self.run_timeout = run_timeout or float(os.getenv("RUN_TIMEOUT_SECONDS", "120"))
        self.registry = RegistryClient(registry_url)
        self.agent_client = RemoteAgentClient(self.registry, timeout=self.run_timeout)
        self.runs: dict[str, RunSummary] = {}
        self._run_tasks: dict[str, asyncio.Task] = {}
        self._run_queue = SQLiteRunQueue(queue_db)
        self._dispatcher_task: asyncio.Task | None = None
        self.max_running_runs = ORCHESTRATOR_MAX_RUNNING_RUNS
        self.queue_check_interval = ORCHESTRATOR_QUEUE_CHECK_INTERVAL
        self._checkpointer = None
        self._checkpointer_lock = asyncio.Lock()
        self.metrics = MetricsRegistry(prefix="orchestrator.")
        self.logger = configure_service_logger(
            "orchestrator",
            agent="orchestrator",
            portal_url=os.getenv("PORTAL_BACKEND_URL", PORTAL_BACKEND_URL),
        )

    def current_manifest_path(self) -> Path:
        return self.manifest_dir / self.active_manifest

    def load_manifest(self) -> GraphManifest:
        return load_graph_manifest(self.current_manifest_path())

    async def _get_shared_checkpointer(self):
        if self._checkpointer is None:
            async with self._checkpointer_lock:
                if self._checkpointer is None:
                    self._checkpointer = await create_shared_checkpointer(self.checkpoint_db)
        return self._checkpointer

    def _recover_queued_runs(self) -> None:
        for item in self._run_queue.list_items():
            run_id = item["run_id"]
            if run_id not in self.runs:
                self.runs[run_id] = RunSummary(
                    run_id=run_id,
                    graph_config=item.get("graph_config"),
                    query=item.get("query", ""),
                    status="queued",
                )

    async def close(self) -> None:
        if self._dispatcher_task is not None:
            self._dispatcher_task.cancel()
            try:
                await self._dispatcher_task
            except asyncio.CancelledError:
                pass
            self._dispatcher_task = None
        if self._checkpointer is not None:
            checkpointer = self._checkpointer
            self._checkpointer = None
            await checkpointer.conn.close()
        self._run_queue.close()

    def list_graph_configs(self) -> list[dict[str, Any]]:
        configs = []
        for path in sorted(self.manifest_dir.glob("*.yaml")):
            try:
                load_graph_manifest(path)
            except Exception:
                continue
            configs.append(
                {
                    "name": path.name,
                    "active": path.name == self.active_manifest,
                }
            )
        return configs

    def set_active_manifest(self, name: str) -> None:
        candidate = (self.manifest_dir / name).resolve()
        if not candidate.is_relative_to(self.manifest_dir):
            raise ValueError("graph config must be inside the manifest directory")
        load_graph_manifest(candidate)
        self.active_manifest = candidate.name

    def apply_graph_config(self, name: str) -> None:
        active_run_ids = self.active_run_ids()
        if active_run_ids:
            raise ValueError(
                "cannot switch graph while runs are active: " + ", ".join(active_run_ids)
            )
        self.set_active_manifest(name)

    def start_run(
        self,
        query: str,
        run_id: str | None = None,
        context: str | None = None,
        manifest_name: str | None = None,
    ) -> RunSummary:
        run_id = run_id or uuid.uuid4().hex
        if manifest_name:
            self.set_active_manifest(manifest_name)
        graph_config = self.active_manifest
        summary = RunSummary(
            run_id=run_id,
            graph_config=graph_config,
            query=query,
            status="queued",
        )
        self.runs[run_id] = summary
        self.metrics.inc("runs.total")
        self._run_queue.enqueue(run_id, query, context, graph_config)
        log_event(self.logger, "INFO", "run queued", event="run.queued", run_id=run_id)
        self._ensure_dispatcher()
        return summary

    def _ensure_dispatcher(self) -> None:
        if self._dispatcher_task is None or self._dispatcher_task.done():
            self._dispatcher_task = asyncio.create_task(self._dispatcher_loop())

    async def _dispatcher_loop(self) -> None:
        while True:
            await asyncio.sleep(self.queue_check_interval)
            await self._process_queue()

    async def _process_queue(self) -> None:
        while self._running_run_ids_count() < self.max_running_runs:
            item = await asyncio.to_thread(self._run_queue.dequeue)
            if item is None:
                break
            run_id = item["run_id"]
            summary = self.runs.get(run_id)
            if summary is None or summary.status != "queued":
                continue
            summary.status = "running"
            summary.updated_at = utcnow()
            self.metrics.inc("runs.active")
            log_event(self.logger, "INFO", "run started", event="run.started", run_id=run_id)
            task = asyncio.create_task(
                self.execute_run(
                    run_id=run_id,
                    query=item["query"],
                    context=item["context"],
                    graph_config=item["graph_config"],
                )
            )
            self._run_tasks[run_id] = task

    def _running_run_ids_count(self) -> int:
        return sum(1 for summary in self.runs.values() if summary.status == "running")

    async def execute_run(
        self,
        run_id: str,
        query: str,
        context: str | None,
        graph_config: str,
    ) -> None:
        summary = self.runs[run_id]
        manifest = self.load_manifest()
        compiler = GraphCompiler(self.agent_client)
        try:
            checkpointer = await self._get_shared_checkpointer()
            graph = compiler.compile(manifest, checkpointer=checkpointer)
            initial: OrchestrationState = {
                "run_id": run_id,
                "query": query,
                "context": context,
                "outputs": {},
                "next": "",
                "steps": 0,
                "events": [],
            }
            result = await asyncio.wait_for(
                graph.ainvoke(
                    initial,
                    config={"configurable": {"thread_id": run_id}},
                ),
                timeout=self.run_timeout,
            )
            error = result.get("error")
            summary.outputs = result.get("outputs", {})
            summary.events = result.get("events", [])
            if error:
                summary.status = "failed"
                summary.error = error
                self.metrics.inc("runs.failed")
            else:
                summary.status = "completed"
                self.metrics.inc("runs.completed")
        except asyncio.TimeoutError:
            summary.status = "failed"
            summary.error = f"run timed out after {self.run_timeout:g}s"
            self.metrics.inc("runs.failed")
        except asyncio.CancelledError:
            summary.status = "failed"
            summary.error = "cancelled"
            self.metrics.inc("runs.failed")
        except Exception as exc:  # noqa: BLE001
            summary.status = "failed"
            summary.error = str(exc)
            self.metrics.inc("runs.failed")
        self.metrics.inc("runs.active", -1)
        log_event(
            self.logger,
            "ERROR" if summary.status == "failed" else "INFO",
            summary.error or "run completed",
            event=f"run.{summary.status}",
            run_id=run_id,
        )
        summary.updated_at = utcnow()
        self.runs[run_id] = summary
        self._run_tasks.pop(run_id, None)

    async def cancel_run(self, run_id: str) -> RunSummary:
        summary = self.runs.get(run_id)
        if summary is None:
            raise KeyError(run_id)
        if summary.status == "queued":
            self._run_queue.remove(run_id)
            summary.status = "failed"
            summary.error = "cancelled"
            summary.updated_at = utcnow()
            return summary
        if summary.status != "running":
            raise ValueError("run is not running")
        task = self._run_tasks.get(run_id)
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                pass
        else:
            summary.status = "failed"
            summary.error = "cancelled"
            summary.updated_at = utcnow()
        if self.runs[run_id].status == "running":
            self.runs[run_id].status = "failed"
            self.runs[run_id].error = "cancelled"
            self.runs[run_id].updated_at = utcnow()
        return summary

    async def run(
        self,
        query: str,
        run_id: str | None = None,
        context: str | None = None,
        manifest_name: str | None = None,
    ) -> RunSummary:
        """Compatibility wrapper: start and await one run."""
        summary = self.start_run(query, run_id, context, manifest_name)
        await self._process_queue()
        task = self._run_tasks.get(summary.run_id)
        while task is None:
            await asyncio.sleep(self.queue_check_interval)
            await self._process_queue()
            task = self._run_tasks.get(summary.run_id)
        await task
        return self.runs[summary.run_id]

    def active_run_ids(self) -> list[str]:
        return [
            run_id
            for run_id, summary in self.runs.items()
            if summary.status in ("queued", "running")
        ]

    def get_run(self, run_id: str) -> RunSummary:
        try:
            return self.runs[run_id]
        except KeyError as exc:
            raise KeyError(run_id) from exc

    def build_app(self) -> FastAPI:
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            self._recover_queued_runs()
            self._ensure_dispatcher()
            yield
            await self.close()

        app = FastAPI(title="Agent Orchestrator", version="1.0.0", lifespan=lifespan)

        @app.post("/runs", response_model=RunSummary)
        async def create_run(payload: dict[str, Any]):
            query = payload.get("query")
            if not query:
                raise HTTPException(status_code=422, detail="query is required")
            return self.start_run(
                query=str(query),
                run_id=payload.get("run_id"),
                context=payload.get("context"),
                manifest_name=payload.get("manifest_name"),
            )

        @app.get("/runs/{run_id}", response_model=RunSummary)
        async def get_run(run_id: str):
            try:
                return self.get_run(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="run not found") from exc

        @app.post("/runs/{run_id}/cancel", response_model=RunSummary)
        async def cancel_run(run_id: str):
            try:
                return await self.cancel_run(run_id)
            except KeyError as exc:
                raise HTTPException(status_code=404, detail="run not found") from exc
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc

        @app.get("/graph")
        async def get_graph():
            manifest = self.load_manifest()
            return {
                "name": manifest.name,
                "entry": manifest.entry,
                "nodes": {
                    node_id: spec.model_dump()
                    for node_id, spec in manifest.nodes.items()
                },
                "edges": [edge.model_dump(by_alias=True) for edge in manifest.edges],
            }

        @app.get("/graph-configs")
        async def get_graph_configs():
            return self.list_graph_configs()

        @app.post("/graph-configs/apply")
        async def apply_graph_config(payload: GraphConfigApply):
            try:
                self.apply_graph_config(payload.name)
            except (ValueError, FileNotFoundError) as exc:
                status_code = 409 if "active" in str(exc) else 422
                raise HTTPException(status_code=status_code, detail=str(exc)) from exc
            return self.list_graph_configs()

        @app.get("/metrics")
        async def metrics():
            return {
                "orchestrator": "main",
                **self.metrics.to_json(),
            }

        @app.get("/runs")
        async def list_runs():
            return sorted(self.runs.values(), key=lambda r: r.created_at, reverse=True)

        @app.get("/health")
        async def health():
            return {"status": "ok"}

        return app


def create_orchestrator_app(
    manifest_path: str | Path,
    registry_url: str = REGISTRY_URL,
    checkpoint_db: str | Path = CHECKPOINT_DB,
    queue_db: str | Path = RUN_QUEUE_DB,
    run_timeout: float | None = None,
) -> FastAPI:
    return Orchestrator(
        manifest_path,
        registry_url,
        checkpoint_db,
        queue_db,
        run_timeout=run_timeout,
    ).build_app()
