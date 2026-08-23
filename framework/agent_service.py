"""Generic FastAPI agent service used by every out-of-process plugin."""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Protocol

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import TypeAdapter

from framework.config import PORTAL_BACKEND_URL, REGISTRY_URL
from framework.logging import configure_service_logger, log_event
from framework.metrics import MetricsRegistry, Timer
from framework.registry_client import RegistryClient
from framework.schemas import AgentManifest, AgentCard, TaskRequest, TaskState, TaskStatus, utcnow


class AgentHandler(Protocol):
    async def run(self, request: TaskRequest) -> str:
        ...


def load_agent_manifest(path: str | Path) -> AgentManifest:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return AgentManifest.model_validate(data)


def import_entrypoint(entrypoint: str) -> Any:
    module_name, _, attr = entrypoint.partition(":")
    import importlib

    module = importlib.import_module(module_name)
    return getattr(module, attr)


class AgentService:
    """Hosts one plugin handler behind a standard task-lifecycle API."""

    def __init__(
        self,
        manifest: AgentManifest,
        handler: AgentHandler,
        registry_url: str = REGISTRY_URL,
        heartbeat_seconds: float = 3.0,
        public_url: str | None = None,
    ) -> None:
        self.manifest = manifest
        self.handler = handler
        self.registry = RegistryClient(registry_url)
        self.heartbeat_seconds = heartbeat_seconds
        self.public_url = (public_url or os.getenv("AGENT_PUBLIC_URL") or f"http://127.0.0.1:{manifest.port}").rstrip("/")
        self.tasks: dict[str, TaskStatus] = {}
        self._heartbeat_task: asyncio.Task | None = None
        self.card = manifest.to_card(self.public_url)
        self.metrics = MetricsRegistry(prefix=f"agent.{manifest.name}.")
        self.logger = configure_service_logger(
            "agent",
            agent=manifest.name,
            portal_url=os.getenv("PORTAL_BACKEND_URL", PORTAL_BACKEND_URL),
        )

        @asynccontextmanager
        async def lifespan(app: FastAPI):
            await self.register()
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            yield
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass

        self.app = FastAPI(title=f"Agent: {manifest.name}", version=manifest.version, lifespan=lifespan)
        self._setup_routes()

    async def register(self) -> None:
        await self.registry.register(self.card)

    async def _heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(self.heartbeat_seconds)
            try:
                await self.registry.heartbeat(self.manifest.name)
            except Exception:
                # Registry may be briefly unavailable during startup; retry next cycle.
                try:
                    await self.registry.register(self.card)
                except Exception:
                    pass

    def _setup_routes(self) -> None:
        @self.app.get("/health")
        async def health():
            return {"status": "ok", "agent": self.manifest.name}

        @self.app.get("/agent-card", response_model=AgentCard)
        async def agent_card():
            return self.card

        @self.app.get("/metrics")
        async def metrics():
            return {
                "agent": self.manifest.name,
                **self.metrics.to_json(),
            }

        @self.app.post("/tasks", response_model=TaskStatus, status_code=202)
        async def create_task(request: TaskRequest):
            task_id = request.task_id or uuid.uuid4().hex
            if task_id in self.tasks:
                raise HTTPException(status_code=409, detail="task already exists")
            task = TaskStatus(task_id=task_id, state=TaskState.SUBMITTED)
            self.tasks[task_id] = task
            self.metrics.inc("requests.total")
            self.metrics.inc("requests.active")
            log_event(
                self.logger,
                "INFO",
                "task submitted",
                event="task.submitted",
                run_id=request.run_id,
                node_id=request.node_id,
            )
            asyncio.create_task(self._execute(task_id, request))
            return task

        @self.app.get("/tasks/{task_id}", response_model=TaskStatus)
        async def get_task(task_id: str):
            task = self.tasks.get(task_id)
            if task is None:
                raise HTTPException(status_code=404, detail="task not found")
            return task

    async def _execute(self, task_id: str, request: TaskRequest) -> None:
        task = self.tasks[task_id]
        task.state = TaskState.WORKING
        task.updated_at = utcnow()
        log_event(
            self.logger,
            "INFO",
            "task working",
            event="task.working",
            run_id=request.run_id,
            node_id=request.node_id,
        )
        timer = Timer(self.metrics, "latency_ms")
        try:
            artifact = await self.handler.run(request)
            task.artifact = artifact
            task.state = TaskState.COMPLETED
            task.completed_at = utcnow()
            self.metrics.inc("requests.completed")
            log_event(
                self.logger,
                "INFO",
                "task completed",
                event="task.completed",
                run_id=request.run_id,
                node_id=request.node_id,
            )
        except Exception as exc:  # noqa: BLE001
            task.error = str(exc)
            task.state = TaskState.FAILED
            task.completed_at = utcnow()
            self.metrics.inc("requests.failed")
            log_event(
                self.logger,
                "ERROR",
                f"task failed: {exc}",
                event="task.failed",
                run_id=request.run_id,
                node_id=request.node_id,
            )
        finally:
            timer.stop()
            self.metrics.inc("requests.active", -1)
        task.updated_at = utcnow()


def create_agent_app(
    manifest_path: str | Path,
    handler_factory: str | Any = "service:build_agent",
    registry_url: str = REGISTRY_URL,
    public_url: str | None = None,
) -> FastAPI:
    """Load a manifest and handler factory, then return the agent ASGI app."""
    manifest_path = Path(manifest_path)
    sys.path.insert(0, str(manifest_path.parent))
    manifest = load_agent_manifest(manifest_path)
    if isinstance(handler_factory, str):
        factory = import_entrypoint(handler_factory)
    else:
        factory = handler_factory
    handler = factory()
    return AgentService(manifest, handler, registry_url=registry_url, public_url=public_url).app
