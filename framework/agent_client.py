"""Async client for calling out-of-process agent services."""

from __future__ import annotations

import asyncio
import os
import time

import httpx

from framework.registry_client import RegistryClient
from framework.schemas import AgentStatusView, TaskRequest, TaskStatus, TaskState


class RemoteAgentClient:
    def __init__(
        self,
        registry: RegistryClient,
        timeout: float | None = None,
        poll_interval: float = 0.15,
        max_retries: int = 2,
        cache_ttl: float = 5.0,
    ) -> None:
        self.registry = registry
        self.timeout = timeout if timeout is not None else float(os.getenv("RUN_TIMEOUT_SECONDS", "180"))
        self.poll_interval = poll_interval
        self.max_retries = max_retries
        self.cache_ttl = cache_ttl
        self._endpoint_cache: dict[str, tuple[float, str]] = {}

    async def _resolve(self, agent_name: str) -> str:
        now = time.monotonic()
        cached = self._endpoint_cache.get(agent_name)
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]
        view = await self.registry.get_agent(agent_name)
        if view.status == "offline":
            raise RuntimeError(f"agent '{agent_name}' is offline")
        endpoint = view.card.endpoint.rstrip("/")
        self._endpoint_cache[agent_name] = (now, endpoint)
        return endpoint

    async def run(self, agent_name: str, request: TaskRequest) -> str:
        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                endpoint = await self._resolve(agent_name)
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    create_resp = await client.post(f"{endpoint}/tasks", json=request.model_dump())
                    create_resp.raise_for_status()
                    task = TaskStatus.model_validate(create_resp.json())
                return await self._poll(endpoint, task.task_id)
            except (httpx.HTTPError, RuntimeError) as exc:
                last_error = exc
                if attempt < self.max_retries:
                    await asyncio.sleep(1.0 * (2**attempt))
        message = str(last_error) or type(last_error).__name__
        raise RuntimeError(f"agent '{agent_name}' call failed: {message}") from last_error

    async def _poll(self, endpoint: str, task_id: str) -> str:
        deadline = time.monotonic() + self.timeout
        async with httpx.AsyncClient(timeout=5.0) as client:
            consecutive_errors = 0
            while time.monotonic() < deadline:
                try:
                    response = await client.get(f"{endpoint}/tasks/{task_id}")
                    response.raise_for_status()
                    task = TaskStatus.model_validate(response.json())
                except httpx.HTTPError as exc:
                    consecutive_errors += 1
                    if consecutive_errors >= 10:
                        message = str(exc) or type(exc).__name__
                        raise RuntimeError(f"agent task polling failed repeatedly: {message}") from exc
                    await asyncio.sleep(self.poll_interval)
                    continue
                consecutive_errors = 0
                if task.state == TaskState.COMPLETED:
                    return task.artifact or ""
                if task.state == TaskState.FAILED:
                    raise RuntimeError(task.error or "agent task failed")
                await asyncio.sleep(self.poll_interval)
        raise TimeoutError(f"agent task '{task_id}' timed out")

    async def get_agent(self, name: str) -> AgentStatusView:
        return await self.registry.get_agent(name)
