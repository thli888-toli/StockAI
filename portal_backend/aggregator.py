"""Background collector that scrapes registry, orchestrator, and agent metrics."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from framework.config import ORCHESTRATOR_URL, REGISTRY_URL, US_ORCHESTRATOR_URL


class PortalAggregator:
    def __init__(
        self,
        store,
        registry_url: str = REGISTRY_URL,
        orchestrator_url: str = ORCHESTRATOR_URL,
        us_orchestrator_url: str = US_ORCHESTRATOR_URL,
        interval_seconds: float = 5.0,
    ) -> None:
        self.store = store
        self.registry_url = registry_url.rstrip("/")
        self.orchestrator_url = orchestrator_url.rstrip("/")
        self.us_orchestrator_url = us_orchestrator_url.rstrip("/")
        self.interval_seconds = interval_seconds

    async def run_forever(self) -> None:
        while True:
            await self.collect_once()
            await asyncio.sleep(self.interval_seconds)

    async def collect_once(self) -> None:
        async with httpx.AsyncClient(timeout=3.0) as client:
            await self._collect_registry(client)
            await self._collect_orchestrator(client, "a", self.orchestrator_url)
            await self._collect_orchestrator(client, "us", self.us_orchestrator_url)

    async def _collect_registry(self, client: httpx.AsyncClient) -> None:
        try:
            response = await client.get(f"{self.registry_url}/agents")
            response.raise_for_status()
            agents = response.json()
        except Exception:
            return
        for view in agents:
            self.store.upsert_agent(view)
            if view.get("status") == "healthy":
                await self._collect_agent_metrics(client, view)

    async def _collect_agent_metrics(self, client: httpx.AsyncClient, view: dict[str, Any]) -> None:
        endpoint = view.get("card", {}).get("endpoint")
        if not endpoint:
            return
        try:
            response = await client.get(f"{endpoint}/metrics")
            response.raise_for_status()
            self.store.insert_metric(view["name"], response.json())
        except Exception:
            return

    async def _collect_orchestrator(
        self,
        client: httpx.AsyncClient,
        market: str,
        orchestrator_url: str,
    ) -> None:
        try:
            response = await client.get(f"{orchestrator_url}/graph")
            response.raise_for_status()
            self.store.set_graph(market, response.json())
        except Exception:
            pass
        try:
            response = await client.get(f"{orchestrator_url}/graph-configs")
            response.raise_for_status()
            self.store.set_graph_configs(market, response.json())
        except Exception:
            pass
        try:
            response = await client.get(f"{orchestrator_url}/metrics")
            response.raise_for_status()
            self.store.insert_metric(
                "orchestrator_us" if market == "us" else "orchestrator",
                response.json(),
            )
        except Exception:
            pass
        try:
            response = await client.get(f"{orchestrator_url}/runs")
            response.raise_for_status()
            runs = response.json()
            for run in runs:
                item = dict(run)
                item["market"] = market
                self.store.upsert_run(item)
            known_ids = {run["run_id"] for run in runs}
            self.store.mark_orphaned_running_runs_failed(known_ids, market)
        except Exception:
            pass
