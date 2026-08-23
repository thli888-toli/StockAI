"""Async client for the live agent registry."""

from __future__ import annotations

import httpx

from framework.config import REGISTRY_URL
from framework.schemas import AgentCard, AgentStatusView


class RegistryClient:
    def __init__(self, base_url: str = REGISTRY_URL, timeout: float = 5.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    async def register(self, card: AgentCard) -> AgentStatusView:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/register", json=card.model_dump())
            response.raise_for_status()
            return AgentStatusView.model_validate(response.json())

    async def heartbeat(self, name: str) -> AgentStatusView:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(f"{self.base_url}/heartbeat/{name}")
            response.raise_for_status()
            return AgentStatusView.model_validate(response.json())

    async def list_agents(self) -> list[AgentStatusView]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/agents")
            response.raise_for_status()
            return [AgentStatusView.model_validate(item) for item in response.json()]

    async def get_agent(self, name: str) -> AgentStatusView:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(f"{self.base_url}/agents/{name}")
            response.raise_for_status()
            return AgentStatusView.model_validate(response.json())
