"""Live agent registry with heartbeat-based status."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status

from framework.schemas import AgentCard, AgentStatusView, RegisteredAgent, utcnow


class AgentRegistry:
    """In-memory registry. Entries become stale/offline without heartbeats."""

    def __init__(self, ttl_seconds: float = 5.0, stale_after_seconds: float = 20.0) -> None:
        self.ttl_seconds = ttl_seconds
        self.stale_after_seconds = stale_after_seconds
        self._agents: dict[str, RegisteredAgent] = {}
        self._lock = asyncio.Lock()

    async def register(self, card: AgentCard) -> AgentStatusView:
        async with self._lock:
            self._agents[card.name] = RegisteredAgent(card=card)
        return self.status_view(card.name)

    async def heartbeat(self, name: str) -> AgentStatusView:
        async with self._lock:
            agent = self._agents.get(name)
            if agent is None:
                raise KeyError(name)
            agent.last_heartbeat = utcnow()
        return self.status_view(name)

    async def list_agents(self) -> list[AgentStatusView]:
        async with self._lock:
            names = list(self._agents)
        return [self.status_view(name) for name in names]

    async def get_agent(self, name: str) -> AgentStatusView:
        if name not in self._agents:
            raise KeyError(name)
        return self.status_view(name)

    def status_view(self, name: str) -> AgentStatusView:
        agent = self._agents[name]
        age = agent.age_seconds
        if age <= self.ttl_seconds:
            health = "healthy"
        elif age <= self.stale_after_seconds:
            health = "stale"
        else:
            health = "offline"
        return AgentStatusView(
            name=name,
            status=health,
            last_heartbeat=agent.last_heartbeat,
            card=agent.card,
        )

    async def evict_offline(self) -> None:
        async with self._lock:
            for name in list(self._agents):
                if self._agents[name].age_seconds > self.stale_after_seconds * 3:
                    self._agents.pop(name, None)


def create_registry_app(
    ttl_seconds: float = 5.0,
    stale_after_seconds: float = 20.0,
) -> FastAPI:
    registry = AgentRegistry(ttl_seconds, stale_after_seconds)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        async def cleanup_loop() -> None:
            while True:
                await asyncio.sleep(max(ttl_seconds, 1.0))
                await registry.evict_offline()

        task = asyncio.create_task(cleanup_loop())
        yield
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    app = FastAPI(title="Agent Registry", version="1.0.0", lifespan=lifespan)

    @app.post("/register", response_model=AgentStatusView, status_code=status.HTTP_200_OK)
    async def register(card: AgentCard):
        return await registry.register(card)

    @app.post("/heartbeat/{name}", response_model=AgentStatusView)
    async def heartbeat(name: str):
        try:
            return await registry.heartbeat(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"agent '{name}' is not registered") from exc

    @app.get("/agents", response_model=list[AgentStatusView])
    async def list_agents():
        return await registry.list_agents()

    @app.get("/agents/{name}", response_model=AgentStatusView)
    async def get_agent(name: str):
        try:
            return await registry.get_agent(name)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=f"agent '{name}' is not registered") from exc

    @app.get("/health")
    async def health(request: Request):
        return {"status": "ok"}

    return app


registry_app = create_registry_app()
