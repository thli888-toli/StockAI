"""Shared Pydantic schemas for the Day 5 orchestration framework."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field, field_validator, model_validator


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class TaskState(StrEnum):
    SUBMITTED = "submitted"
    WORKING = "working"
    COMPLETED = "completed"
    FAILED = "failed"


class AgentCapabilities(BaseModel):
    streaming: bool = False
    push_notifications: bool = False


class AgentCard(BaseModel):
    name: str
    description: str = ""
    version: str = "1.0.0"
    endpoint: str
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)


class AgentManifest(BaseModel):
    api_version: str = Field(alias="apiVersion")
    kind: Literal["Agent"] = "Agent"
    name: str
    version: str = "1.0.0"
    description: str = ""
    entrypoint: str
    port: int = 8011
    capabilities: AgentCapabilities = Field(default_factory=AgentCapabilities)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def to_card(self, endpoint: str) -> AgentCard:
        return AgentCard(
            name=self.name,
            description=self.description,
            version=self.version,
            endpoint=endpoint,
            capabilities=self.capabilities,
            inputs=self.inputs,
            outputs=self.outputs,
        )


class TaskRequest(BaseModel):
    task_id: str | None = None
    run_id: str | None = None
    node_id: str | None = None
    query: str = ""
    context: str | None = None
    inputs: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(BaseModel):
    task_id: str
    state: TaskState = TaskState.SUBMITTED
    artifact: str | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    completed_at: datetime | None = None


class RegisteredAgent(BaseModel):
    card: AgentCard
    last_heartbeat: datetime = Field(default_factory=utcnow)

    @property
    def age_seconds(self) -> float:
        return (utcnow() - self.last_heartbeat).total_seconds()


class AgentStatusView(BaseModel):
    name: str
    status: Literal["healthy", "stale", "offline"]
    last_heartbeat: datetime
    card: AgentCard


class NodeSpec(BaseModel):
    type: Literal["agent", "supervisor"]
    agent: str | None = None
    input: dict[str, str] = Field(default_factory=dict)
    output_key: str | None = None
    prompt: str | None = None
    options: list[str] = Field(default_factory=list)
    decision_key: str = "next"

    @model_validator(mode="after")
    def _validate_agent_node(self) -> "NodeSpec":
        if self.type == "agent" and not self.agent:
            raise ValueError("agent node requires an 'agent' name")
        if self.type == "supervisor" and not self.options and not self.prompt:
            raise ValueError("supervisor node requires 'options' or 'prompt'")
        return self


class EdgeSpec(BaseModel):
    source: str = Field(alias="from")
    to: str
    when: str | None = None

    model_config = {"populate_by_name": True}


class GraphManifest(BaseModel):
    version: int = 1
    name: str
    entry: str
    nodes: dict[str, NodeSpec]
    edges: list[EdgeSpec] = Field(default_factory=list)

    @field_validator("nodes")
    @classmethod
    def _non_empty_nodes(cls, value: dict[str, NodeSpec]) -> dict[str, NodeSpec]:
        if not value:
            raise ValueError("graph must define at least one node")
        return value

    @model_validator(mode="after")
    def _validate_refs(self) -> "GraphManifest":
        if self.entry not in self.nodes:
            raise ValueError(f"entry node '{self.entry}' is not defined")
        node_ids = set(self.nodes)
        for edge in self.edges:
            if edge.source not in node_ids:
                raise ValueError(f"edge source '{edge.source}' is not a node")
            if edge.to not in node_ids and edge.to != "END":
                raise ValueError(f"edge target '{edge.to}' is not a node")
        for node_id, spec in self.nodes.items():
            if spec.type == "agent" and spec.agent:
                # The agent is resolved from the live registry at runtime, not from YAML.
                continue
            if spec.type == "supervisor":
                allowed = set(spec.options)
                for edge in self.edges:
                    if edge.source == node_id and edge.when:
                        allowed.add(edge.when)
                unknown = allowed - node_ids - {"finish", "END"}
                if unknown:
                    raise ValueError(
                        f"supervisor '{node_id}' references unknown options: {sorted(unknown)}"
                    )
        return self


class OrchestrationState(TypedDict, total=False):
    run_id: str
    query: str
    context: str
    outputs: dict[str, Any]
    next: str
    steps: int
    error: str
    events: list[dict[str, Any]]


class RunSummary(BaseModel):
    run_id: str
    graph_config: str | None = None
    status: Literal["running", "completed", "failed"] = "running"
    query: str
    outputs: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
    events: list[dict[str, Any]] = Field(default_factory=list)
