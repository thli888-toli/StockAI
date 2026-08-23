# Day 5 Agent Orchestration — Requirements

> Status: Approved for implementation  
> Scope: local development / learning module  
> Last updated: 2026-08-22

## 1. Overview

Build a hot-pluggable agent orchestration framework with a simple management portal. Agents run as independent FastAPI services, register themselves with a live registry, and are orchestrated by a LangGraph-backed runtime using declarative graph manifests. Two supported graph modes are provided: a fixed pipeline and an LLM-decided supervisor graph.

The management portal initially provides four monitoring capabilities:

1. Status of each agent.
2. Orchestration topology and run path.
3. Traffic of each agent.
4. Standard structured logs per agent, searchable and filterable.

The system is designed to remain runnable without an LLM API key by using deterministic no-LLM fallbacks in example agents.

## 2. Goals

- Run agents as independently deployable processes.
- Hot-add new agents without restarting the orchestrator or portal.
- Define orchestration graphs declaratively in `orchestration.yaml`.
- Switch between `orchestration.yaml` and `orchestration.llm.yaml` at runtime from the portal.
- Let a supervisor node use the configured LLM to choose the next agent.
- Track every run with its source graph configuration.
- Cancel or time out long-running jobs.
- Keep framework extension limited to adding plugin folders and graph manifest entries.
- Provide a single management portal for status, topology, traffic, and logs.
- Use Python standard library `logging` with structured JSON records.
- Persist orchestration state and portal observability data in SQLite.

## 3. Users

### Developer

- Creates and launches agent plugins.
- Edits `orchestration.yaml`.
- Reads technical documentation and runs local smoke tests.

### Operator

- Starts the local stack.
- Uses the portal to monitor agent health, orchestration runs, traffic, and logs.
- Adds a new agent while the system is running and verifies it appears in the portal.

## 4. In Scope

- Registry service with heartbeat-based discovery and health status.
- Generic out-of-process agent service.
- LangGraph-backed orchestrator with SQLite checkpointing.
- Declarative `agent.yaml` and `orchestration.yaml`.
- Declarative `orchestration.llm.yaml`.
- Management portal backend and React/Vite frontend.
- Standard JSON logging and per-agent log viewing.
- Request/latency/error metrics and per-agent traffic viewing.
- Three example plugins: `researcher`, `analyst`, `writer`.
- Local CLI to run registry, agents, orchestrator, portal backend, and portal UI.

## 5. Out of Scope

- Docker and container orchestration.
- Authentication, authorization, and multi-user access.
- Multi-host / production cluster deployment.
- External Redis, NATS, Kafka, Prometheus, Grafana, or OpenTelemetry.
- Token usage and cost tracking.
- Live migration of in-flight tasks after topology changes.
- Plugins distributed as pip packages or entry points.

## 6. Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| FR-1 | Registry discovers and health-checks out-of-process agents via heartbeat. | Agents register with an `AgentCard`; registry returns `healthy`, `stale`, or `offline`; entries expire after TTL without heartbeat. |
| FR-2 | Agent plugins register with a live registry and expose `/health`, `/agent-card`, `/metrics`, and `/tasks`. | Each endpoint returns the documented response shape; agent lifecycle moves through `submitted`, `working`, `completed`, or `failed`. |
| FR-3 | Orchestrator compiles a central graph manifest into LangGraph. | Valid `orchestration.yaml` produces an executable LangGraph graph; invalid manifests fail with actionable validation errors. |
| FR-4 | Orchestrator executes `agent` and `supervisor` nodes. | Fixed pipelines and supervisor conditional routing both run to completion with deterministic no-LLM fallback. |
| FR-5 | Adding an agent requires only a new plugin folder plus a node/edge in the central graph; framework code is unchanged. | A fourth plugin can be added without editing files outside `plugins/` and `orchestration.yaml`. |
| FR-6 | Orchestration state is checkpointed to SQLite and resumable by run id. | Interrupted runs can be resumed with the same run id; final state and node outputs are queryable. |
| FR-7 | Portal shows per-agent status: `healthy`, `stale`, `offline`. | Status updates reflect registry heartbeat state within one portal polling interval. |
| FR-8 | Portal shows orchestration topology and highlights the current/selected run path. | Nodes and edges render from the manifest; the selected run highlights visited nodes/edges. |
| FR-9 | Portal shows per-agent traffic: request volume, active tasks, success/error counts, and latency. | Traffic charts update from `/metrics` snapshots and support a simple time range. |
| FR-10 | All services use Python standard `logging` with JSON records; portal can filter logs by agent, level, time, run id, node, event, and text. | Structured records are ingestible by the portal backend and searchable through portal APIs/UI. |
| FR-11 | Portal can switch graph configuration files and apply the change without restarting services. | Selecting a graph and clicking Apply updates the active config; the next run carries that `graph_config`. |
| FR-12 | `POST /runs` accepts a run immediately and returns `status: running`; final output is available by polling `GET /runs/{id}`. | A POST response contains `run_id` and `graph_config`; a later GET returns completed outputs or a failed error. |
| FR-13 | Running jobs can be cancelled and timed out. | `POST /runs/{id}/cancel` marks the run failed; runs exceeding `RUN_TIMEOUT_SECONDS` are marked failed with a timeout error. |
| FR-14 | Portal job list can be filtered by graph configuration. | Selecting a config in the Orchestration page shows only jobs created with that config; an All option remains available. |

## 7. Non-Functional Requirements

- **Local only**: services bind to localhost in v1.
- **No authentication**: acceptable only for local development.
- **Polling**: portal refreshes every 2–5 seconds.
- **No external infrastructure**: everything runs in local Python processes plus SQLite.
- **LLM optional**: examples run without API keys.
- **Determinism**: no-LLM paths produce stable, testable output.
- **Observability**: every service writes stdout and local JSONL logs.
- **Resilience**: transient HTTP failures are retried; stale agents are excluded.
- **Run lifecycle**: default timeout is 120 seconds; active jobs block graph switching until they finish or are cancelled.
- **Maintainability**: Pydantic schemas are the single source of truth for wire contracts.

## 8. Acceptance Criteria / Definition of Done

The feature is complete when all of the following are true:

1. A developer can launch the local stack with a single documented command.
2. All three example agents register and appear healthy in the portal.
3. Submitting a run accepts immediately and, when polled, executes `researcher -> analyst -> writer` and produces a report.
4. The run path is visible in the orchestration view.
5. Each agent's traffic metrics update after a run.
6. Structured logs from each agent can be filtered and viewed in the portal.
7. Orchestration checkpoints exist in `state/orchestrator.db`.
8. Portal metrics/logs are persisted in `state/portal.db`.
9. A new agent can be added, launched, and included in a graph without restarting the orchestrator or portal.
10. Switching to `orchestration.llm.yaml` produces a supervisor topology and a run tagged with that config.
11. Cancel and timeout endpoints mark running jobs failed and reconcile stale running rows.
12. All unit, integration, frontend smoke, and no-LLM end-to-end tests pass.

## 9. Constraints and Dependencies

- Python 3.12 is used by the existing repository virtual environment.
- Existing `day1-agent-fundamentals/` and `day4-agentic-rag-tool-use/` folders must not be modified.
- Existing DeepSeek/OpenAI-compatible `ModelConfig` behavior is reused rather than replaced.
- SQLite checkpointing requires the separate `langgraph-checkpoint-sqlite` package.

## 10. References

- `day5-agent-orchestration/agent.md` — technical design and platform topology.
- `day5-agent-orchestration/iteration.md` — implementation iteration plan.
- Existing Day 4 A2A/LangGraph demo for task lifecycle and event bus concepts.
