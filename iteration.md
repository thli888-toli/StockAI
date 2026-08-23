# Day 5 Agent Orchestration — Iteration Plan

> Purpose: split implementation into three reviewable iterations. Each iteration must pass its own tests before the next begins.

## Global Definition of Done

At the end of Iteration 3, the local stack must:

- Run registry, agent services, orchestrator, portal backend, and portal UI.
- Support adding a new agent without restarting the orchestrator or portal.
- Expose agent status, orchestration topology/run path, per-agent traffic, and structured logs through the portal.
- Run fully without an LLM API key.
- Persist orchestration checkpoints and portal data in SQLite.
- Switch between fixed-pipeline and LLM-supervised graph configs from the portal.
- Track each run's source graph config, allow cancellation, and apply a run timeout.

## Iteration 1 — Core Framework and Orchestration Runtime

### Objective

Prove that a declarative graph can orchestrate multiple out-of-process FastAPI agents with SQLite checkpointing.

### Scope

- Create `day5-agent-orchestration/` and documentation placeholders.
- Implement shared Pydantic schemas.
- Implement registry and registry client.
- Implement generic async agent service with task lifecycle.
- Implement `agent.yaml` and `orchestration.yaml` parsing/validation.
- Implement LangGraph compiler for pipeline and supervisor nodes.
- Implement remote agent HTTP client with polling, timeout, retries, and TTL-aware lookup.
- Implement orchestrator run API and SQLite checkpointer.
- Add `researcher`, `analyst`, and `writer` example plugins.
- Add no-LLM deterministic fallback for all example agents.
- Add CLI commands for registry, agent, orchestrator, and local run.

### Deliverables

- `framework/schemas.py`
- `framework/registry.py`
- `framework/registry_client.py`
- `framework/agent_service.py`
- `framework/agent_client.py`
- `framework/graph_compiler.py`
- `framework/orchestrator.py`
- `framework/checkpoint.py`
- `framework/runner.py`
- `main.py`
- Three plugins under `plugins/`
- `config/orchestration.yaml`
- Unit and integration tests for registry, agent service, graph compiler, and run execution

### Key Acceptance Criteria

1. Three example agents run as separate processes.
2. A submitted run is accepted as `running`, then completes `researcher -> analyst -> writer` and returns a report.
3. Missing or invalid manifest fields produce actionable validation errors.
4. Agent tasks progress through `submitted`, `working`, and `completed`/`failed`.
5. Orchestration checkpoints exist in `state/orchestrator.db`.
6. Existing `day1-*` and `day4-*` folders are not modified.
7. Run summaries include `graph_config`.

## Iteration 2 — Observability and Portal Backend

### Objective

Make every service observable and give the portal backend a complete, queryable view of the system.

### Scope

- Add standard-library JSON logging setup and async log forwarding.
- Add in-process metrics counters/histograms and `/metrics` endpoints.
- Implement `portal_backend/` aggregation.
- Persist metrics snapshots, logs, and run summaries in `state/portal.db`.
- Implement portal APIs for agents, graph, runs, metrics, and logs.
- Add `/logs/ingest` and `/metrics/ingest`.
- Add portal backend tests for filtering, aggregation, and error cases.

### Deliverables

- `framework/logging.py`
- `framework/metrics.py`
- `portal_backend/app.py`
- `portal_backend/store.py`
- `portal_backend/aggregator.py`
- `portal_backend/api.py`
- JSON logging applied to registry, agent service, and orchestrator
- Backend integration tests

### Key Acceptance Criteria

1. Each service emits valid JSON logs.
2. Logs are searchable by agent, level, time, run id, node, event, and text.
3. Portal backend can return current status for all healthy agents.
4. Portal backend can return graph topology and recent runs.
5. Portal backend can return per-agent traffic time series.
6. Portal backend remains usable when an agent is stale or offline.
7. Portal SQLite stores run graph configs and supports filtering runs by config.

## Iteration 3 — React Portal UI and End-to-End Hardening

### Objective

Deliver the management portal UI and validate the full hot-plug workflow end to end.

### Scope

- Create Vite/React/TypeScript portal UI.
- Implement Dashboard, Orchestration, Traffic, and Logs views.
- Render topology with `@xyflow/react`.
- Highlight selected/current run path.
- Render traffic charts with `recharts`.
- Implement 2–5 second HTTP polling.
- Serve the built portal UI from the portal backend.
- Add frontend smoke tests with mocked portal APIs.
- Add end-to-end no-LLM test for adding a fourth agent without restarting anything.
- Finalize README and run instructions.

### Deliverables

- `portal-ui/` React application
- Portal backend static serving for `portal-ui/dist`
- Frontend smoke tests
- End-to-end hot-plug test
- `README.md`

### Key Acceptance Criteria

1. Portal displays healthy/stale/offline status for each agent.
2. Orchestration view renders nodes/edges and highlights the selected run path.
3. Traffic view shows request volume, latency, active tasks, and success/error ratio.
4. Logs view filters by agent, level, time, run id, node, event, and text.
5. A new agent appears in the portal without restarting orchestrator or portal.
6. Full stack runs with a single documented command.
7. Orchestration page can filter jobs by graph config and cancel running jobs.
8. Apply waits for active jobs, then switches graph configs and refreshes topology immediately.

## Cross-Iteration Dependencies

- Iteration 2 depends on Iteration 1's registry, agent service, and orchestrator.
- Iteration 3 depends on Iteration 2's portal backend APIs and SQLite store.
- No iteration modifies `day1-agent-fundamentals/` or `day4-agentic-rag-tool-use/`.

## Recommended Milestone Checkpoints

| Milestone | Completion Signal |
|-----------|-------------------|
| M1 | A local run completes through three HTTP agents with checkpointing. |
| M2 | Portal backend returns complete status, topology, metrics, and logs. |
| M3 | Portal UI is usable and hot-plug addition works end to end. |

## Implemented Post-Iteration Hardening

The following runtime fixes are now part of the delivered implementation:

- `RunSummary.graph_config` records which graph manifest created a run.
- `POST /runs` returns immediately with `status=running`; clients poll `GET /runs/{id}`.
- `POST /runs/{id}/cancel` cancels a running task and marks it failed.
- `RUN_TIMEOUT_SECONDS` wraps graph execution and marks timed-out runs failed.
- `POST /graph-configs/apply` rejects with `409` while active runs exist.
- Portal `/api/graph` and `/api/graph-configs` proxy the orchestrator live, eliminating stale topology after Apply.
- Portal job list supports a separate graph-config filter and shows config labels.
- Portal Apply auto-waits for active jobs before switching.
- Portal SQLite migrates existing `runs` tables with `graph_config` and reconciles stale running rows.
