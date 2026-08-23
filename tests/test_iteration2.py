from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.logging import JsonFormatter, configure_service_logger  # noqa: E402
from framework.metrics import MetricsRegistry  # noqa: E402
from portal_backend.app import create_portal_app  # noqa: E402
from portal_backend.store import PortalStore  # noqa: E402


def test_json_formatter_contains_standard_fields():
    logger = logging.getLogger("test_json")
    logger.handlers.clear()
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    record = logger.makeRecord(
        "test_json",
        logging.INFO,
        "fn",
        1,
        "hello",
        (),
        None,
    )
    record.service = "researcher"
    record.agent = "researcher"
    record.run_id = "r1"
    record.node_id = "researcher"
    record.event = "task.completed"
    record.extra = {"a": 1}
    payload = json.loads(handler.format(record))

    assert payload["level"] == "INFO"
    assert payload["message"] == "hello"
    assert payload["service"] == "researcher"
    assert payload["run_id"] == "r1"
    assert payload["extra"] == {"a": 1}


def test_metrics_registry_counters_and_percentiles():
    metrics = MetricsRegistry(prefix="agent.researcher.")
    metrics.inc("requests.total", 2)
    metrics.inc("requests.completed", 2)
    metrics.observe_latency("latency_ms", 0.1)
    metrics.observe_latency("latency_ms", 0.3)
    payload = metrics.to_json()

    assert payload["agent.researcher.requests.total"] == 2
    assert payload["agent.researcher.latency_ms.p50"] == 0.1
    assert payload["agent.researcher.latency_ms.p95"] == 0.3
    assert payload["agent.researcher.latency_ms.max"] == 0.3


def test_portal_log_ingest_and_filters(tmp_path):
    app = create_portal_app(db_path=tmp_path / "portal.db")
    with TestClient(app) as client:
        response = client.post(
            "/logs/ingest",
            json={
                "records": [
                    {
                        "timestamp": "2026-08-22T12:00:00Z",
                        "level": "INFO",
                        "service": "researcher",
                        "agent": "researcher",
                        "run_id": "r1",
                        "node_id": "researcher",
                        "event": "task.completed",
                        "message": "task completed",
                        "extra": {},
                    },
                    {
                        "timestamp": "2026-08-22T12:00:01Z",
                        "level": "ERROR",
                        "service": "analyst",
                        "agent": "analyst",
                        "event": "task.failed",
                        "message": "boom",
                        "extra": {},
                    },
                ]
            },
        )
        assert response.status_code == 200

        logs = client.get("/api/agents/researcher/logs?level=INFO&run_id=r1").json()
        assert len(logs) == 1
        assert logs[0]["message"] == "task completed"

        all_logs = client.get("/api/agents/analyst/logs").json()
        assert len(all_logs) == 1
        assert all_logs[0]["level"] == "ERROR"

        related_logs = client.get("/api/logs?run_id=r1").json()
        assert [row["agent"] for row in related_logs] == ["researcher"]
        assert all(row["run_id"] == "r1" for row in related_logs)

        all_agents = client.get("/api/logs").json()
        assert {row["agent"] for row in all_agents} == {"researcher", "analyst"}


def test_portal_metrics_ingest_and_query(tmp_path):
    app = create_portal_app(db_path=tmp_path / "portal.db")
    with TestClient(app) as client:
        client.post(
            "/metrics/ingest",
            json={"service": "researcher", "metrics": {"requests.total": 3}},
        )
        rows = client.get("/api/agents/researcher/metrics").json()
        assert rows[-1]["requests.total"] == 3


def test_portal_store_filters_runs_by_graph_config(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    store.upsert_run(
        {
            "run_id": "r1",
            "graph_config": "orchestration.yaml",
            "status": "completed",
            "query": "q",
            "outputs": {},
            "events": [],
        }
    )
    store.upsert_run(
        {
            "run_id": "r2",
            "graph_config": "orchestration.llm.yaml",
            "status": "completed",
            "query": "q",
            "outputs": {},
            "events": [],
        }
    )

    assert len(store.list_runs()) == 2
    filtered = store.list_runs(graph_config="orchestration.llm.yaml")
    assert [run["run_id"] for run in filtered] == ["r2"]
