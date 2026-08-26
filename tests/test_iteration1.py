from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest
import yaml
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from framework.graph_compiler import GraphCompiler  # noqa: E402
from framework.orchestrator import Orchestrator, create_orchestrator_app  # noqa: E402
from framework.registry import create_registry_app  # noqa: E402
from framework.schemas import EdgeSpec, GraphManifest, NodeSpec, RunSummary, TaskRequest  # noqa: E402


class FakeAgentClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, TaskRequest]] = []

    async def run(self, agent_name: str, request: TaskRequest) -> str:
        self.calls.append((agent_name, request))
        return f"{agent_name}:{request.query}"


def pipeline_manifest() -> GraphManifest:
    return GraphManifest(
        name="pipeline",
        entry="a",
        nodes={
            "a": NodeSpec(type="agent", agent="a", input={"query": "{query}"}, output_key="a_out"),
            "b": NodeSpec(type="agent", agent="b", input={"query": "{query}", "context": "{a_out}"}, output_key="b_out"),
        },
        edges=[
            EdgeSpec(source="a", to="b"),
            EdgeSpec(source="b", to="END"),
        ],
    )


def supervisor_manifest() -> GraphManifest:
    return GraphManifest(
        name="supervisor",
        entry="super",
        nodes={
            "super": NodeSpec(type="supervisor", options=["worker", "finish"]),
            "worker": NodeSpec(type="agent", agent="worker", input={"query": "{query}"}, output_key="result"),
        },
        edges=[
            EdgeSpec(source="super", to="worker", when="worker"),
            EdgeSpec(source="super", to="END", when="finish"),
            EdgeSpec(source="worker", to="super"),
        ],
    )


@pytest.mark.asyncio
async def test_pipeline_graph_executes_nodes_in_order():
    client = FakeAgentClient()
    graph = GraphCompiler(client).compile(pipeline_manifest())
    result = await graph.ainvoke(
        {"run_id": "r1", "query": "hello", "outputs": {}, "steps": 0, "events": []}
    )

    assert result["outputs"] == {
        "a_out": "a:hello",
        "b_out": "b:hello",
    }
    assert [name for name, _ in client.calls] == ["a", "b"]


@pytest.mark.asyncio
async def test_supervisor_routes_then_finishes():
    client = FakeAgentClient()
    graph = GraphCompiler(client, max_steps=10).compile(supervisor_manifest())
    result = await graph.ainvoke(
        {"run_id": "r2", "query": "hello", "outputs": {}, "next": "", "steps": 0, "events": []}
    )

    assert result["outputs"]["result"] == "worker:hello"
    assert [name for name, _ in client.calls] == ["worker"]


def test_graph_manifest_rejects_unknown_edge_target():
    with pytest.raises(ValueError):
        GraphManifest(
            name="bad",
            entry="a",
            nodes={"a": NodeSpec(type="agent", agent="a")},
            edges=[EdgeSpec(source="a", to="missing")],
        )


def test_registry_health_and_unknown_agent():
    app = create_registry_app(ttl_seconds=5, stale_after_seconds=20)
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/agents/missing").status_code == 404


def test_orchestrator_lists_and_applies_graph_configs(tmp_path):
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "a_graph",
                "entry": "a",
                "nodes": {"a": {"type": "agent", "agent": "a"}},
                "edges": [{"from": "a", "to": "END"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "b_graph",
                "entry": "b",
                "nodes": {"b": {"type": "agent", "agent": "b"}},
                "edges": [{"from": "b", "to": "END"}],
            }
        ),
        encoding="utf-8",
    )
    orchestrator = Orchestrator(tmp_path / "a.yaml", queue_db=tmp_path / "queue.db")
    configs = orchestrator.list_graph_configs()
    assert {item["name"] for item in configs} == {"a.yaml", "b.yaml"}
    assert next(item for item in configs if item["name"] == "a.yaml")["active"] is True

    orchestrator.set_active_manifest("b.yaml")
    assert orchestrator.active_manifest == "b.yaml"
    assert orchestrator.load_manifest().name == "b_graph"


def test_orchestrator_apply_endpoint_accepts_body(tmp_path):
    (tmp_path / "a.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "a_graph",
                "entry": "a",
                "nodes": {"a": {"type": "agent", "agent": "a"}},
                "edges": [{"from": "a", "to": "END"}],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "b.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": "b_graph",
                "entry": "b",
                "nodes": {"b": {"type": "agent", "agent": "b"}},
                "edges": [{"from": "b", "to": "END"}],
            }
        ),
        encoding="utf-8",
    )
    app = create_orchestrator_app(
        tmp_path / "a.yaml",
        checkpoint_db=tmp_path / "orchestrator.db",
        queue_db=tmp_path / "queue.db",
    )
    with TestClient(app) as client:
        response = client.post(
            "/graph-configs/apply",
            json={"name": "b.yaml"},
        )
        assert response.status_code == 200
        configs = response.json()
        assert next(item for item in configs if item["name"] == "b.yaml")["active"] is True
        assert client.get("/graph").json()["entry"] == "b"


@pytest.mark.asyncio
async def test_supervisor_uses_llm_when_configured(monkeypatch):
    manifest = GraphManifest(
        name="llm_supervisor",
        entry="super",
        nodes={
            "super": NodeSpec(
                type="supervisor",
                prompt="Decide the next step.",
                options=["worker", "finish"],
            ),
            "worker": NodeSpec(type="agent", agent="worker", input={"query": "{query}"}, output_key="result"),
        },
        edges=[
            EdgeSpec(source="super", to="worker", when="worker"),
            EdgeSpec(source="super", to="END", when="finish"),
            EdgeSpec(source="worker", to="super"),
        ],
    )

    import framework.graph_compiler as graph_compiler

    monkeypatch.setattr(graph_compiler, "llm_configured", lambda: True)
    async def fake_llm(system: str, user: str, max_tokens: int = 10) -> str:
        return "worker"

    monkeypatch.setattr(graph_compiler, "llm_reply", fake_llm)
    client = FakeAgentClient()
    graph = GraphCompiler(client, max_steps=10).compile(manifest)
    result = await graph.ainvoke(
        {"run_id": "r3", "query": "hello", "outputs": {}, "next": "", "steps": 0, "events": []}
    )

    assert result["outputs"]["result"] == "worker:hello"
    assert [name for name, _ in client.calls] == ["worker"]


def test_run_summary_includes_graph_config():
    summary = RunSummary(run_id="r1", graph_config="orchestration.llm.yaml", query="q")
    data = summary.model_dump()
    assert data["graph_config"] == "orchestration.llm.yaml"


def _write_manifest(tmp_path, name: str):
    path = tmp_path / f"{name}.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "name": name,
                "entry": "a",
                "nodes": {"a": {"type": "agent", "agent": "a"}},
                "edges": [{"from": "a", "to": "END"}],
            }
        ),
        encoding="utf-8",
    )
    return path


@pytest.mark.asyncio
async def test_orchestrator_marks_run_failed_on_timeout(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, "a")
    orchestrator = Orchestrator(manifest, queue_db=tmp_path / "queue.db", run_timeout=0.05)

    class HangingClient:
        async def run(self, agent_name, request):
            await asyncio.sleep(1)
            return "never"

    orchestrator.agent_client = HangingClient()
    summary = orchestrator.start_run("q", run_id="timeout_run")
    await orchestrator._process_queue()
    task = orchestrator._run_tasks[summary.run_id]
    await task

    final = orchestrator.get_run("timeout_run")
    assert final.status == "failed"
    assert "timed out" in final.error
    await orchestrator.close()


@pytest.mark.asyncio
async def test_orchestrator_cancels_running_run(tmp_path, monkeypatch):
    manifest = _write_manifest(tmp_path, "a")
    orchestrator = Orchestrator(manifest, queue_db=tmp_path / "queue.db", run_timeout=30)

    class HangingClient:
        async def run(self, agent_name, request):
            await asyncio.sleep(10)
            return "never"

    orchestrator.agent_client = HangingClient()
    summary = orchestrator.start_run("q", run_id="cancel_run")
    await orchestrator._process_queue()
    await asyncio.sleep(0.01)
    cancelled = await orchestrator.cancel_run(summary.run_id)
    assert cancelled.status == "failed"
    assert cancelled.error == "cancelled"
    await orchestrator.close()


@pytest.mark.asyncio
async def test_apply_graph_config_rejects_active_runs(tmp_path, monkeypatch):
    a_path = _write_manifest(tmp_path, "a")
    b_path = _write_manifest(tmp_path, "b")
    orchestrator = Orchestrator(a_path, queue_db=tmp_path / "queue.db", run_timeout=30)

    class HangingClient:
        async def run(self, agent_name, request):
            await asyncio.sleep(10)
            return "never"

    orchestrator.agent_client = HangingClient()
    orchestrator.start_run("q", run_id="active")
    with pytest.raises(ValueError, match="active"):
        orchestrator.apply_graph_config(b_path.name)
    await orchestrator.cancel_run("active")
    await orchestrator.close()


@pytest.mark.asyncio
async def test_concurrent_runs_share_checkpointer_and_complete(tmp_path):
    manifest = _write_manifest(tmp_path, "a")
    orchestrator = Orchestrator(
        manifest,
        checkpoint_db=tmp_path / "orchestrator.db",
        queue_db=tmp_path / "queue.db",
        run_timeout=10,
    )
    orchestrator.agent_client = FakeAgentClient()
    try:
        first = orchestrator.start_run("q1", run_id="concurrent-1")
        second = orchestrator.start_run("q2", run_id="concurrent-2")
        await orchestrator._process_queue()
        await asyncio.gather(
            orchestrator._run_tasks[first.run_id],
            orchestrator._run_tasks[second.run_id],
        )
        assert orchestrator.get_run("concurrent-1").status == "completed"
        assert orchestrator.get_run("concurrent-2").status == "completed"
        assert orchestrator._checkpointer is not None
    finally:
        await orchestrator.close()
