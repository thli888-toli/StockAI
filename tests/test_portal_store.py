from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from portal_backend.store import PortalStore  # noqa: E402


def _run(run_id: str, status: str, updated_at: str) -> dict:
    return {
        "run_id": run_id,
        "graph_config": "orchestration.yaml",
        "status": status,
        "query": "600519",
        "outputs": {},
        "error": None,
        "created_at": "2026-08-26T00:00:00+00:00",
        "updated_at": updated_at,
        "events": [],
    }


def test_orphaned_running_run_is_marked_failed(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    store.upsert_run(_run("r1", "running", "2026-08-26T00:00:00+00:00"))
    store.mark_orphaned_running_runs_failed(known_run_ids={"r2"})
    run = store.get_run("r1")
    assert run["status"] == "failed"
    assert run["error"] == "run no longer present in orchestrator"


def test_running_run_still_tracked_is_not_marked_failed(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    # An old "running" row whose run is still known to the orchestrator must
    # stay running regardless of age (long-running pipelines exceed any TTL).
    store.upsert_run(
        _run("r1", "running", "2026-08-26T00:00:00+00:00")
    )
    store.mark_orphaned_running_runs_failed(known_run_ids={"r1"})
    assert store.get_run("r1")["status"] == "running"


def test_completed_and_failed_runs_are_untouched(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    store.upsert_run(_run("r1", "completed", "2026-08-26T00:00:00+00:00"))
    store.upsert_run(_run("r2", "failed", "2026-08-26T00:00:00+00:00"))
    store.mark_orphaned_running_runs_failed(known_run_ids=set())
    assert store.get_run("r1")["status"] == "completed"
    assert store.get_run("r2")["status"] == "failed"


def test_portal_store_keeps_graphs_per_market(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    store.set_graph("a", {"name": "stock_analysis", "nodes": {}, "edges": []})
    store.set_graph("us", {"name": "us_stock_analysis", "nodes": {}, "edges": []})
    store.set_graph_configs("a", [{"name": "orchestration.yaml", "active": True}])
    store.set_graph_configs(
        "us",
        [{"name": "orchestration_us.yaml", "active": True}],
    )
    assert store.get_graph("a")["name"] == "stock_analysis"
    assert store.get_graph("us")["name"] == "us_stock_analysis"
    assert [item["name"] for item in store.get_graph_configs("us")] == [
        "orchestration_us.yaml"
    ]


def test_portal_store_runs_carry_market(tmp_path):
    store = PortalStore(tmp_path / "portal.db")
    store.upsert_run({**_run("r-us", "running", "2026-08-26T00:00:00+00:00"), "market": "us"})
    store.mark_orphaned_running_runs_failed(known_run_ids={"r-us"}, market="us")
    assert store.get_run("r-us")["status"] == "running"
    store.mark_orphaned_running_runs_failed(known_run_ids=set(), market="us")
    assert store.get_run("r-us")["status"] == "failed"
