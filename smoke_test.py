"""Local end-to-end smoke test: start the stack and submit one run."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import httpx


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def wait_health(url: str, timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = httpx.get(url, timeout=1.0)
            if response.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.2)
    raise RuntimeError(f"service did not become healthy: {url}")


def main() -> int:
    processes: list[subprocess.Popen] = []
    try:
        processes.append(subprocess.Popen([PYTHON, "-m", "main", "registry"], cwd=ROOT))
        wait_health("http://127.0.0.1:8001/health")

        for plugin in ["stock_data", "stock_news", "stock_quant", "stock_analyst"]:
            processes.append(
                subprocess.Popen(
                    [PYTHON, "-m", "main", "agent", "--manifest", f"plugins/{plugin}/agent.yaml"],
                    cwd=ROOT,
                )
            )
        for port in (8021, 8022, 8024, 8023):
            wait_health(f"http://127.0.0.1:{port}/health")

        processes.append(
            subprocess.Popen(
                [PYTHON, "-m", "main", "orchestrator", "--manifest", "config/orchestration.yaml"],
                cwd=ROOT,
            )
        )
        wait_health("http://127.0.0.1:8020/health")
        processes.append(subprocess.Popen([PYTHON, "-m", "main", "portal"], cwd=ROOT))
        wait_health("http://127.0.0.1:8030/health")

        with httpx.Client(timeout=30.0) as client:
            run_response = client.post(
                "http://127.0.0.1:8020/runs",
                json={"query": "600519"},
            )
            run_response.raise_for_status()
            run = run_response.json()
            deadline = time.monotonic() + 180.0
            while time.monotonic() < deadline:
                run = client.get(f"http://127.0.0.1:8020/runs/{run['run_id']}").json()
                if run["status"] in {"completed", "failed"}:
                    break
                time.sleep(0.2)
            assert run["status"] == "completed", run
            assert "report" in run["outputs"], run
            print(f"RUN OK: {run['run_id']} -> {run['status']}")
            print(run["outputs"]["report"])

            time.sleep(6.0)
            agents = client.get("http://127.0.0.1:8030/api/agents")
            agents.raise_for_status()
            assert agents.json(), "portal returned no agents"
            runs = client.get("http://127.0.0.1:8030/api/runs")
            runs.raise_for_status()
            assert any(item["run_id"] == run["run_id"] for item in runs.json())
            logs = client.get(
                f"http://127.0.0.1:8030/api/agents/stock_data/logs?run_id={run['run_id']}"
            )
            logs.raise_for_status()
            assert logs.json(), "portal returned no logs for the run"
            print("PORTAL OK: agents, runs, and logs are visible")

        return 0
    finally:
        for proc in reversed(processes):
            if proc.poll() is None:
                proc.terminate()
        for proc in reversed(processes):
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()


if __name__ == "__main__":
    raise SystemExit(main())
