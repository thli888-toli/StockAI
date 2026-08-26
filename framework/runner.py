"""Local development process helpers."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import uvicorn

from framework.config import CHECKPOINT_DB, PORTAL_BACKEND_URL, REGISTRY_URL


def _python() -> str:
    return sys.executable


def run_registry(host: str = "127.0.0.1", port: int = 8001) -> None:
    uvicorn.run("framework.registry:registry_app", host=host, port=port)


def run_orchestrator(
    manifest_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8020,
) -> None:
    os.environ.setdefault("REGISTRY_URL", REGISTRY_URL)
    os.environ.setdefault("CHECKPOINT_DB", CHECKPOINT_DB)
    app = __import__("framework.orchestrator", fromlist=["create_orchestrator_app"]).create_orchestrator_app(
        manifest_path
    )
    uvicorn.run(app, host=host, port=port)


def run_agent(manifest_path: str | Path, host: str = "127.0.0.1") -> None:
    manifest_path = Path(manifest_path).resolve()
    manifest_dir = manifest_path.parent
    plugin_name = manifest_dir.name
    sys.path.insert(0, str(manifest_dir.parent.parent))
    port = _manifest_port(manifest_path)
    app = __import__(
        "framework.agent_service", fromlist=["create_agent_app"]
    ).create_agent_app(manifest_path, registry_url=REGISTRY_URL)
    uvicorn.run(app, host=host, port=port)


def run_portal_backend(host: str = "127.0.0.1", port: int = 8030) -> None:
    from portal_backend.app import create_portal_app

    app = create_portal_app()
    uvicorn.run(app, host=host, port=port)


def run_stock_portal(host: str = "127.0.0.1", port: int = 8040) -> None:
    from stockportal.app import create_stock_portal_app

    app = create_stock_portal_app()
    uvicorn.run(app, host=host, port=port)


def _manifest_port(path: Path) -> int:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return int(data.get("port", 8011))


def start_all(
    registry_port: int = 8001,
    orchestrator_port: int = 8020,
    manifest_path: str = "config/orchestration.yaml",
    plugin_paths: list[str] | None = None,
) -> None:
    """Launch registry, agent services, and orchestrator as separate processes."""
    root = Path.cwd()
    plugin_paths = plugin_paths or [
        "plugins/stock_data/agent.yaml",
        "plugins/stock_news/agent.yaml",
        "plugins/stock_quant/agent.yaml",
        "plugins/stock_fundamental/agent.yaml",
        "plugins/stock_analyst/agent.yaml",
    ]
    processes: list[subprocess.Popen] = []
    try:
        processes.append(
            subprocess.Popen(
                [_python(), "-m", "main", "registry"],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        )
        time.sleep(0.3)
        for plugin in plugin_paths:
            processes.append(
                subprocess.Popen(
                    [_python(), "-m", "main", "agent", "--manifest", plugin],
                    cwd=root,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.STDOUT,
                )
            )
        processes.append(
            subprocess.Popen(
                [_python(), "-m", "main", "orchestrator", "--manifest", manifest_path],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        )
        processes.append(
            subprocess.Popen(
                [_python(), "-m", "main", "portal"],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        )
        processes.append(
            subprocess.Popen(
                [_python(), "-m", "main", "stockportal"],
                cwd=root,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
            )
        )
        print("Started registry, agents, orchestrator, monitoring portal, and stock portal. Press Ctrl+C to stop.")
        while True:
            for proc in processes:
                if proc.poll() is not None:
                    raise RuntimeError("a service exited unexpectedly")
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
