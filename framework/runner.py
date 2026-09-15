"""Local development process helpers."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import uvicorn

from framework.config import (
    CHECKPOINT_DB,
    PORTAL_BACKEND_URL,
    REGISTRY_URL,
    RUN_QUEUE_DB,
)


def _python() -> str:
    return sys.executable


def _service_log_path(root: Path, label: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("_") or "service"
    return root / "logs" / f"{safe}.log"


def _spawn_service(
    label: str,
    argv: list[str],
    root: Path,
    handles: list,
    commands: dict[str, list[str]] | None = None,
) -> tuple[str, subprocess.Popen]:
    """Start a child service with its output captured under logs/."""
    if commands is not None:
        commands[label] = list(argv)
    path = _service_log_path(root, label)
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a", encoding="utf-8", errors="replace")
    handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} {label} ===\n")
    handle.flush()
    handles.append(handle)
    process = subprocess.Popen(
        argv,
        cwd=root,
        stdout=handle,
        stderr=subprocess.STDOUT,
    )
    return label, process


def _tail_text(path: Path, lines: int = 15) -> str:
    if not path.exists():
        return ""
    try:
        content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:  # noqa: BLE001
        return ""
    return "\n".join(content[-lines:])


def _wait_for_registry(
    port: int,
    process: subprocess.Popen | None = None,
    timeout: float = 30.0,
) -> None:
    """Block until the local registry answers /health (or raise)."""
    import httpx

    deadline = time.time() + timeout
    url = f"http://127.0.0.1:{port}/health"
    last_error = ""
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(
                f"registry exited during startup with code {process.returncode}"
            )
        try:
            response = httpx.get(url, timeout=1.5)
            if response.status_code == 200:
                return
            last_error = f"HTTP {response.status_code}"
        except Exception as exc:  # noqa: BLE001
            last_error = str(exc)
        time.sleep(0.3)
    raise RuntimeError(
        f"registry did not become healthy at {url} within {timeout:g}s "
        f"(last error: {last_error})"
    )


def run_registry(host: str = "127.0.0.1", port: int = 8001) -> None:
    uvicorn.run("framework.registry:registry_app", host=host, port=port)


def run_orchestrator(
    manifest_path: str | Path,
    host: str = "127.0.0.1",
    port: int = 8020,
    checkpoint_db: str | Path = CHECKPOINT_DB,
    queue_db: str | Path = RUN_QUEUE_DB,
) -> None:
    os.environ.setdefault("REGISTRY_URL", REGISTRY_URL)
    app = __import__("framework.orchestrator", fromlist=["create_orchestrator_app"]).create_orchestrator_app(
        manifest_path,
        checkpoint_db=checkpoint_db,
        queue_db=queue_db,
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
    us_manifest_path: str = "config/orchestration_us.yaml",
    us_orchestrator_port: int = 8029,
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
        "plugins/us_data/agent.yaml",
        "plugins/us_fundamental/agent.yaml",
        "plugins/us_validator/agent.yaml",
    ]
    processes: list[tuple[str, subprocess.Popen]] = []
    log_handles: list = []
    commands: dict[str, list[str]] = {}
    try:
        registry_entry = _spawn_service(
            "registry",
            [_python(), "-m", "main", "registry"],
            root,
            log_handles,
            commands,
        )
        processes.append(registry_entry)
        registry_process = registry_entry[1]
        _wait_for_registry(registry_port, registry_process)
        for plugin in plugin_paths:
            processes.append(
                _spawn_service(
                    f"agent:{plugin}",
                    [_python(), "-m", "main", "agent", "--manifest", plugin],
                    root,
                    log_handles,
                    commands,
                )
            )
            time.sleep(0.8)
        processes.append(
            _spawn_service(
                f"orchestrator:{manifest_path}",
                [
                    _python(),
                    "-m",
                    "main",
                    "orchestrator",
                    "--manifest",
                    manifest_path,
                    "--port",
                    str(orchestrator_port),
                    "--checkpoint-db",
                    str(CHECKPOINT_DB),
                    "--queue-db",
                    str(RUN_QUEUE_DB),
                ],
                root,
                log_handles,
                commands,
            )
        )
        processes.append(
            _spawn_service(
                f"orchestrator_us:{us_manifest_path}",
                [
                    _python(),
                    "-m",
                    "main",
                    "orchestrator",
                    "--manifest",
                    us_manifest_path,
                    "--port",
                    str(us_orchestrator_port),
                    "--checkpoint-db",
                    "state/orchestrator_us.db",
                    "--queue-db",
                    "state/orchestrator_queue_us.db",
                ],
                root,
                log_handles,
                commands,
            )
        )
        processes.append(
            _spawn_service(
                "portal",
                [_python(), "-m", "main", "portal"],
                root,
                log_handles,
                commands,
            )
        )
        processes.append(
            _spawn_service(
                "stockportal",
                [_python(), "-m", "main", "stockportal"],
                root,
                log_handles,
                commands,
            )
        )
        print(
            "Started registry, CN+US agents, CN orchestrator (8020), US orchestrator (8029), "
            "monitoring portal, and stock portal. Press Ctrl+C to stop."
        )
        restarts: dict[str, int] = {}
        max_restarts = 2
        while True:
            for index, (label, proc) in enumerate(list(processes)):
                if proc.poll() is not None:
                    log_path = _service_log_path(root, label)
                    count = restarts.get(label, 0)
                    if count < max_restarts and label in commands:
                        restarts[label] = count + 1
                        print(
                            f"service '{label}' exited with code {proc.returncode}; "
                            f"restarting ({count + 1}/{max_restarts})",
                            flush=True,
                        )
                        processes[index] = _spawn_service(
                            label,
                            commands[label],
                            root,
                            log_handles,
                            commands,
                        )
                        continue
                    raise RuntimeError(
                        f"service '{label}' exited unexpectedly "
                        f"with code {proc.returncode}; log: {log_path}\n"
                        f"--- last log lines ---\n{_tail_text(log_path)}"
                    )
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for _, proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for _, proc in processes:
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        for handle in log_handles:
            try:
                handle.close()
            except Exception:  # noqa: BLE001
                pass
