"""Standard-library JSON logging and optional portal log forwarding."""

from __future__ import annotations

import json
import logging
import queue
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": _now(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in (
            "service",
            "agent",
            "run_id",
            "node_id",
            "event",
        ):
            value = getattr(record, key, None)
            if value is not None:
                payload[key] = value
        extra = getattr(record, "extra", {})
        if isinstance(extra, dict) and extra:
            payload["extra"] = extra
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class _PortalLogForwarder(logging.Handler):
    """Buffers JSON records and posts them to the portal backend."""

    def __init__(self, portal_url: str, batch_size: int = 20, flush_seconds: float = 2.0) -> None:
        super().__init__()
        self.portal_url = portal_url.rstrip("/") + "/logs/ingest"
        self.batch_size = batch_size
        self.flush_seconds = flush_seconds
        self.queue: queue.Queue[str] = queue.Queue()
        self._client = httpx.Client(timeout=3.0)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.queue.put_nowait(self.format(record))
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        import time

        batch: list[str] = []
        while True:
            try:
                deadline = time.monotonic() + self.flush_seconds
                while len(batch) < self.batch_size:
                    timeout = max(0.0, deadline - time.monotonic())
                    try:
                        batch.append(self.queue.get(timeout=timeout))
                    except queue.Empty:
                        break
                if batch:
                    self._send(batch)
                    batch = []
            except Exception:  # noqa: BLE001
                continue

    def _send(self, records: list[str]) -> None:
        try:
            payload = [json.loads(record) for record in records]
            self._client.post(self.portal_url, json={"records": payload})
        except Exception:  # noqa: BLE001
            pass


def configure_service_logger(
    service: str,
    agent: str | None = None,
    log_dir: str | Path | None = "logs",
    portal_url: str | None = None,
) -> logging.Logger:
    """Configure the standard logger used by one service."""
    logger = logging.getLogger("agent_framework")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()

    formatter = JsonFormatter()

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path / "service.jsonl", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if portal_url:
        forwarder = _PortalLogForwarder(portal_url)
        forwarder.setFormatter(formatter)
        logger.addHandler(forwarder)

    # Stored on the logger for reuse by structured calls.
    setattr(logger, "service", service)
    setattr(logger, "agent", agent or service)
    return logger


def log_event(
    logger: logging.Logger,
    level: str,
    message: str,
    *,
    event: str,
    run_id: str | None = None,
    node_id: str | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    kwargs: dict[str, Any] = {"extra": {"extra": extra or {}}}
    for key, value in (
        ("service", getattr(logger, "service", None)),
        ("agent", getattr(logger, "agent", None)),
        ("run_id", run_id),
        ("node_id", node_id),
        ("event", event),
    ):
        if value is not None:
            kwargs["extra"][key] = value
    logger.log(getattr(logging, level.upper(), logging.INFO), message, **kwargs)
