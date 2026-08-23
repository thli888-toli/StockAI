"""Small in-process metrics registry for request/latency monitoring."""

from __future__ import annotations

import math
from collections import defaultdict
from time import perf_counter
from typing import Any


class MetricsRegistry:
    def __init__(self, prefix: str = "") -> None:
        self.prefix = prefix
        self.counters: dict[str, int] = defaultdict(int)
        self.latencies: dict[str, list[float]] = defaultdict(list)
        self.active = 0

    def inc(self, name: str, amount: int = 1) -> None:
        self.counters[f"{self.prefix}{name}"] += amount

    def set_active(self, value: int) -> None:
        self.active = value

    def observe_latency(self, name: str, elapsed: float) -> None:
        self.latencies[f"{self.prefix}{name}"].append(elapsed)

    def _percentile(self, values: list[float], pct: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = max(0, min(len(ordered) - 1, math.ceil((pct / 100.0) * len(ordered)) - 1))
        return round(ordered[index], 4)

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = dict(self.counters)
        payload[f"{self.prefix}active"] = self.active
        for key, values in self.latencies.items():
            payload[f"{key}.p50"] = self._percentile(values, 50)
            payload[f"{key}.p95"] = self._percentile(values, 95)
            payload[f"{key}.max"] = round(max(values), 4) if values else 0.0
        return payload


class Timer:
    def __init__(self, registry: MetricsRegistry, metric_name: str) -> None:
        self.registry = registry
        self.metric_name = metric_name
        self.start = perf_counter()

    def stop(self) -> float:
        elapsed = perf_counter() - self.start
        self.registry.observe_latency(self.metric_name, elapsed)
        return elapsed
