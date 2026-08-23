import type { AgentStatus, GraphConfig, GraphData, LogRecord, MetricRow, RunSummary } from "./types";

const json = async <T>(url: string): Promise<T> => {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
};

export const api = {
  agents: () => json<AgentStatus[]>("/api/agents"),
  graph: () => json<GraphData>("/api/graph"),
  graphConfigs: () => json<GraphConfig[]>("/api/graph-configs"),
  applyGraphConfig: (name: string) =>
    fetch("/api/graph-configs/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name })
    }).then((response) => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json() as Promise<GraphConfig[]>;
    }),
  cancelRun: (runId: string) =>
    fetch(`/api/runs/${runId}/cancel`, { method: "POST" }).then((response) => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json() as Promise<RunSummary>;
    }),
  runs: () => json<RunSummary[]>("/api/runs"),
  run: (runId: string) => json<RunSummary>(`/api/runs/${runId}`),
  metrics: (agent: string) => json<MetricRow[]>(`/api/agents/${agent}/metrics`),
  logs: (agent: string, params: URLSearchParams) =>
    json<LogRecord[]>(`/api/agents/${agent}/logs?${params.toString()}`),
  allLogs: (params: URLSearchParams) =>
    json<LogRecord[]>(`/api/logs?${params.toString()}`)
};
