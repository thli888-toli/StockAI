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
  graph: (market: string = "a", manifest?: string) =>
    json<GraphData>(
      `/api/graph?market=${encodeURIComponent(market)}${
        manifest ? `&manifest=${encodeURIComponent(manifest)}` : ""
      }`
    ),
  graphConfigs: (market: string = "a") =>
    json<GraphConfig[]>(`/api/graph-configs?market=${encodeURIComponent(market)}`),
  applyGraphConfig: (name: string, market: string = "a") =>
    fetch("/api/graph-configs/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, market })
    }).then((response) => {
      if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
      return response.json() as Promise<GraphConfig[]>;
    }),
  cancelRun: (runId: string, market: string = "a") =>
    fetch(`/api/runs/${runId}/cancel?market=${encodeURIComponent(market)}`, {
      method: "POST"
    }).then((response) => {
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
