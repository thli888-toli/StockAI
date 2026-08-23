export type AgentStatus = {
  name: string;
  status: "healthy" | "stale" | "offline";
  last_heartbeat: string;
  endpoint: string;
  description: string;
};

export type RunEvent = {
  node?: string;
  event?: string;
  error?: string;
  decision?: string;
  output_key?: string;
};

export type RunSummary = {
  run_id: string;
  graph_config?: string | null;
  status: "running" | "completed" | "failed";
  query: string;
  outputs: Record<string, unknown>;
  error?: string | null;
  created_at: string;
  updated_at: string;
  events: RunEvent[];
};

export type GraphNodeSpec = {
  type: "agent" | "supervisor";
  agent?: string | null;
  input?: Record<string, string>;
  output_key?: string | null;
  prompt?: string | null;
  options?: string[];
};

export type GraphEdge = {
  from?: string;
  source?: string;
  to: string;
  when?: string | null;
};

export type GraphData = {
  name: string;
  entry: string;
  nodes: Record<string, GraphNodeSpec>;
  edges: GraphEdge[];
};

export type GraphConfig = {
  name: string;
  active: boolean;
};

export type LogRecord = {
  id: number;
  ts: string;
  level: string;
  service: string;
  agent: string;
  run_id?: string | null;
  node_id?: string | null;
  event?: string | null;
  message: string;
};

export type MetricRow = Record<string, string | number>;
