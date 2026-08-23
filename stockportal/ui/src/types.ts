export type RunSummary = {
  run_id: string;
  graph_config?: string | null;
  status: "running" | "completed" | "failed";
  query: string;
  outputs: Record<string, unknown>;
  error?: string | null;
  created_at: string;
  updated_at: string;
};

export type RunStatus = "running" | "completed" | "failed";

export type WatchlistItem = {
  symbol: string;
  company_name: string;
  industry: string;
  run_id: string | null;
  status: RunStatus;
  error: string | null;
  outputs: Record<string, unknown>;
  created_at: string;
  updated_at: string;
};
