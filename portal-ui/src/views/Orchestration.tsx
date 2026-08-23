import { useEffect, useMemo, useState } from "react";
import { ReactFlow, Background, Controls, MarkerType } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { api } from "../api";
import type { GraphConfig, GraphData, LogRecord, RunSummary } from "../types";

function sourceOf(edge: { from?: string; source?: string }) {
  return edge.from ?? edge.source ?? "";
}

function layoutGraph(graph: GraphData) {
  const levels: Record<string, number> = {};
  const queue = [graph.entry];
  levels[graph.entry] = 0;

  while (queue.length) {
    const current = queue.shift()!;
    for (const edge of graph.edges) {
      if (sourceOf(edge) !== current || edge.to === "END") continue;
      if (levels[edge.to] === undefined) {
        levels[edge.to] = levels[current] + 1;
        queue.push(edge.to);
      }
    }
  }

  Object.keys(graph.nodes).forEach((id) => {
    if (levels[id] === undefined) levels[id] = 0;
  });

  const byLevel: Record<number, string[]> = {};
  Object.entries(levels).forEach(([id, level]) => {
    if (!graph.nodes[id]) return;
    (byLevel[level] ??= []).push(id);
  });

  const positions: Record<string, { x: number; y: number }> = {};
  Object.entries(byLevel).forEach(([level, ids]) => {
    ids.forEach((id, index) => {
      positions[id] = { x: Number(level) * 230, y: index * 130 };
    });
  });

  const maxLevel = Math.max(0, ...Object.values(levels));
  if (graph.edges.some((edge) => edge.to === "END")) {
    positions["END"] = { x: (maxLevel + 1) * 230, y: 0 };
  }

  return positions;
}

export default function Orchestration({ runs }: { runs: RunSummary[] }) {
  const [graph, setGraph] = useState<GraphData | null>(null);
  const [configs, setConfigs] = useState<GraphConfig[]>([]);
  const [selectedConfig, setSelectedConfig] = useState("");
  const [applying, setApplying] = useState(false);
  const [waiting, setWaiting] = useState(false);
  const [selectedRunId, setSelectedRunId] = useState<string>("");
  const [runFilter, setRunFilter] = useState("");
  const [showRunLogs, setShowRunLogs] = useState(false);
  const [runLogs, setRunLogs] = useState<LogRecord[]>([]);
  const filteredRuns = useMemo(
    () => (runFilter ? runs.filter((run) => run.graph_config === runFilter) : runs),
    [runFilter, runs]
  );
  const selectedRun =
    filteredRuns.find((run) => run.run_id === selectedRunId) ?? filteredRuns[0];
  const activeRunId = selectedRun?.run_id ?? "";

  useEffect(() => {
    setShowRunLogs(false);
    setRunLogs([]);
  }, [selectedRunId]);

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const [data, nextConfigs] = await Promise.all([api.graph(), api.graphConfigs()]);
        if (alive) {
          setGraph(data);
          setConfigs(nextConfigs);
          const active = nextConfigs.find((config) => config.active);
          if (active) setSelectedConfig((previous) => previous || active.name);
        }
      } catch {
        // Backend may be unavailable.
      }
    };
    refresh();
    const id = window.setInterval(refresh, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    if (!showRunLogs || !activeRunId) return;
    let alive = true;
    const refresh = async () => {
      try {
        const params = new URLSearchParams();
        params.set("run_id", activeRunId);
        const data = await api.allLogs(params);
        if (alive) setRunLogs(data);
      } catch {
        // Backend may be unavailable.
      }
    };
    refresh();
    const id = window.setInterval(refresh, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, [showRunLogs, activeRunId]);

  const applyConfig = async () => {
    if (!selectedConfig) return;
    setWaiting(true);
    try {
      const deadline = Date.now() + 120_000;
      while (Date.now() < deadline) {
        const latestRuns = await api.runs();
        if (!latestRuns.some((run) => run.status === "running")) break;
        await new Promise((resolve) => window.setTimeout(resolve, 2000));
      }
      if ((await api.runs()).some((run) => run.status === "running")) {
        throw new Error("Timed out waiting for running jobs to complete");
      }
    } catch (error) {
      console.error(error);
      setWaiting(false);
      return;
    }
    setWaiting(false);
    setApplying(true);
    try {
      const nextConfigs = await api.applyGraphConfig(selectedConfig);
      setConfigs(nextConfigs);
      const data = await api.graph();
      setGraph(data);
    } catch (error) {
      console.error(error);
    } finally {
      setApplying(false);
    }
  };

  const cancelSelectedRun = async () => {
    if (!selectedRun) return;
    try {
      await api.cancelRun(selectedRun.run_id);
    } catch (error) {
      console.error(error);
    }
  };

  const visited = useMemo(() => {
    if (!selectedRun) return new Set<string>();
    return new Set(
      selectedRun.events
        .map((event) => event.node)
        .filter((node): node is string => Boolean(node))
    );
  }, [selectedRun]);

  const flowNodes = useMemo(() => {
    if (!graph) return [];
    const positions = layoutGraph(graph);
    const ids = Object.keys(graph.nodes);
    if (positions["END"]) ids.push("END");
    return ids.map((id) => ({
      id,
      data: {
        label: id === "END" ? "END" : graph.nodes[id].agent ? `${id} (${graph.nodes[id].agent})` : id
      },
      position: positions[id] ?? { x: 0, y: 0 },
      style: {
        background: visited.has(id) ? "#dff7e8" : "#ffffff",
        border: visited.has(id) ? "2px solid #157347" : "1px solid #9aa1ad"
      }
    }));
  }, [graph, visited]);

  const flowEdges = useMemo(() => {
    if (!graph) return [];
    return graph.edges.map((edge, index) => ({
      id: `${sourceOf(edge)}-${edge.to}-${index}`,
      source: sourceOf(edge),
      target: edge.to === "END" ? "END" : edge.to,
      label: edge.when || undefined,
      markerEnd: { type: MarkerType.ArrowClosed }
    }));
  }, [graph]);

  return (
    <section className="card">
      <h2>Orchestration Graph</h2>
      <div className="controls">
        <select value={selectedConfig} onChange={(event) => setSelectedConfig(event.target.value)}>
          {configs.map((config) => (
            <option key={config.name} value={config.name}>
              {config.name} {config.active ? "(active)" : ""}
            </option>
          ))}
        </select>
        <button disabled={!selectedConfig || applying || waiting} onClick={applyConfig}>
          {waiting ? "Waiting for jobs..." : applying ? "Applying..." : "Apply"}
        </button>
      </div>
      <div className="controls">
        <select value={runFilter} onChange={(event) => setRunFilter(event.target.value)}>
          <option value="">All configurations</option>
          {configs.map((config) => (
            <option key={config.name} value={config.name}>
              {config.name}
            </option>
          ))}
        </select>
        <select value={selectedRun?.run_id ?? ""} onChange={(event) => setSelectedRunId(event.target.value)}>
          <option value="">Select job</option>
          {filteredRuns.map((run) => (
            <option key={run.run_id} value={run.run_id}>
              {run.run_id.slice(0, 8)} — {run.status}
              {run.graph_config ? ` — ${run.graph_config}` : ""}
            </option>
          ))}
        </select>
        {selectedRun && (
          <button onClick={() => setShowRunLogs((current) => !current)}>
            {showRunLogs ? "Hide logs" : "View logs"}
          </button>
        )}
        {selectedRun?.status === "running" && (
          <button onClick={cancelSelectedRun}>Cancel selected job</button>
        )}
      </div>
      {graph ? (
        <div className="flow">
          <ReactFlow nodes={flowNodes} edges={flowEdges} fitView>
            <Background />
            <Controls />
          </ReactFlow>
        </div>
      ) : (
        <p>Graph not available.</p>
      )}
      {showRunLogs && (
        <div className="run-logs">
          <h3>Logs for job {selectedRun?.run_id.slice(0, 8)}</h3>
          {runLogs.length ? (
            <table className="table">
              <thead>
                <tr>
                  <th>Time</th>
                  <th>Agent</th>
                  <th>Level</th>
                  <th>Node</th>
                  <th>Event</th>
                  <th>Message</th>
                </tr>
              </thead>
              <tbody>
                {runLogs.map((row) => (
                  <tr key={row.id}>
                    <td>{new Date(row.ts).toLocaleTimeString()}</td>
                    <td>{row.agent}</td>
                    <td>{row.level}</td>
                    <td>{row.node_id ?? ""}</td>
                    <td>{row.event ?? ""}</td>
                    <td>{row.message}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p>No logs for the selected job.</p>
          )}
        </div>
      )}
    </section>
  );
}
