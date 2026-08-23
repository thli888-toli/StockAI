import { useEffect, useState } from "react";
import { api } from "../api";
import type { AgentStatus, LogRecord } from "../types";

export default function Logs({ agents }: { agents: AgentStatus[] }) {
  const [agent, setAgent] = useState("");
  const [level, setLevel] = useState("");
  const [runId, setRunId] = useState("");
  const [q, setQ] = useState("");
  const [rows, setRows] = useState<LogRecord[]>([]);

  useEffect(() => {
    if (agents.length && !agent) {
      setAgent(agents[0].name);
    }
  }, [agents, agent]);

  useEffect(() => {
    if (!agent) return;
    let alive = true;
    const refresh = async () => {
      try {
        const params = new URLSearchParams();
        if (level) params.set("level", level);
        if (runId) params.set("run_id", runId);
        if (q) params.set("q", q);
        const data = await api.logs(agent, params);
        if (alive) setRows(data);
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
  }, [agent, level, runId, q]);

  return (
    <section className="card">
      <h2>Agent Logs</h2>
      <div className="controls">
        <select value={agent} onChange={(event) => setAgent(event.target.value)}>
          {agents.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name}
            </option>
          ))}
        </select>
        <select value={level} onChange={(event) => setLevel(event.target.value)}>
          <option value="">All levels</option>
          <option value="INFO">INFO</option>
          <option value="WARNING">WARNING</option>
          <option value="ERROR">ERROR</option>
        </select>
        <input
          value={runId}
          onChange={(event) => setRunId(event.target.value)}
          placeholder="Run id"
        />
        <input value={q} onChange={(event) => setQ(event.target.value)} placeholder="Search message" />
      </div>
      <table className="table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Level</th>
            <th>Run</th>
            <th>Node</th>
            <th>Event</th>
            <th>Message</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.id}>
              <td>{new Date(row.ts).toLocaleTimeString()}</td>
              <td>{row.level}</td>
              <td>{row.run_id ?? ""}</td>
              <td>{row.node_id ?? ""}</td>
              <td>{row.event ?? ""}</td>
              <td>{row.message}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
