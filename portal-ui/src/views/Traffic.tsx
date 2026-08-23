import { useEffect, useMemo, useState } from "react";
import { CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api";
import type { AgentStatus, MetricRow } from "../types";

export default function Traffic({ agents }: { agents: AgentStatus[] }) {
  const [agent, setAgent] = useState("");
  const [rows, setRows] = useState<MetricRow[]>([]);

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
        const data = await api.metrics(agent);
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
  }, [agent]);

  const metricRows = useMemo(() => {
    const prefix = `agent.${agent}.`;
    const value = (row: MetricRow, key: string) => row[`${prefix}${key}`] ?? row[key] ?? 0;
    return rows.map((row) => ({
      ...row,
      "requests.total": value(row, "requests.total"),
      "latency_ms.p50": value(row, "latency_ms.p50"),
      "requests.failed": value(row, "requests.failed")
    }));
  }, [agent, rows]);

  return (
    <section className="card">
      <h2>Agent Traffic</h2>
      <div className="controls">
        <select value={agent} onChange={(event) => setAgent(event.target.value)}>
          {agents.map((item) => (
            <option key={item.name} value={item.name}>
              {item.name}
            </option>
          ))}
        </select>
      </div>
      <ResponsiveContainer width="100%" height={360}>
        <LineChart data={metricRows}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="ts" hide />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="requests.total" stroke="#3b6fe0" dot={false} />
          <Line type="monotone" dataKey="latency_ms.p50" stroke="#e08c3b" dot={false} />
          <Line type="monotone" dataKey="requests.failed" stroke="#d64545" dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </section>
  );
}
