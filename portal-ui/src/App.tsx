import { useEffect, useState } from "react";
import { api } from "./api";
import type { AgentStatus, RunSummary } from "./types";
import Dashboard from "./views/Dashboard";
import Logs from "./views/Logs";
import Orchestration from "./views/Orchestration";
import Traffic from "./views/Traffic";

type Tab = "dashboard" | "orchestration" | "traffic" | "logs";

const TABS: Array<{ id: Tab; label: string }> = [
  { id: "dashboard", label: "Agents" },
  { id: "orchestration", label: "Orchestration" },
  { id: "traffic", label: "Traffic" },
  { id: "logs", label: "Logs" }
];

export default function App() {
  const [tab, setTab] = useState<Tab>("dashboard");
  const [agents, setAgents] = useState<AgentStatus[]>([]);
  const [runs, setRuns] = useState<RunSummary[]>([]);

  useEffect(() => {
    let alive = true;
    const refresh = async () => {
      try {
        const [nextAgents, nextRuns] = await Promise.all([api.agents(), api.runs()]);
        if (alive) {
          setAgents(nextAgents);
          setRuns(nextRuns);
        }
      } catch {
        // Portal backend may be starting.
      }
    };
    refresh();
    const id = window.setInterval(refresh, 3000);
    return () => {
      alive = false;
      window.clearInterval(id);
    };
  }, []);

  return (
    <main className="app">
      <h1>Agent Orchestration Portal</h1>
      <nav className="tabs">
        {TABS.map((item) => (
          <button
            key={item.id}
            className={tab === item.id ? "active" : ""}
            onClick={() => setTab(item.id)}
          >
            {item.label}
          </button>
        ))}
      </nav>

      {tab === "dashboard" && <Dashboard agents={agents} />}
      {tab === "orchestration" && <Orchestration runs={runs} />}
      {tab === "traffic" && <Traffic agents={agents} />}
      {tab === "logs" && <Logs agents={agents} />}
    </main>
  );
}
