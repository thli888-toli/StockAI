import type { AgentStatus } from "../types";

export default function Dashboard({ agents }: { agents: AgentStatus[] }) {
  return (
    <section>
      <div className="card">
        <h2>Agent Status</h2>
        <div className="agent-grid">
          {agents.map((agent) => (
            <div className="card" key={agent.name}>
              <strong>{agent.name}</strong>
              <span className={`status ${agent.status}`}>{agent.status}</span>
              <p>{agent.description}</p>
              <small>{agent.endpoint}</small>
              <br />
              <small>Last heartbeat: {new Date(agent.last_heartbeat).toLocaleTimeString()}</small>
            </div>
          ))}
          {agents.length === 0 && <p>No agents discovered yet.</p>}
        </div>
      </div>
    </section>
  );
}
