"""Compile a GraphManifest into a LangGraph state graph."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from framework.agent_client import RemoteAgentClient
from framework.llm import llm_configured, llm_reply
from framework.schemas import EdgeSpec, GraphManifest, NodeSpec, OrchestrationState, TaskRequest


def _targets_for(manifest: GraphManifest, source: str) -> list[str]:
    return [edge.to for edge in manifest.edges if edge.source == source]


def _render_inputs(
    spec: NodeSpec,
    state: OrchestrationState,
    outputs: dict[str, Any],
) -> dict[str, Any]:
    context = {
        "query": state.get("query", ""),
        "context": state.get("context"),
        **outputs,
    }
    rendered: dict[str, Any] = {}
    for key, template in spec.input.items():
        try:
            rendered[key] = template.format_map(_SafeDict(context))
        except KeyError as exc:
            raise ValueError(f"node '{spec.agent or key}' input key missing: {exc}") from exc
    return rendered


class _SafeDict(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class GraphCompiler:
    def __init__(self, agent_client: RemoteAgentClient, max_steps: int = 30) -> None:
        self.agent_client = agent_client
        self.max_steps = max_steps

    def compile(self, manifest: GraphManifest, checkpointer=None):
        # GraphManifest.model_validate already runs validators when constructed via Pydantic.
        workflow = StateGraph(OrchestrationState)

        for node_id, spec in manifest.nodes.items():
            if spec.type == "agent":
                node_fn = self._agent_node(node_id, spec)
            else:
                node_fn = self._supervisor_node(node_id, spec, manifest)
            workflow.add_node(node_id, node_fn)

        workflow.set_entry_point(manifest.entry)

        for node_id, spec in manifest.nodes.items():
            edges = [edge for edge in manifest.edges if edge.source == node_id]
            if spec.type == "supervisor":
                mapping = self._conditional_mapping(edges)
                workflow.add_conditional_edges(node_id, self._route, mapping)
            else:
                if edges:
                    for edge in edges:
                        workflow.add_edge(node_id, END if edge.to == "END" else edge.to)
                else:
                    workflow.add_edge(node_id, END)

        return workflow.compile(checkpointer=checkpointer)

    def _agent_node(self, node_id: str, spec: NodeSpec):
        output_key = spec.output_key or node_id

        async def node_fn(state: OrchestrationState) -> dict[str, Any]:
            outputs = dict(state.get("outputs", {}))
            inputs = _render_inputs(spec, state, outputs)
            request = TaskRequest(
                run_id=state.get("run_id"),
                node_id=node_id,
                query=str(inputs.get("query", state.get("query", ""))),
                context=inputs.get("context"),
                inputs=inputs,
            )
            try:
                artifact = await self.agent_client.run(spec.agent, request)
            except Exception as exc:
                error = f"agent node '{node_id}' failed: {exc}"
                return {
                    "error": error,
                    "events": [{"node": node_id, "event": "failed", "error": error}],
                }

            outputs[output_key] = artifact
            return {
                "outputs": {output_key: artifact},
                "events": [{"node": node_id, "event": "completed", "output_key": output_key}],
            }

        return node_fn

    def _supervisor_node(self, node_id: str, spec: NodeSpec, manifest: GraphManifest):
        targets = _targets_for(manifest, node_id)

        async def node_fn(state: OrchestrationState) -> dict[str, Any]:
            steps = int(state.get("steps", 0)) + 1
            outputs = state.get("outputs", {})
            ordered = spec.options or [t for t in targets if t != "END"]
            decision = "finish"

            if llm_configured() and spec.prompt:
                allowed = [c for c in ordered if c != "finish"]
                if allowed:
                    state_lines = [
                        f"Query: {state.get('query', '')}",
                    ]
                    for candidate in allowed:
                        output_key = (
                            manifest.nodes[candidate].output_key
                            if candidate in manifest.nodes and manifest.nodes[candidate].output_key
                            else candidate
                        )
                        state_lines.append(f"{candidate}_done: {output_key in outputs}")
                    state_lines.append(f"step: {steps}")

                    raw = await llm_reply(
                        spec.prompt,
                        "\n".join(state_lines),
                        max_tokens=12,
                    )
                    raw = raw.strip().lower().replace(".", "").split()[0]
                    if raw in allowed:
                        output_key = (
                            manifest.nodes[raw].output_key
                            if raw in manifest.nodes and manifest.nodes[raw].output_key
                            else raw
                        )
                        if output_key not in outputs:
                            decision = raw
                if decision == "finish" and "finish" not in ordered:
                    decision = "finish"

            if decision == "finish":
                for candidate in ordered:
                    if candidate == "finish":
                        break
                    output_key = (
                        manifest.nodes[candidate].output_key
                        if candidate in manifest.nodes and manifest.nodes[candidate].output_key
                        else candidate
                    )
                    if output_key not in outputs:
                        decision = candidate
                        break

            if steps >= self.max_steps:
                decision = "finish"
            return {
                "next": decision,
                "steps": steps,
                "events": [{"node": node_id, "event": "supervisor", "decision": decision, "step": steps}],
            }

        return node_fn

    @staticmethod
    def _conditional_mapping(edges: list[EdgeSpec]) -> dict[str, str]:
        mapping: dict[str, str] = {}
        for edge in edges:
            target = edge.to
            key = "finish" if target == "END" else (edge.when or target)
            mapping[key] = END if target == "END" else target
        mapping.setdefault("finish", END)
        return mapping

    @staticmethod
    def _route(state: OrchestrationState) -> str:
        return state.get("next", "finish")
