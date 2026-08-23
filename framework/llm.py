"""Optional LLM helpers for agent plugins."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage

from framework.config import model_config


def llm_configured() -> bool:
    return model_config.is_configured()


async def llm_reply(system: str, user: str, max_tokens: int = 700) -> str:
    """Call the configured OpenAI-compatible model and return text."""
    llm = model_config.create_langchain_llm(max_tokens=max_tokens)
    result = await llm.ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
    return (result.content or "").strip()
