"""Environment configuration and OpenAI-compatible model helpers."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parents[2]
_DAY5_DIR = Path(__file__).resolve().parents[1]
load_dotenv(_ROOT / "day4-agentic-rag-tool-use" / ".env", override=False)
load_dotenv(_DAY5_DIR / ".env", override=False)


@dataclass(frozen=True)
class ModelConfig:
    api_key: str
    base_url: str
    model: str
    provider_name: str

    @classmethod
    def from_env(cls) -> "ModelConfig":
        if os.getenv("DEEPSEEK_API_KEY"):
            return cls(
                api_key=os.getenv("DEEPSEEK_API_KEY", ""),
                base_url="https://api.deepseek.com/v1",
                model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
                provider_name="DeepSeek",
            )
        if os.getenv("OPENAI_API_KEY"):
            return cls(
                api_key=os.getenv("OPENAI_API_KEY", ""),
                base_url="https://api.openai.com/v1",
                model=os.getenv("OPENAI_MODEL", "gpt-4o"),
                provider_name="OpenAI",
            )
        return cls(
            api_key=os.getenv("MODEL_API_KEY", "sk-no-key-needed"),
            base_url=os.getenv("MODEL_BASE_URL", "https://api.openai.com/v1"),
            model=os.getenv("MODEL_NAME", "gpt-4o"),
            provider_name="Custom",
        )

    def is_configured(self) -> bool:
        key = self.api_key.strip()
        return bool(key) and key not in {"", "sk-no-key-needed"} and "your-" not in key

    def create_langchain_llm(self, temperature: float = 0.0, max_tokens: int | None = None):
        from langchain_openai import ChatOpenAI

        kwargs = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            "temperature": temperature,
        }
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens
        return ChatOpenAI(**kwargs)


model_config = ModelConfig.from_env()


def env_url(name: str, default: str) -> str:
    return os.getenv(name, default).rstrip("/")


REGISTRY_URL = env_url("REGISTRY_URL", "http://127.0.0.1:8001")
ORCHESTRATOR_URL = env_url("ORCHESTRATOR_URL", "http://127.0.0.1:8020")
PORTAL_BACKEND_URL = env_url("PORTAL_BACKEND_URL", "http://127.0.0.1:8030")
STOCKPORTAL_URL = env_url("STOCKPORTAL_URL", "http://127.0.0.1:8040")
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "state/orchestrator.db")
PORTAL_DB = os.getenv("PORTAL_DB", "state/portal.db")
STOCK_CACHE_DB = os.getenv("STOCK_CACHE_DB", "state/stock_cache.db")
STOCK_PORTAL_DB = os.getenv("STOCK_PORTAL_DB", "state/stock_portal.db")
