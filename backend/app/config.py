"""Settings loaded from environment variables (and the repo-root .env file).

The only place model names, URLs and limits live (CLAUDE.md hard rule). Defaults below are the
ones the PRD states (Section 12.4); values the PRD leaves open have no default and must be set.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Ollama (runs natively on Windows; containers use host.docker.internal)
    ollama_base_url: str = "http://host.docker.internal:11434"
    fast_model: str = "gemma4:e2b"
    answer_model: str = "gemma4:e2b"

    # Embedding model and vector size are tied (PRD 8.3): no defaults, so they are always set together.
    embed_model: str
    embed_dim: int = Field(gt=0, le=2000)  # pgvector HNSW indexes support at most 2000 dimensions

    database_url: str

    # Agent limits (PRD 10.4)
    rrf_k: int = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_regen: int = Field(default=1, ge=0)

    # How long /health waits for Postgres or Ollama before reporting them down.
    health_timeout_s: float = Field(default=3.0, gt=0)

    # Offline rule (PRD 5.1): hosted LangChain tracing must stay off.
    langchain_tracing_v2: bool = False

    @field_validator("langchain_tracing_v2")
    @classmethod
    def tracing_must_be_off(cls, v: bool) -> bool:
        if v:
            raise ValueError("LANGCHAIN_TRACING_V2 must be false: the system must make no external network calls")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()
