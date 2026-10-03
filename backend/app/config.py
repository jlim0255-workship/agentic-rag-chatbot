"""Settings loaded from environment variables (and the repo-root .env file).

The only place model names, URLs and limits live. Fields without a default must be set in .env.
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

    # Embedding model and vector size are tied (changing either means re-ingesting): no defaults, so they are always set together.
    embed_model: str
    embed_dim: int = Field(gt=0, le=2000)  # pgvector HNSW indexes support at most 2000 dimensions
    embed_batch_size: int = Field(default=16, gt=0)   # chunks per Ollama embed request
    embed_timeout_s: float = Field(default=120, gt=0)  # per embed request; first call also loads the model

    database_url: str

    # Retrieval. No defaults: set in .env (keep TOP_K_FINAL small for 4 GB VRAM).
    top_k_vector: int = Field(gt=0)
    top_k_bm25: int = Field(gt=0)
    top_k_final: int = Field(gt=0)

    # LLM calls. num_ctx is set explicitly so long prompts are never silently truncated.
    llm_num_ctx: int = Field(default=8192, gt=0)
    llm_call_timeout_s: float = Field(gt=0)
    question_budget_s: float = Field(gt=0)

    # Agent limits
    rrf_k: int = Field(default=60, gt=0)
    max_retries: int = Field(default=2, ge=0)
    max_sub_queries: int = Field(default=3, gt=0)
    writer_max_chunks: int = Field(default=10, gt=0)  # agentic: chunks judged sufficient + other retrieved context

    # How long /health waits for Postgres or Ollama before reporting them down.
    health_timeout_s: float = Field(default=3.0, gt=0)

    # Offline rule: hosted LangChain tracing must stay off.
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
