from __future__ import annotations

import pytest

from app.config import Settings


@pytest.fixture
def settings() -> Settings:
    """Settings built only from explicit values: no .env file, no real services."""
    return Settings(
        _env_file=None,
        embed_model="qwen3-embedding:0.6b",
        embed_dim=1024,
        database_url="postgresql://test:test@localhost:1/test",
        ollama_base_url="http://ollama.test:11434",
        health_timeout_s=1,
    )
