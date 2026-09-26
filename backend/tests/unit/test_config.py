from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {"EMBED_MODEL": "qwen3-embedding:0.6b", "EMBED_DIM": "1024",
            "DATABASE_URL": "postgresql://u:p@localhost:5432/db"}


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in list(Settings.model_fields):
        monkeypatch.delenv(name.upper(), raising=False)


def set_env(monkeypatch: pytest.MonkeyPatch, **values: str) -> None:
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_loads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED, FAST_MODEL="other:1b", MAX_RETRIES="3")
    s = Settings(_env_file=None)
    assert s.embed_model == "qwen3-embedding:0.6b"
    assert s.embed_dim == 1024
    assert s.fast_model == "other:1b"
    assert s.max_retries == 3


def test_prd_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED)
    s = Settings(_env_file=None)
    assert (s.fast_model, s.answer_model) == ("gemma4:e2b", "gemma4:e2b")
    assert s.ollama_base_url == "http://host.docker.internal:11434"
    assert (s.rrf_k, s.max_retries, s.max_regen) == (60, 2, 1)


@pytest.mark.parametrize("missing", sorted(REQUIRED))
def test_required_values_have_no_default(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    set_env(monkeypatch, **{k: v for k, v in REQUIRED.items() if k != missing})
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("dim", ["0", "2001", "abc"])
def test_invalid_embed_dim_rejected(monkeypatch: pytest.MonkeyPatch, dim: str) -> None:
    set_env(monkeypatch, **{**REQUIRED, "EMBED_DIM": dim})
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_hosted_tracing_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED, LANGCHAIN_TRACING_V2="true")
    with pytest.raises(ValidationError, match="LANGCHAIN_TRACING_V2"):
        Settings(_env_file=None)
