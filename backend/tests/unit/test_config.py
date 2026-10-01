from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.config import Settings

REQUIRED = {"EMBED_MODEL": "qwen3-embedding:0.6b", "EMBED_DIM": "1024",
            "DATABASE_URL": "postgresql://u:p@localhost:5432/db", "STAFF_DATABASE_URL": "postgresql://u:p@localhost:5432/staff",
            "TOP_K_VECTOR": "20", "TOP_K_BM25": "20", "TOP_K_FINAL": "5",
            "LLM_CALL_TIMEOUT_S": "90", "QUESTION_BUDGET_S": "300"}


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


@pytest.mark.parametrize("missing", ["EMBED_MODEL", "EMBED_DIM", "DATABASE_URL"])
def test_required_values_have_no_default(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    set_env(monkeypatch, **{k: v for k, v in REQUIRED.items() if k != missing})
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


@pytest.mark.parametrize("dim", ["0", "abc"])
def test_invalid_embed_dim_rejected(monkeypatch: pytest.MonkeyPatch, dim: str) -> None:
    set_env(monkeypatch, **{**REQUIRED, "EMBED_DIM": dim})
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_hosted_tracing_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    set_env(monkeypatch, **REQUIRED, LANGCHAIN_TRACING_V2="true")
    with pytest.raises(ValidationError, match="LANGCHAIN_TRACING_V2"):
        Settings(_env_file=None)
