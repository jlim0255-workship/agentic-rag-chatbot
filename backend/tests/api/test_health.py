"""/health and the startup embedding check, with Postgres and Ollama replaced by fakes."""
from __future__ import annotations

from contextlib import contextmanager

import psycopg
import pytest
from fastapi.testclient import TestClient

from app.api import health as health_module
from app.api import main as main_module
from app.config import Settings, get_settings
from app.db import meta
from app.llm import ollama

ALL_MODELS = ["gemma4:e2b", "qwen3-embedding:0.6b"]


@contextmanager
def fake_connection(*_args, **_kwargs):
    yield object()


def db_down(*_args, **_kwargs):
    raise psycopg.OperationalError("connection refused")


def ollama_down(*_args, **_kwargs):
    raise ollama.OllamaError("Ollama unreachable at http://ollama.test:11434: connection refused")


@pytest.fixture
def client(settings: Settings, monkeypatch: pytest.MonkeyPatch):
    """Healthy by default; each test breaks one thing."""
    monkeypatch.setattr(health_module.connection, "connect", fake_connection)
    monkeypatch.setattr(meta, "read_embedding_meta", lambda conn: (None, None))
    monkeypatch.setattr(meta, "embedding_column_dim", lambda conn: 1024)
    monkeypatch.setattr(ollama, "list_models", lambda url, timeout: list(ALL_MODELS))
    main_module.app.dependency_overrides[get_settings] = lambda: settings
    yield TestClient(main_module.app)  # not used as a context manager, so startup checks do not run
    main_module.app.dependency_overrides.clear()


def test_healthy(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["database"]["ok"] and body["ollama"]["ok"]
    assert body["ollama"]["missing_models"] == []
    assert body["embedding"]["status"] == "not_ingested"
    assert body["models"] == {"fast_model": "gemma4:e2b", "answer_model": "gemma4:e2b",
                              "embed_model": "qwen3-embedding:0.6b", "embed_dim": 1024}


def test_database_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health_module.connection, "connect", db_down)
    r = client.get("/health")
    assert r.status_code == 503
    body = r.json()
    assert body["database"] == {"ok": False, "error": "connection refused"}
    assert body["embedding"]["status"] == "unknown"
    assert body["ollama"]["ok"]


def test_ollama_down(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "list_models", ollama_down)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["ollama"]["ok"] is False
    assert "unreachable" in r.json()["ollama"]["error"]


def test_configured_model_not_installed(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ollama, "list_models", lambda url, timeout: ["gemma4:e2b"])
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["ollama"]["missing_models"] == ["qwen3-embedding:0.6b"]


def test_untagged_model_name_matches_latest(settings: Settings, client: TestClient,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    main_module.app.dependency_overrides[get_settings] = lambda: settings.model_copy(update={"fast_model": "llama3"})
    monkeypatch.setattr(ollama, "list_models", lambda url, timeout: [*ALL_MODELS, "llama3:latest"])
    assert client.get("/health").json()["ollama"]["missing_models"] == []


def test_embedding_mismatch(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(meta, "read_embedding_meta", lambda conn: ("nomic-embed-text:latest", 1024))
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["embedding"]["status"] == "mismatch"


# --- startup: refuse to start on an embedding model mismatch (PRD 8.3) ---

def test_startup_refuses_on_mismatch(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.connection, "connect", fake_connection)
    monkeypatch.setattr(meta, "read_embedding_meta", lambda conn: ("nomic-embed-text:latest", 768))
    monkeypatch.setattr(meta, "embedding_column_dim", lambda conn: 768)
    with pytest.raises(main_module.EmbeddingModelMismatch, match="nomic-embed-text"):
        main_module.verify_embedding_model(settings)


def test_startup_allows_match(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.connection, "connect", fake_connection)
    monkeypatch.setattr(meta, "read_embedding_meta", lambda conn: ("qwen3-embedding:0.6b", 1024))
    monkeypatch.setattr(meta, "embedding_column_dim", lambda conn: 1024)
    main_module.verify_embedding_model(settings)


def test_startup_continues_when_db_down(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main_module.connection, "connect", db_down)
    main_module.verify_embedding_model(settings)
