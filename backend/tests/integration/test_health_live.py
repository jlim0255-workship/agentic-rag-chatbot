"""Real Postgres (docker compose up db) and real Ollama. Run with: pytest -m integration"""
from __future__ import annotations

import pytest

from app.api.health import build_health_report
from app.config import get_settings

pytestmark = pytest.mark.integration


def test_live_health() -> None:
    report = build_health_report(get_settings())
    assert report.database.ok, report.database.error
    assert report.ollama.ok, report.ollama.error
    assert report.ollama.missing_models == []
    assert report.embedding.status != "mismatch", report.embedding.problems
