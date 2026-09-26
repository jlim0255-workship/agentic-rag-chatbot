from __future__ import annotations

from app.db.meta import check_embedding

MODEL, DIM = "qwen3-embedding:0.6b", 1024


def test_not_ingested_before_first_embed() -> None:
    check = check_embedding(MODEL, DIM, stored_model=None, stored_dim=None, column_dim=DIM)
    assert check.status == "not_ingested"
    assert check.problems == []


def test_match() -> None:
    assert check_embedding(MODEL, DIM, MODEL, DIM, DIM).status == "match"


def test_different_model_is_mismatch() -> None:
    check = check_embedding(MODEL, DIM, "nomic-embed-text:latest", DIM, DIM)
    assert check.status == "mismatch"
    assert "nomic-embed-text" in check.problems[0]


def test_column_size_differs_from_config_is_mismatch_even_before_ingestion() -> None:
    check = check_embedding(MODEL, 768, None, None, column_dim=1024)
    assert check.status == "mismatch"
    assert "vector(1024)" in check.problems[0]


def test_stored_dim_differs_is_mismatch() -> None:
    assert check_embedding(MODEL, DIM, MODEL, 512, DIM).status == "mismatch"


def test_no_chunks_table_yet() -> None:
    assert check_embedding(MODEL, DIM, None, None, column_dim=None).status == "not_ingested"
