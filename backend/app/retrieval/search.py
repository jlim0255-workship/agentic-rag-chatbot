"""Hybrid retrieval: pgvector cosine + BM25, fused with RRF (PRD 9). Shared by both modes."""
from __future__ import annotations

from collections import defaultdict

import psycopg
from psycopg.rows import dict_row
from pydantic import BaseModel

from app.config import Settings
from app.ingest.db import to_vector_literal
from app.llm import ollama
from app.retrieval.bm25 import bm25_scores

# Qwen3-Embedding wants an instruction on the query side only (documents were embedded without one).
QUERY_INSTRUCTION = ("Instruct: Given a question about Malaysian cyber security law, "
                     "retrieve the legal provisions that answer it\nQuery: ")


class Filters(BaseModel):
    documents: list[str] = []  # short_name values, e.g. "ACT854"
    parts: list[str] = []      # Roman numerals, e.g. "IV"


class Retrieved(BaseModel):
    id: str
    short_name: str
    section: str
    subsection: str | None
    heading: str
    part: str | None
    page_start: int
    page_end: int
    text: str
    score: float  # fused RRF score

    @property
    def label(self) -> str:
        """Human citation label: 'ACT854 s23', 'PUA220 reg 2', 'ACT854 s4 “computer”', 'ACT854 Schedule'."""
        if self.section in ("long_title", "Schedule"):
            return f"{self.short_name} {'Long title' if self.section == 'long_title' else 'Schedule'}"
        unit = ("reg " if self.id.split(":")[1].startswith("r") else "s") + self.section
        return f"{self.short_name} {unit}" + (f" “{self.subsection}”" if self.subsection else "")

    @property
    def pages(self) -> str:
        return str(self.page_start) if self.page_start == self.page_end else f"{self.page_start}–{self.page_end}"


def rrf(rankings: list[list[str]], k: int) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion: score(id) = sum over rankings of 1 / (k + rank), rank from 1."""
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, id_ in enumerate(ranking, start=1):
            scores[id_] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def retrieve(conn: psycopg.Connection, settings: Settings, query: str, filters: Filters | None = None,
             timeout_s: float | None = None) -> list[Retrieved]:
    """Top TOP_K_FINAL chunks for the query. Filters apply to both searches, before fusion.

    Exact cosine search in SQL (no HNSW): with ~100 chunks it is instant, and approximate
    indexes can return too few rows when combined with a WHERE filter.
    """
    filters = filters or Filters()
    [qvec] = ollama.embed(settings.ollama_base_url, settings.embed_model, [QUERY_INSTRUCTION + query],
                          timeout_s or settings.llm_call_timeout_s)
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """
            SELECT c.id, d.short_name, c.section, c.subsection, c.heading, c.part, c.page_start, c.page_end,
                   c.text, c.contextual_text, c.embedding <=> %(q)s::vector AS distance
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE (cardinality(%(docs)s::text[]) = 0 OR d.short_name = ANY(%(docs)s))
              AND (cardinality(%(parts)s::text[]) = 0 OR c.part = ANY(%(parts)s))
            """,
            {"q": to_vector_literal(qvec), "docs": filters.documents, "parts": filters.parts},
        ).fetchall()

    with_vectors = [r for r in rows if r["distance"] is not None]
    by_vector = sorted(with_vectors, key=lambda r: r["distance"])[:settings.top_k_vector]
    scored = zip(bm25_scores(query, [r["contextual_text"] for r in rows]), rows)
    by_bm25 = [r for s, r in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0][:settings.top_k_bm25]

    fused = rrf([[r["id"] for r in by_vector], [r["id"] for r in by_bm25]], settings.rrf_k)
    row_by_id = {r["id"]: r for r in rows}
    return [Retrieved(**row_by_id[id_], score=score) for id_, score in fused[:settings.top_k_final]]
