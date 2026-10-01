"""Hybrid retrieval: pgvector cosine + BM25, fused with RRF. Shared by both modes."""
from __future__ import annotations

from collections import defaultdict
from typing import Literal

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


Method = Literal["hybrid", "vector", "bm25"]


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
    score: float  # fused RRF score: depends on rank only, so it cannot say how relevant a chunk is
    similarity: float | None = None  # cosine similarity to the query (None for BM25-only searches)

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
             timeout_s: float | None = None, method: Method = "hybrid",
             top_k: int | None = None) -> list[Retrieved]:
    """Top TOP_K_FINAL (or top_k) chunks for the query. Filters apply to both searches, before fusion.

    `method` exists for the retrieval comparison in the evaluation; the app always uses hybrid.
    With one search, RRF simply keeps that search's order.

    Exact cosine search in SQL (no HNSW): with ~100 chunks it is instant, and approximate
    indexes can return too few rows when combined with a WHERE filter.
    """
    filters = filters or Filters()
    qvec = None
    # Step 1: embed the user query
    if method != "bm25":
        [qvec] = ollama.embed(settings.ollama_base_url, settings.embed_model, [QUERY_INSTRUCTION + query],
                              timeout_s or settings.llm_call_timeout_s)

    # Step 2: fetch the candidate chunks from Postgres, with their BM25 scores and vector distances
    with conn.cursor(row_factory=dict_row) as cur:
        rows = cur.execute(
            """
            SELECT c.id, d.short_name, c.section, c.subsection, c.heading, c.part, c.page_start, c.page_end,
                   c.text, c.contextual_text, c.embedding <=> %(q)s::vector AS distance
            FROM chunks c JOIN documents d ON d.id = c.document_id
            WHERE (cardinality(%(docs)s::text[]) = 0 OR d.short_name = ANY(%(docs)s))
              AND (cardinality(%(parts)s::text[]) = 0 OR c.part = ANY(%(parts)s))
            """,
            {"q": to_vector_literal(qvec) if qvec else None, "docs": filters.documents, "parts": filters.parts},
        ).fetchall()

    with_vectors = [r for r in rows if r["distance"] is not None] # check if all vectors have distance
    by_vector = sorted(with_vectors, key=lambda r: r["distance"])[:settings.top_k_vector] # winners by vector results by distance (smallest first) take top K
    scored = zip(bm25_scores(query, [r["contextual_text"] for r in rows]), rows) # BM25 gives each row a score for how well its words match the question's words. Bigger means a better match. zip then pairs each score with its row:
    by_bm25 = [r for s, r in sorted(scored, key=lambda x: x[0], reverse=True) if s > 0][:settings.top_k_bm25] # winners by BM25 results (biggest first) take top K

    rankings = []
    if method != "bm25":
        rankings.append([r["id"] for r in by_vector])
    if method != "vector":
        rankings.append([r["id"] for r in by_bm25])
    fused = rrf(rankings, settings.rrf_k)
    row_by_id = {r["id"]: r for r in rows}
    return [Retrieved(**row_by_id[id_], score=score,
                      similarity=None if row_by_id[id_]["distance"] is None else 1 - row_by_id[id_]["distance"])
            for id_, score in fused[:top_k or settings.top_k_final]]
