"""Experiment A: retrieval only (semantic vs BM25 vs hybrid), no LLM. Writes one Markdown report to
eval/reports/. Collecting results lives here; turning them into Markdown lives in report.py.

Usage (from backend/; DB ingested, Ollama running for the query embeddings):
    python -m eval.run_retrieval                 # ~1 min
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

from app.config import get_settings
from app.db.connection import connect
from app.retrieval.search import retrieve
from eval.report import load_golden, outcome, retrieval_report, section_label, write_md

METHODS = {"Semantic": ("vector", None), "BM25": ("bm25", None), "Hybrid": ("hybrid", None),
           "Hybrid@10": ("hybrid", 10)}


def expected_sections(conn, question: dict) -> list[tuple[str, set[str]]]:
    """(label, acceptable chunk IDs) per expected section. A section without a term (e.g. s4) accepts any
    of its chunks."""
    out = []
    for e in question["expected_sections"]:
        rows = conn.execute(
            """SELECT c.id FROM chunks c JOIN documents d ON d.id = c.document_id
               WHERE d.short_name = %s AND c.section = %s AND (%s::text IS NULL OR c.subsection = %s)""",
            [e["document"], e["section"], e.get("term"), e.get("term")]).fetchall()
        if not rows:
            raise ValueError(f"{question['id']}: expected section {e} matches no chunk")
        out.append((section_label(e), {r[0] for r in rows}))
    return out


def run_retrieval() -> Path:
    settings = get_settings()
    questions = load_golden()
    rows, best = [], {}
    with connect(settings.database_url, settings.health_timeout_s) as conn:
        retrieve(conn, settings, "warm-up")  # load the embedding model before timing anything
        for q in questions:
            # Best-match similarity for every question (incl. not-in-docs), to judge a relevance cut-off.
            best[q["id"]] = retrieve(conn, settings, q["question"], method="vector", top_k=1)[0].similarity
            if not q["expected_sections"]:
                continue
            expected = expected_sections(conn, q)
            for name, (method, top_k) in METHODS.items():
                start = time.perf_counter()
                got = [c.id for c in retrieve(conn, settings, q["question"], method=method, top_k=top_k)]
                rows.append({"id": q["id"], "category": q["category"], "method": name,
                             "outcome": outcome(expected, got), "ms": round((time.perf_counter() - start) * 1000, 1)})
    n = sum(bool(q["expected_sections"]) for q in questions)
    header = [f"# Retrieval: semantic vs keyword (BM25) vs hybrid ({datetime.now():%Y-%m-%d %H:%M})", "",
              f"{n} questions with known answer sections ({len(questions)} for similarity) · {settings.top_k_final} "
              f"chunks passed on (Hybrid@10: 10) · candidates per search: vector {settings.top_k_vector}, "
              f"BM25 {settings.top_k_bm25} · RRF k={settings.rrf_k} · no LLM involved · this is the search "
              f"traditional mode runs; time includes embedding the question (none for BM25)."]
    return write_md("retrieval", retrieval_report(rows, list(METHODS), questions, header, best))


def main() -> None:
    argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter).parse_args()
    print(f"report: {run_retrieval()}")


if __name__ == "__main__":
    main()
