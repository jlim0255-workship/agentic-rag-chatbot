"""Evaluation runner: runs the golden questions and writes reports to eval/reports/.
Collecting results lives here; turning them into Markdown lives in report.py.

Usage (from backend/; DB ingested, Ollama running):
    python -m eval.run retrieval                 # Experiment A: semantic vs BM25 vs hybrid, no LLM (~1 min)
    python -m eval.run e2e                       # Experiment B: traditional vs agentic via the API (~25 min)
    python -m eval.run e2e --mode agentic        #   one mode only
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import httpx

from app.config import get_settings
from app.db.connection import connect
from app.retrieval.search import retrieve
from eval.report import answers_file, e2e_report, outcome, retrieval_report, section_label

EVAL_DIR = Path(__file__).parent
GOLDEN = EVAL_DIR / "golden.jsonl"
REPORTS = EVAL_DIR / "reports"
METHODS = {"Semantic": ("vector", None), "BM25": ("bm25", None), "Hybrid": ("hybrid", None),
           "Hybrid@10": ("hybrid", 10)}


def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def save(name: str, markdown: str, rows: list[dict]) -> Path:
    """Write <timestamp>-<name>.md and a flat .csv (one line per row) for spreadsheets."""
    REPORTS.mkdir(exist_ok=True)
    stem = REPORTS / f"{datetime.now():%Y%m%d-%H%M}-{name}"
    stem.with_suffix(".md").write_text(markdown, encoding="utf-8")
    flat = [{k: (v["level"] if k == "outcome" and v else json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v)
             for k, v in r.items() if k != "answer"} for r in rows]
    with stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return stem


# --- Experiment A: retrieval only ---

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
                             "outcome": outcome(expected, got), "ms": round((time.perf_counter() - start) * 1000, 1),
                             "retrieved": " ".join(got), "best_similarity": round(best[q["id"]], 3)})
    n = sum(bool(q["expected_sections"]) for q in questions)
    header = [f"# Retrieval: semantic vs keyword (BM25) vs hybrid ({datetime.now():%Y-%m-%d %H:%M})", "",
              f"{n} questions with known answer sections ({len(questions)} for similarity) · {settings.top_k_final} "
              f"chunks passed on (Hybrid@10: 10) · candidates per search: vector {settings.top_k_vector}, "
              f"BM25 {settings.top_k_bm25} · RRF k={settings.rrf_k} · no LLM involved · this is the search "
              f"traditional mode runs; time includes embedding the question (none for BM25)."]
    return save("retrieval", retrieval_report(rows, list(METHODS), questions, header, best), rows)


# --- Experiment B: end to end through the API ---

def parse_sse(lines) -> list[tuple[str, object]]:
    events, name = [], None
    for line in lines:
        if line.startswith("event: "):
            name = line[7:]
        elif line.startswith("data: "):
            events.append((name, json.loads(line[6:])))
    return events


def ask(api: str, question: str, mode: str, timeout_s: float) -> dict:
    with httpx.stream("POST", f"{api}/chat", json={"question": question, "mode": mode},
                      timeout=httpx.Timeout(10, read=timeout_s)) as r:
        r.raise_for_status()
        return collect(parse_sse(r.iter_lines()))


def collect(events) -> dict:
    """One question's (event, data) pairs -> the fields the report needs. Shared by the API runner and
    run_cloud.py (which runs the pipeline in-process)."""
    result = {"status": "error", "answer": "", "label": "", "citations": [], "passed": [], "metrics": {}}
    for name, data in events:
        # Evidence = the chunks given to the writer. None if the writer was never called.
        if name == "trace" and data["node"] == "generate" and data["status"] == "start":
            result["passed"] = data["data"]["chunks"]
        elif name == "answer":
            result["answer"], result["label"] = data["text"], data["label"]
        elif name == "citations":
            result["citations"] = [c["chunk_id"] for c in data]
        elif name == "metrics":
            result["metrics"] = data
        elif name == "done":
            result["status"] = data["status"]
        elif name == "error":
            result["answer"] = f"ERROR: {data['message']}"
    return result


def run_e2e(modes: list[str], ask_fn, settings, name: str = "e2e") -> Path:
    """ask_fn(question, mode) -> collect()-style dict. `settings` must hold the models ask_fn really uses,
    so the report header names them."""
    questions = load_golden()
    rows = []
    with connect(settings.database_url, settings.health_timeout_s) as conn:
        expected = {q["id"]: expected_sections(conn, q) for q in questions}
    for mode in modes:
        ask_fn("warm-up: what is the short title of the Act?", mode)  # load models; not counted
        for i, q in enumerate(questions, 1):
            print(f"[{mode} {i}/{len(questions)}] {q['id']}", flush=True)
            res = ask_fn(q["question"], mode)
            allowed = set().union(*(ids for _, ids in expected[q["id"]]))
            rows.append({
                "id": q["id"], "category": q["category"], "mode": mode, "status": res["status"],
                "outcome": outcome(expected[q["id"]], res["passed"]) if expected[q["id"]] else None,
                "label": res["label"], "cited": len(res["citations"]),
                "cited_expected": sum(c in allowed for c in res["citations"]),
                "latency_ms": res["metrics"].get("latency_ms"), "llm_calls": res["metrics"].get("llm_calls"),
                "retries": res["metrics"].get("retries"), "passed": " ".join(res["passed"]),
                "citations": " ".join(res["citations"]), "answer": res["answer"]})

    when = f"{datetime.now():%Y-%m-%d %H:%M}"
    header = [f"# Traditional vs agentic, end to end ({when})", "",
              f"{len(questions)} questions · answer model `{settings.answer_model}` · embedding `{settings.embed_model}` · "
              f"latency excludes one warm-up question per mode."]
    stem = save(name, e2e_report(rows, modes, questions, header), rows)
    # Raw rows incl. answer text, so eval.judge can score them without generating again.
    with stem.with_name(f"{stem.name}-rows.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "model": settings.answer_model}, ensure_ascii=False) + "\n")
    for mode in modes:
        text = answers_file(mode, [r for r in rows if r["mode"] == mode], questions, when)
        stem.with_name(f"{stem.name}-{mode}-answers.md").write_text(text, encoding="utf-8")
    return stem


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["retrieval", "e2e"])
    parser.add_argument("--mode", choices=["traditional", "agentic"], help="run one mode only (default: both)")
    parser.add_argument("--api", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    if args.experiment == "retrieval":
        stem = run_retrieval()
    else:
        settings = get_settings()
        timeout = settings.question_budget_s + 30
        stem = run_e2e([args.mode] if args.mode else ["traditional", "agentic"],
                       lambda question, mode: ask(args.api, question, mode, timeout), settings)
    print(f"report: {stem}.md")


if __name__ == "__main__":
    main()
