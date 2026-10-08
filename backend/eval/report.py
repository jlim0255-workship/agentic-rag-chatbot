"""Markdown reports for the evaluation, plus the golden set and report paths shared by run_retrieval,
run_e2e and judge. Report functions are pure: rows in, Markdown out. Every report is Markdown only.

Verdicts (plain text, so tables can be pasted into slides):
  FULL  every needed section was among the chunks passed on      PART  some were      MISS  none were
Every rate is shown with its count, e.g. "7/10 (70%)".
"""
from __future__ import annotations

import json
import statistics
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).parent
GOLDEN = EVAL_DIR / "golden.jsonl"
REPORTS = EVAL_DIR / "reports"

DECLINED = ("not_found", "out_of_scope")  # statuses where no answer from the documents was given
CATEGORIES = ["single_hop", "multi_hop", "not_in_corpus", "ambiguous"]
VERDICT = {"full": "FULL", "partial": "PART", "miss": "MISS"}
LEGEND = "FULL = every needed section found · PART = some · MISS = none · #n = rank of the first needed section"


# --- golden set and files ---

def load_golden() -> list[dict]:
    return [json.loads(line) for line in GOLDEN.read_text(encoding="utf-8").splitlines() if line.strip()]


def stem(name: str) -> Path:
    """reports/<timestamp>-<name>, the prefix every file of one run shares."""
    REPORTS.mkdir(exist_ok=True)
    return REPORTS / f"{datetime.now():%Y%m%d-%H%M}-{name}"


def write_md(name: str, markdown: str) -> Path:
    path = stem(name).with_suffix(".md")
    path.write_text(markdown, encoding="utf-8")
    return path


# --- small helpers ---

def rate(hits: int, n: int) -> str:
    return f"{hits}/{n} ({hits / n:.0%})" if n else "n/a (0 questions)"


def section_label(e: dict) -> str:
    """Golden-set expected section -> 'ACT854 s23', 'PUA220 reg 2', 'ACT854 Schedule', 'ACT854 s4 “term”'."""
    if e["section"] in ("Schedule", "long_title"):
        return f"{e['document']} {e['section']}"
    unit = "reg " if e["document"].startswith("PUA") else "s"
    return f"{e['document']} {unit}{e['section']}" + (f" “{e['term']}”" if e.get("term") else "")


def outcome(expected: list[tuple[str, set[str]]], got: list[str]) -> dict:
    """expected: (label, acceptable chunk IDs) per needed section; got: chunk IDs in rank order."""
    found = [label for label, ids in expected if ids & set(got)]
    ranks = [i for i, c in enumerate(got, 1) if any(c in ids for _, ids in expected)]
    level = "full" if len(found) == len(expected) else "partial" if found else "miss"
    return {"level": level, "found": len(found), "total": len(expected),
            "missing": [label for label, _ in expected if label not in found], "rank": ranks[0] if ranks else None}


def cell(o: dict) -> str:
    if o["level"] == "full":
        return f"FULL #{o['rank']}"
    if o["level"] == "partial":
        return f"PART {o['found']}/{o['total']} (missing {', '.join(o['missing'])})"
    return "MISS"


def short(text: str, n: int = 48) -> str:
    return text if len(text) <= n else text[:n - 1] + "…"


def table(head: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(head) + " |", "|" + "---|" * len(head), *("| " + " | ".join(r) + " |" for r in rows)]


def count(qs: list[dict], category: str) -> int:
    return sum(q["category"] == category for q in qs)


# --- Experiment A: retrieval comparison (no LLM) ---

def retrieval_report(rows: list[dict], methods: list[str], questions: list[dict], header: list[str],
                     best_similarity: dict[str, float]) -> str:
    """rows: one per (question with known sections, method): id, category, method, outcome, ms.
    best_similarity: question id -> cosine similarity of its best-matching chunk (all questions)."""
    by = {(r["id"], r["method"]): r for r in rows}
    graded = [q for q in questions if q["expected_sections"]]
    md = [*header, "", LEGEND, ""]

    md += ["## Table 1: Retrieval accuracy by method", ""]
    t1 = []
    for m in methods:
        sel = [r for r in rows if r["method"] == m]
        ranks = [r["outcome"]["rank"] for r in sel if r["outcome"]["rank"]]
        t1.append([m, *(rate(sum(r["outcome"]["level"] == lvl for r in sel), len(sel)) for lvl in VERDICT),
                   f"{sum(ranks) / len(ranks):.1f}" if ranks else "–", f"{sum(r['ms'] for r in sel) / len(sel):.0f} ms"])
    md += table(["Method", "FULL", "PART", "MISS", "Avg rank of first needed section", "Avg time"], t1)

    cats = [c for c in CATEGORIES if count(graded, c)]
    md += ["", "## Table 2: FULL by category", ""]
    md += table(["Method", *(f"{c} ({count(graded, c)})" for c in cats)],
                [[m, *(rate(sum(by[(q["id"], m)]["outcome"]["level"] == "full" for q in graded if q["category"] == c),
                            count(graded, c)) for c in cats)] for m in methods])

    md += ["", "## Table 3: Best-match similarity by category (for a relevance cut-off)", "",
           "Cosine similarity between the question and its closest chunk. A cut-off could only reject "
           "not_in_corpus questions safely if their values sit below every answerable question's.", ""]
    t3 = []
    for c in [c for c in CATEGORIES if count(questions, c)]:
        vals = [best_similarity[q["id"]] for q in questions if q["category"] == c]
        t3.append([f"{c} ({len(vals)})", f"{min(vals):.3f}", f"{statistics.median(vals):.3f}", f"{max(vals):.3f}"])
    md += table(["Category", "Min", "Median", "Max"], t3)

    md += ["", "## Table 4: Per question", ""]
    md += table(["ID", "Question", "Needs", *methods, "Best similarity"],
                [[q["id"], short(q["question"]), ", ".join(section_label(e) for e in q["expected_sections"]) or "–",
                  *(cell(by[(q["id"], m)]["outcome"]) if q["expected_sections"] else "–" for m in methods),
                  f"{best_similarity[q['id']]:.3f}"] for q in questions])
    return "\n".join(md) + "\n"


# --- Experiment B: per-answer file for manual review (the headline report is judge_report below) ---

STATUS = {"answered": "answered", "partial": "partial answer", "not_found": "said not found",
          "out_of_scope": "refused: out of scope", "timeout": "timeout", "error": "error"}


def answers_file(mode: str, rows: list[dict], questions: list[dict], when: str) -> str:
    by = {r["id"]: r for r in rows}
    md = [f"# Answers: {mode} mode ({when})", "",
          "For manual review: each answer next to the key facts a correct answer contains.", ""]
    for q in questions:
        r = by[q["id"]]
        md += [f"## {q['id']} ({q['category']}): {q['question']}", "",
               f"**Status:** {STATUS.get(r['status'], r['status'])}", "",
               f"**Given to the writer:** {r['passed'] or 'none'} · **Cited:** {r['citations'] or 'none'}", "",
               r["answer"] or "_(no answer)_", "", "**Key facts expected:**", *[f"- {n}" for n in q["answer_notes"]], ""]
    return "\n".join(md)


# --- Experiment B judged: the RAG triad per configuration (model x mode) ---

TRIAD = {"context_relevance": "Context relevance", "faithfulness": "Faithfulness", "relevance": "Answer relevance"}


def mean(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    return f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "n/a (n=0)"


def judged_cell(r: dict) -> str:
    """Per-question scores: CR = context relevance, F = faithfulness, AR = answer relevance."""
    def num(v: float | None) -> str:
        return "–" if v is None else f"{v:.2f}"
    cr = f"CR {num(r['context_relevance'])}"
    if r["status"] not in ("answered", "partial"):
        return f"{cr} · {STATUS.get(r['status'], r['status'])}"
    return f"{cr} · F {num(r['faithfulness'])} · AR {num(r['relevance'])}"


def judge_report(rows: list[dict], questions: list[dict], header: list[str]) -> str:
    """rows: judged rows (eval.judge) with `config` = "<model> <mode>"; one per question and config."""
    configs = list(dict.fromkeys(r["config"] for r in rows))
    sel = {c: [r for r in rows if r["config"] == c] for c in configs}

    md = [*header, "",
          "Context relevance = share of the chunks given to the writer that help answer the question (judged "
          "whenever the writer got chunks, even if it then declined; agentic mode gives the writer up to "
          "10 chunks, traditional 5). Faithfulness = share of the answer's claims supported by those chunks. "
          "Answer relevance = does the answer address the question (yes 1, partly 0.5, no 0). Faithfulness "
          "and answer relevance are judged only on answers given. n = rows scored. One run per configuration: "
          "differences of 1-2 questions may be run-to-run noise. Scores come from an LLM judge and can be "
          "wrong in both directions; none of them checks the answer against the golden key facts.", "",
          "## Table 1: RAG triad", ""]
    def given(rs: list[dict]) -> int:
        return sum(r["status"] in ("answered", "partial") for r in rs)

    md += table(["", *configs], [
        *([f"{name}, mean", *(mean([r[key] for r in sel[c]]) for c in configs)] for key, name in TRIAD.items()),
        ["Answers given", *(str(given(sel[c])) for c in configs)],
        ["Declined (not found, out of scope, timeout, error)", *(str(len(sel[c]) - given(sel[c])) for c in configs)],
        ["Judge calls invalid twice", *(str(sum(len(r["unjudged"]) for r in sel[c])) for c in configs)]])

    by = {(r["id"], r["config"]): r for r in rows}
    md += ["", "## Table 2: Per question", "",
           "CR = context relevance · F = faithfulness · AR = answer relevance · – = not judged", ""]
    md += table(["ID", "Question", *configs],
                [[q["id"], short(q["question"], 40), *(judged_cell(by[(q["id"], c)]) if (q["id"], c) in by else "–"
                                                      for c in configs)] for q in questions])
    return "\n".join(md) + "\n"
