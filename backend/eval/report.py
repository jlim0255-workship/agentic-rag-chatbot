"""Markdown reports for the evaluation runner. Pure functions: rows in, Markdown out.

Verdicts (plain text, so tables can be pasted into slides):
  FULL  every needed section was among the chunks passed on      PART  some were      MISS  none were
  OK / WRONG  for questions the documents cannot answer: declined (not found / out of scope) or not
Honesty rule (PRD 13.3): every rate is shown with its count, e.g. "7/10 (70%)".
"""
from __future__ import annotations

import math
import statistics

DECLINED = ("not_found", "out_of_scope")  # statuses where no answer from the documents was given
CATEGORIES = ["single_hop", "multi_hop", "not_in_corpus", "ambiguous"]
VERDICT = {"full": "FULL", "partial": "PART", "miss": "MISS"}
LEGEND = ("FULL = every needed section found · PART = some · MISS = none · #n = rank of the first needed "
          "section · OK / WRONG = a question the documents cannot answer was declined / was answered")


# --- small helpers ---

def rate(hits: int, n: int) -> str:
    return f"{hits}/{n} ({hits / n:.0%})" if n else "n/a (0 questions)"


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile, p in (0, 1]."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p * len(ordered)) - 1)] if ordered else float("nan")


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


# --- Experiment B: traditional vs agentic, end to end ---

STATUS = {"answered": "answered", "partial": "partial answer", "not_found": "said not found",
          "out_of_scope": "refused: out of scope", "timeout": "timeout", "error": "error"}


def answer_cell(r: dict) -> str:
    """Per-question result for one mode: evidence given to the writer, final status."""
    status = STATUS.get(r["status"], r["status"])
    if r["category"] == "not_in_corpus":
        return f"{'OK' if r['status'] in DECLINED else 'WRONG'} · {status}"
    evidence = cell(r["outcome"]) if r["outcome"] else "–"
    return f"{evidence} · {status}"


def e2e_metrics(sel: list[dict]) -> dict[str, str]:
    """Headline values for one mode's rows."""
    graded = [r for r in sel if r["outcome"]]  # questions with known answer sections
    off = [r for r in sel if r["category"] == "not_in_corpus"]
    answerable = [r for r in sel if r["category"] in ("single_hop", "multi_hop")]
    lat = [r["latency_ms"] / 1000 for r in sel if r["latency_ms"] is not None]

    def full(rows: list[dict]) -> str:
        return rate(sum(r["outcome"]["level"] == "full" for r in rows), len(rows))

    return {
        "Right sections given to the writer (FULL)": full(graded),
        "Multi-hop questions: FULL": full([r for r in graded if r["category"] == "multi_hop"]),
        "Not-in-docs questions declined (OK)": rate(sum(r["status"] in DECLINED for r in off), len(off)),
        "Answerable questions wrongly declined": rate(sum(r["status"] in DECLINED for r in answerable), len(answerable)),
        "Citations pointing at a needed section":
            rate(sum(r["cited_expected"] for r in graded), sum(r["cited"] for r in graded)),
        "Latency p50 / p95": f"{percentile(lat, .5):.1f} s / {percentile(lat, .95):.1f} s",
        "LLM calls per question (mean)": f"{sum(r['llm_calls'] or 0 for r in sel) / len(sel):.1f}",
        "Timeouts / errors": f"{sum(r['status'] == 'timeout' for r in sel)} / {sum(r['status'] == 'error' for r in sel)}",
    }


def category_cell(sel: list[dict], category: str) -> str:
    """FULL rate for categories with known sections; OK rate for not_in_corpus."""
    if category == "not_in_corpus":
        return "OK " + rate(sum(r["status"] in DECLINED for r in sel), len(sel))
    graded = [r for r in sel if r["outcome"]]
    return "FULL " + rate(sum(r["outcome"]["level"] == "full" for r in graded), len(graded))


def e2e_report(rows: list[dict], modes: list[str], questions: list[dict], header: list[str]) -> str:
    """rows: one per (question, mode) with keys id, category, mode, status, outcome, label, cited,
    cited_expected, latency_ms, llm_calls, retries."""
    by = {(r["id"], r["mode"]): r for r in rows}
    metrics = {m: e2e_metrics([r for r in rows if r["mode"] == m]) for m in modes}
    md = [*header, "", LEGEND, "", "## Table 1: Headline", ""]
    md += table(["", *modes], [[name, *(metrics[m][name] for m in modes)] for name in metrics[modes[0]]])

    cats = [c for c in CATEGORIES if count(questions, c)]
    md += ["", "## Table 2: By category", "", "FULL for questions with known sections; OK for not-in-docs questions.", ""]
    md += table(["Category", *modes],
                [[f"{c} ({count(questions, c)})",
                  *(category_cell([r for r in rows if r["mode"] == m and r["category"] == c], c) for m in modes)]
                 for c in cats])

    # Diff compares what matters (evidence level + final status), not the source-check part only agentic has.
    def key(r: dict) -> tuple:
        return (r["outcome"]["level"] if r["outcome"] else None, r["status"])

    md += ["", "## Table 3: Per question", "", "Diff = * where the modes differ in evidence or final status.", ""]
    md += table(["ID", "Question", "Needs", *modes, "Diff"],
                [[q["id"], short(q["question"]), ", ".join(section_label(e) for e in q["expected_sections"]) or "–",
                  *(answer_cell(by[(q["id"], m)]) for m in modes),
                  "*" if len({key(by[(q["id"], m)]) for m in modes}) > 1 else ""] for q in questions])
    md += ["", "Answer correctness is not scored automatically; see the `-answers.md` file(s) for manual review."]
    return "\n".join(md) + "\n"


def answers_file(mode: str, rows: list[dict], questions: list[dict], when: str) -> str:
    by = {r["id"]: r for r in rows}
    md = [f"# Answers: {mode} mode ({when})", "",
          "For manual review: each answer next to the key facts a correct answer contains.", ""]
    for q in questions:
        r = by[q["id"]]
        md += [f"## {q['id']} ({q['category']}): {q['question']}", "",
               f"**Result:** {answer_cell(r)} · **Label:** {r['label'] or '–'}", "",
               f"**Given to the writer:** {r['passed'] or 'none'} · **Cited:** {r['citations'] or 'none'}", "",
               r["answer"] or "_(no answer)_", "", "**Key facts expected:**", *[f"- {n}" for n in q["answer_notes"]], ""]
    return "\n".join(md)


# --- Experiment B judged: LLM-as-judge scores per configuration (model x mode) ---

ATTRIBUTION = ["Success", "Correct but not grounded", "Hallucination", "Writer missed facts it had", "Retrieval failure"]


def mean(values: list[float | None]) -> str:
    vals = [v for v in values if v is not None]
    return f"{sum(vals) / len(vals):.2f} (n={len(vals)})" if vals else "n/a (n=0)"


def judged_cell(r: dict) -> str:
    """Per-question scores: C = correctness, F = faithfulness, R = relevance."""
    def num(v: float | None) -> str:
        return "–" if v is None else f"{v:.2f}"
    if r["faithfulness"] is None and r["relevance"] is None and not r["unjudged"]:
        return f"C {num(r['correctness'])} · {STATUS.get(r['status'], r['status'])}"
    return f"C {num(r['correctness'])} · F {num(r['faithfulness'])} · R {num(r['relevance'])}"


def judge_report(rows: list[dict], questions: list[dict], header: list[str], correct_at: float) -> str:
    """rows: judged rows (eval.judge) with `config` = "<model> <mode>"; one per question and config."""
    configs = list(dict.fromkeys(r["config"] for r in rows))
    sel = {c: [r for r in rows if r["config"] == c] for c in configs}

    def correct(rs: list[dict]) -> str:
        return rate(sum((r["correctness"] or 0) >= correct_at for r in rs), len(rs))

    def headline(rs: list[dict]) -> dict[str, str]:
        answered = [r for r in rs if r["faithfulness"] is not None]
        graded = [r for r in rs if r["outcome"]]
        off = [r for r in rs if r["category"] == "not_in_corpus"]
        lat = [r["latency_ms"] / 1000 for r in rs if r["latency_ms"] is not None]
        return {
            "Correct (score >= %.2f), all questions" % correct_at: correct(rs),
            "Correctness, mean": mean([r["correctness"] for r in rs]),
            "Faithfulness, mean (answers with claims)": mean([r["faithfulness"] for r in rs]),
            "Fully faithful answers": rate(sum(r["faithfulness"] == 1.0 for r in answered), len(answered)),
            "Answer relevance, mean (answers given)": mean([r["relevance"] for r in rs]),
            "Answers contradicting a key fact": str(sum("contradicted" in (r["verdicts"] or []) for r in rs)),
            "Right sections given to the writer (FULL)":
                rate(sum(r["outcome"]["level"] == "full" for r in graded), len(graded)),
            "Not-in-docs questions declined (OK)": rate(sum(r["correctness"] == 1.0 for r in off), len(off)),
            "Latency p50 / p95": f"{percentile(lat, .5):.1f} s / {percentile(lat, .95):.1f} s",
            "LLM calls per question (mean)": f"{sum(r['llm_calls'] or 0 for r in rs) / len(rs):.1f}",
            "Judge calls invalid twice (unjudged)": str(sum(len(r["unjudged"]) for r in rs)),
        }

    heads = {c: headline(sel[c]) for c in configs}
    md = [*header, "",
          f"Correctness = share of the golden key facts the answer covers (partly = half), judged against "
          f"`answer_notes`; a not-in-docs question scores 1 if declined, 0 if answered. Faithfulness = share of "
          f"the answer's claims supported by the chunks given to the writer. Relevance = does it address the "
          f"question (yes 1, partly 0.5, no 0). One run per configuration: differences of 1-2 questions may be "
          f"run-to-run noise. Scores come from an LLM judge and can be wrong in both directions.", "",
          "## Table 1: Headline", ""]
    md += table(["", *configs], [[name, *(heads[c][name] for c in configs)] for name in heads[configs[0]]])

    cats = [c for c in CATEGORIES if count(questions, c)]
    md += ["", "## Table 2: Correct by category", ""]
    md += table(["Category", *configs], [[f"{cat} ({count(questions, cat)})",
                                          *(correct([r for r in sel[c] if r["category"] == cat]) for c in configs)]
                                         for cat in cats])

    md += ["", "## Table 3: Why answers were wrong (questions with known answer sections)", "",
           f"Correct = score >= {correct_at:.2f}; faithful = every claim supported (or no claims). "
           "Order of checks: correct? then faithful? then were all needed sections given to the writer?", ""]
    att = {c: [r["attribution"] for r in sel[c] if r["attribution"]] for c in configs}
    md += table(["Diagnosis", *configs], [[a, *(rate(att[c].count(a), len(att[c])) for c in configs)]
                                          for a in ATTRIBUTION])

    by = {(r["id"], r["config"]): r for r in rows}
    md += ["", "## Table 4: Per question", "", "C = correctness · F = faithfulness · R = relevance · – = not judged", ""]
    md += table(["ID", "Question", *configs],
                [[q["id"], short(q["question"], 40), *(judged_cell(by[(q["id"], c)]) if (q["id"], c) in by else "–"
                                                      for c in configs)] for q in questions])
    return "\n".join(md) + "\n"
