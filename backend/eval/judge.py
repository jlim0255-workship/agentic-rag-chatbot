"""LLM-as-judge for Experiment B: scores saved answers on faithfulness, answer relevance and correctness.

Reads the `-rows.jsonl` files written by `eval.run e2e` / `eval.run_cloud` (answers are generated once,
judged as often as needed). Retrieval quality is not judged here: FULL/PART/MISS against the golden
sections already measures it exactly.

  faithfulness  answer + chunks given to the writer   share of the answer's claims the chunks support
  relevance     question + answer                     does the answer address the question (yes/partly/no)
  correctness   question + answer + golden notes      share of the expected key facts covered

Eval only: a `-cloud` judge sends questions, answers and chunks to Ollama Cloud. The shipped system stays local.

Usage (from backend/; DB ingested, Ollama running):
    python -m eval.judge eval/reports/<stem>-rows.jsonl [more rows files] [--judge gpt-oss:120b-cloud]
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.config import Settings, get_settings
from app.db.connection import connect
from app.llm import ollama
from app.llm.structured import call_structured
from eval.report import DECLINED, judge_report
from eval.run import load_golden, save

DEFAULT_JUDGE = "gpt-oss:120b-cloud"
CORRECT_AT = 0.75  # correctness score at or above which an answer counts as correct in the attribution table
CONTACTS = "**Who to contact**"  # staff list appended by code, not by the writer: not judged

FAITHFULNESS = """You check an answer that was written only from the legal source extracts below.
Split the ANSWER into its separate factual claims (ignore the [chunk id] citations and polite filler).
For each claim set "supported" to true only if the SOURCES state it (paraphrase is fine); false if the
sources do not say it or say something different.

SOURCES:
{sources}

ANSWER:
{answer}"""

RELEVANCE = """Does the ANSWER respond to the QUESTION? Judge only whether it addresses what was asked, not
whether it is true. "yes" = it answers the question; "partly" = it answers only part of it or mostly talks
about something else; "no" = it does not answer the question (a refusal or "not found" is "no").

QUESTION: {question}

ANSWER:
{answer}"""

CORRECTNESS = """Compare the ANSWER with the KEY FACTS a correct answer contains. For each key fact, in order,
give one verdict: "covered" = the answer states it (paraphrase is fine); "partly" = only part of it, or
vaguer; "missing" = not in the answer; "contradicted" = the answer states something different.

QUESTION: {question}

KEY FACTS:
{facts}

ANSWER:
{answer}"""


class Claim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessOutput(BaseModel):
    claims: list[Claim]


class RelevanceOutput(BaseModel):
    addresses: Literal["yes", "partly", "no"]


Verdict = Literal["covered", "partly", "missing", "contradicted"]


class CorrectnessOutput(BaseModel):
    verdicts: list[Verdict]


# --- scoring (pure) ---

RELEVANCE_SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}
FACT_SCORE = {"covered": 1.0, "partly": 0.5, "missing": 0.0, "contradicted": 0.0}


def declined(row: dict) -> bool:
    """No answer from the documents: not found, out of scope, timeout or error."""
    return row["status"] in (*DECLINED, "timeout", "error")


def faithfulness_score(out: FaithfulnessOutput) -> float | None:
    """Share of claims supported; None if the judge found no claims."""
    return sum(c.supported for c in out.claims) / len(out.claims) if out.claims else None


def correctness_score(verdicts: list[str]) -> float:
    return sum(FACT_SCORE[v] for v in verdicts) / len(verdicts)


def attribute(row: dict) -> str:
    """Why a question with known answer sections did or did not get a correct answer."""
    correct = (row["correctness"] or 0) >= CORRECT_AT
    faithful = row["faithfulness"] is None or row["faithfulness"] == 1.0
    if correct:
        return "Success" if faithful else "Correct but not grounded"
    if not faithful:
        return "Hallucination"
    return "Writer missed facts it had" if row["outcome"]["level"] == "full" else "Retrieval failure"


# --- judging ---

def strip_contacts(answer: str) -> str:
    return answer.split(CONTACTS, 1)[0].strip()


def chunk_texts(settings: Settings, ids: set[str]) -> dict[str, str]:
    with connect(settings.database_url, settings.health_timeout_s) as conn:
        rows = conn.execute("SELECT id, heading, text FROM chunks WHERE id = ANY(%s)", [sorted(ids)]).fetchall()
    return {id_: f"[{id_}] {heading}\n{text}" for id_, heading, text in rows}


def judge_row(row: dict, question: dict, chunks: dict[str, str], settings: Settings) -> dict:
    """Adds faithfulness, relevance, correctness (None = not applicable) and `unjudged` (judge calls that
    were invalid twice) to one saved row."""
    answer, timeout = strip_contacts(row["answer"]), settings.llm_call_timeout_s
    res = {"faithfulness": None, "claims": None, "relevance": None, "correctness": None, "verdicts": None,
           "unjudged": []}
    if question["category"] == "not_in_corpus" and declined(row):
        res["correctness"] = 1.0  # declining is the correct answer; nothing to judge
        return res
    if declined(row):
        res["correctness"] = 0.0  # an answerable question got no answer
        return res

    sources = "\n\n".join(chunks.get(c, f"[{c}] (text not found)") for c in row["passed"].split())
    out, _ = call_structured(FAITHFULNESS.format(sources=sources or "(none)", answer=answer),
                             FaithfulnessOutput, settings, timeout)
    if out is None:
        res["unjudged"].append("faithfulness")
    else:
        res["faithfulness"], res["claims"] = faithfulness_score(out), [c.model_dump() for c in out.claims]

    out, _ = call_structured(RELEVANCE.format(question=question["question"], answer=answer),
                             RelevanceOutput, settings, timeout)
    if out is None:
        res["unjudged"].append("relevance")
    else:
        res["relevance"] = RELEVANCE_SCORE[out.addresses]

    if question["category"] == "not_in_corpus":
        res["correctness"] = 0.0  # answered a question the documents cannot answer
        return res
    notes = question["answer_notes"]
    schema = CorrectnessOutput.model_json_schema()
    schema["properties"]["verdicts"] |= {"minItems": len(notes), "maxItems": len(notes)}
    out, _ = call_structured(CORRECTNESS.format(question=question["question"], answer=answer,
                                                facts="\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1))),
                             CorrectnessOutput, settings, timeout, json_schema=schema)
    if out is None or len(out.verdicts) != len(notes):
        res["unjudged"].append("correctness")
    else:
        res["correctness"], res["verdicts"] = correctness_score(out.verdicts), out.verdicts
    return res


def preflight(settings: Settings) -> None:
    try:
        ollama.chat(settings.ollama_base_url, settings.fast_model, "Reply with OK.",
                    settings.llm_call_timeout_s, settings.llm_num_ctx)
    except ollama.OllamaError as e:
        raise SystemExit(f"Judge model {settings.fast_model!r} is not usable: {e}\n"
                         f"Run `ollama pull {settings.fast_model}` (and `ollama signin` for a cloud model).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("rows", nargs="+", type=Path, help="-rows.jsonl files from eval.run / eval.run_cloud")
    parser.add_argument("--judge", default=DEFAULT_JUDGE, help=f"judge model (default: {DEFAULT_JUDGE})")
    args = parser.parse_args()

    settings = get_settings().model_copy(update={"fast_model": args.judge})
    preflight(settings)
    questions = {q["id"]: q for q in load_golden()}
    rows = [json.loads(line) for p in args.rows for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    chunks = chunk_texts(settings, {c for r in rows for c in r["passed"].split()})
    judged = []
    for i, r in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {r['model']} {r['mode']} {r['id']}", flush=True)
        judged.append({**r, **judge_row(r, questions[r["id"]], chunks, settings)})
    for r in judged:
        r["config"] = f"{r['model']} {r['mode']}"
        r["attribution"] = attribute(r) if r["outcome"] else None

    when = f"{datetime.now():%Y-%m-%d %H:%M}"
    header = [f"# Experiment B judged: faithfulness, relevance, correctness ({when})", "",
              f"{len(questions)} questions per configuration · judge `{args.judge}` · rows: "
              + ", ".join(f"`{p.name}`" for p in args.rows)]
    slug = "".join(ch if ch.isalnum() or ch == "." else "-" for ch in args.judge)
    stem = save(f"judge-{slug}", judge_report(judged, list(questions.values()), header, CORRECT_AT), judged)
    with stem.with_name(f"{stem.name}-rows.jsonl").open("w", encoding="utf-8") as f:
        for r in judged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"report: {stem}.md")


if __name__ == "__main__":
    main()
