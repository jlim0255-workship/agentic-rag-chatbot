"""LLM-as-judge for Experiment B: scores saved answers on the RAG triad and writes the Markdown report.

Reads the `-rows.jsonl` files written by `eval.run_e2e` (answers are generated once, judged as often as
needed).

  context relevance  question + chunks given to the writer   share of the chunks that help answer it
  faithfulness       answer + chunks given to the writer      share of the answer's claims the chunks support
  answer relevance   question + answer                        does the answer address the question (yes/partly/no)

Context relevance is judged whenever the writer got chunks; the other two only when an answer was given.
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
from app.pipeline.nodes.generate import NOT_COVERED
from eval.report import DECLINED, judge_report, load_golden, write_md

DEFAULT_JUDGE = "gpt-oss:120b-cloud"
NOT_COVERED_PREFIX = NOT_COVERED.split("{", 1)[0]  # appended by code on the partial path, not by the writer

CONTEXT_RELEVANCE = """You check passages retrieved from Malaysian cyber security law for a QUESTION.
For each PASSAGE, in order, set "relevant" to true if it contains information that helps answer the
question (fully or in part), false if it does not help.

QUESTION: {question}

PASSAGES:
{passages}"""

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


class ContextRelevanceOutput(BaseModel):
    relevant: list[bool]


class Claim(BaseModel):
    claim: str
    supported: bool


class FaithfulnessOutput(BaseModel):
    claims: list[Claim]


class RelevanceOutput(BaseModel):
    addresses: Literal["yes", "partly", "no"]


# --- scoring (pure) ---

RELEVANCE_SCORE = {"yes": 1.0, "partly": 0.5, "no": 0.0}


def declined(row: dict) -> bool:
    """No answer from the documents: not found, out of scope, timeout or error."""
    return row["status"] in (*DECLINED, "timeout", "error")


def faithfulness_score(out: FaithfulnessOutput) -> float | None:
    """Share of claims supported; None if the judge found no claims."""
    return sum(c.supported for c in out.claims) / len(out.claims) if out.claims else None


def writer_text(answer: str) -> str:
    """The part of the answer the writer wrote: drops the not-covered note code appends."""
    return "\n".join(line for line in answer.splitlines() if not line.startswith(NOT_COVERED_PREFIX)).strip()


# --- judging ---

def chunk_texts(settings: Settings, ids: set[str]) -> dict[str, str]:
    with connect(settings.database_url, settings.health_timeout_s) as conn:
        rows = conn.execute("SELECT id, heading, text FROM chunks WHERE id = ANY(%s)", [sorted(ids)]).fetchall()
    return {id_: f"[{id_}] {heading}\n{text}" for id_, heading, text in rows}


def judge_row(row: dict, question: dict, chunks: dict[str, str], settings: Settings) -> dict:
    """Adds context_relevance, faithfulness, relevance (None = not applicable) and `unjudged` (judge calls
    that were invalid twice) to one saved row."""
    timeout = settings.llm_call_timeout_s
    res = {"context_relevance": None, "chunk_verdicts": None, "faithfulness": None, "claims": None,
           "relevance": None, "unjudged": []}
    passed = row["passed"].split()
    texts = [chunks.get(c, f"[{c}] (text not found)") for c in passed]

    if passed:
        schema = ContextRelevanceOutput.model_json_schema()
        schema["properties"]["relevant"] |= {"minItems": len(passed), "maxItems": len(passed)}
        passages = "\n\n".join(f"PASSAGE {i}:\n{t}" for i, t in enumerate(texts, 1))
        out, _ = call_structured(CONTEXT_RELEVANCE.format(question=question["question"], passages=passages),
                                 ContextRelevanceOutput, settings, timeout, json_schema=schema)
        if out is None or len(out.relevant) != len(passed):
            res["unjudged"].append("context_relevance")
        else:
            res["context_relevance"] = sum(out.relevant) / len(passed)
            res["chunk_verdicts"] = dict(zip(passed, out.relevant))
    if declined(row):
        return res  # no answer: nothing to check for faithfulness or relevance

    answer = writer_text(row["answer"])
    out, _ = call_structured(FAITHFULNESS.format(sources="\n\n".join(texts) or "(none)", answer=answer),
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
    parser.add_argument("rows", nargs="+", type=Path, help="-rows.jsonl files from eval.run_e2e")
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
        judged.append({**r, **judge_row(r, questions[r["id"]], chunks, settings),
                       "config": f"{r['model']} {r['mode']}"})

    when = f"{datetime.now():%Y-%m-%d %H:%M}"
    header = [f"# Experiment B judged: RAG triad ({when})", "",
              f"{len(questions)} questions per configuration · judge `{args.judge}` · rows: "
              + ", ".join(f"`{p.name}`" for p in args.rows)]
    slug = "".join(ch if ch.isalnum() or ch == "." else "-" for ch in args.judge)
    print(f"report: {write_md(f'judge-{slug}', judge_report(judged, list(questions.values()), header))}")


if __name__ == "__main__":
    main()
