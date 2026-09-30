"""Rewrite node (PRD 10.2): new searches for each sub-query still missing evidence.

Escalates one technique per retry (one LLM call per unanswered sub-query per retry):
  retry 1: paraphrase + expansion into formal legal terms, searched together (multi-query, fused by RRF)
  retry 2: step-back, a broader question about the concept behind it

Drift gate: each variant's cosine similarity to its sub-query is recorded in the trace. Variants below
REWRITE_MIN_SIMILARITY are rejected like repeats (0 = gate off until calibrated from recorded values).
"""
from __future__ import annotations

import math

from pydantic import BaseModel

from app.llm import ollama
from app.llm.structured import call_structured
from app.pipeline.core import Ctx, State
from app.retrieval.search import Filters

CONTEXT = """A search over Malaysia's Cyber Security Act 2024 and its regulations did not find enough evidence.

Question: {question}
Search query that failed: {failed}
What is missing: {missing}
Queries already tried (do not repeat any of them):
{history}

"""

RETRY_1 = CONTEXT + """Write two new search queries:
- "paraphrase": the same question in different everyday words.
- "expanded": the question rewritten with the formal terms and synonyms a statute would use
  (the defined legal terms rather than casual words). Keep both short."""

RETRY_2 = CONTEXT + """Write one "step_back" query: a broader, more general question about the legal concept behind
this question, e.g. instead of "must my bakery register?" ask "which organisations must register under the law?"."""


class Retry1Output(BaseModel):
    paraphrase: str
    expanded: str


class Retry2Output(BaseModel):
    step_back: str


TECHNIQUES = [("paraphrase + expansion", RETRY_1, Retry1Output), ("step-back", RETRY_2, Retry2Output)]


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b)) / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


def similarities(original: str, variants: list[str], ctx: Ctx) -> dict[str, float]:
    """Cosine similarity of each variant to the sub-query, in one embedding call."""
    vectors = ollama.embed(ctx.settings.ollama_base_url, ctx.settings.embed_model, [original, *variants], ctx.timeout())
    return {v: round(cosine(vectors[0], vec), 3) for v, vec in zip(variants, vectors[1:])}


def rewrite_node(state: State, ctx: Ctx) -> dict:
    # ponytail: retries beyond the listed techniques reuse the last one; add techniques if MAX_RETRIES > 2.
    technique, prompt, schema = TECHNIQUES[min(state["retry_count"], len(TECHNIQUES) - 1)]
    ctx.emit("rewrite", "start", f"Rewriting the search ({technique})…")
    subs = [s.model_copy() for s in state["sub_queries"]]
    history = list(state["query_history"])
    calls, current, report = 0, [], []
    for i in state["current"]:
        s = subs[i]
        if s.answered or s.exhausted:
            continue
        new: list[str] = []
        sims: dict[str, float] = {}
        for _ in range(2):  # PRD: repeats (and drifted variants) are rejected, with one more attempt before giving up
            out, n = call_structured(
                prompt.format(question=s.original, failed=s.query, missing=s.missing or "not stated",
                              history="\n".join(f"- {q}" for q in history)),
                schema, ctx.settings, ctx.timeout())
            calls += n
            seen = {_norm(q) for q in history}
            fresh = [q for q in dict.fromkeys(v.strip() for v in (out.model_dump().values() if out else []))
                     if q and _norm(q) not in seen]
            if fresh:
                sims.update(similarities(s.original, fresh, ctx))
            new = [q for q in fresh if sims[q] >= ctx.settings.rewrite_min_similarity]
            if new:
                break
        if not new:
            s.exhausted = True
            report.append({"technique": technique, "from": s.query, "to": [], "gave_up": True, "similarity": sims})
            continue
        report.append({"technique": technique, "from": s.query, "to": new, "gave_up": False,
                       "similarity": sims, "filters_removed": s.filters.model_dump()})
        s.query, s.variants = new[0], new
        s.filters = Filters()  # a wrong planner filter can hide the answer: retry without it (PRD allows removal)
        history += new
        current.append(i)
    ctx.emit("rewrite", "end", f"{technique}: {len(current)} sub-quer{'y' if len(current) == 1 else 'ies'} rewritten",
             {"rewrites": report})
    return {"sub_queries": subs, "current": current, "query_history": history,
            "retry_count": state["retry_count"] + 1, "llm_calls": state["llm_calls"] + calls}
