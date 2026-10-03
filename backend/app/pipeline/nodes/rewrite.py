"""Rewrite node: a new search query for each sub-query still missing evidence.

Escalates one technique per retry (one LLM call per unanswered sub-query per retry, one new query each):
  retry 1: expansion into the formal legal terms a statute would use
  retry 2: step-back, a broader question about the concept behind it
"""
from __future__ import annotations

from pydantic import BaseModel

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

RETRY_1 = CONTEXT + """Write one "expanded" query: the question rewritten with the formal terms and synonyms a statute
would use (the defined legal terms rather than casual words). Keep it short."""

RETRY_2 = CONTEXT + """Write one "step_back" query: a broader, more general question about the legal concept behind
this question, e.g. instead of "must my bakery register?" ask "which organisations must register under the law?"."""


class Retry1Output(BaseModel):
    expanded: str


class Retry2Output(BaseModel):
    step_back: str


TECHNIQUES = [("expansion", RETRY_1, Retry1Output), ("step-back", RETRY_2, Retry2Output)]


def _norm(q: str) -> str:
    return " ".join(q.lower().split())


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
        seen = {_norm(q) for q in history}
        new = ""
        for _ in range(2):  # a repeat is rejected, with one more attempt before giving up
            out, n = call_structured(
                prompt.format(question=s.original, failed=s.query, missing=s.missing or "not stated",
                              history="\n".join(f"- {q}" for q in history)),
                schema, ctx.settings, ctx.timeout())
            calls += n
            new = next(iter(out.model_dump().values())).strip() if out else ""  # each schema has one field
            if new and _norm(new) not in seen:
                break
            new = ""
        if not new:
            s.exhausted = True
            report.append({"technique": technique, "from": s.query, "to": "", "gave_up": True})
            continue
        report.append({"technique": technique, "from": s.query, "to": new, "gave_up": False,
                       "filters_removed": s.filters.model_dump()})
        s.query = new
        s.filters = Filters()  # a wrong planner filter can hide the answer: retry without it
        history.append(new)
        current.append(i)
    ctx.emit("rewrite", "end", f"{technique}: {len(current)} sub-quer{'y' if len(current) == 1 else 'ies'} rewritten",
             {"rewrites": report})
    return {"sub_queries": subs, "current": current, "query_history": history,
            "retry_count": state["retry_count"] + 1, "llm_calls": state["llm_calls"] + calls}
