"""Grade node (EXPERIMENT, branch exp/yesno-grader): a plain sufficiency check.

One fast-model call per sub-query: are the retrieved passages enough to answer it, yes or no, and if not,
what is missing (a short note that steers Rewrite). Nothing in the reply is checked by code: a "yes" is
taken at its word. Yes -> the sub-query is answered and its retrieved chunks count as relevant (they go
first to the writer); no -> Rewrite, while retries are left. The grader never removes evidence from the
writer. Compare with the needs-and-quotes grader on main (eval/reports).
"""
from __future__ import annotations

from pydantic import BaseModel

from app.llm.structured import call_structured
from app.pipeline.core import Ctx, State

PROMPT = """You check whether passages from Malaysian cyber security law contain enough information to answer
a question.

Question: {query}

Answer "sufficient": true if the passages together state everything needed to fully answer the question,
otherwise false. If false, say in "missing" what information is not in the passages (one short phrase);
if true, leave "missing" empty.

Passages:
{passages}"""


class GradeOutput(BaseModel):
    sufficient: bool
    missing: str = ""


def build_prompt(query: str, chunks) -> str:
    passages = "\n\n".join(f"[{c.id}] {c.label} — {c.heading}\n{c.text}" for c in chunks)
    return PROMPT.format(query=query, passages=passages)


def grade_node(state: State, ctx: Ctx) -> dict:
    ctx.emit("grade", "start", "Checking whether the sources are enough…")
    subs = [s.model_copy() for s in state["sub_queries"]]
    relevant = dict(state.get("relevant", {}))
    calls, report = 0, []
    for i in state["current"]:
        chunks = state["retrieved"].get(i, [])
        if not chunks:
            subs[i].missing = "no passages retrieved"
            report.append({"query": subs[i].query, "sufficient": False, "missing": subs[i].missing})
            continue
        # Grade against what the sub-query needs (its original wording), not the rewritten search text.
        out, n = call_structured(build_prompt(subs[i].original, chunks), GradeOutput, ctx.settings, ctx.timeout())
        calls += n
        if out is None:  # fallback: proceed as sufficient, and say so in the trace
            subs[i].answered, subs[i].missing = True, ""
            report.append({"query": subs[i].query, "sufficient": True, "fallback": True})
            continue
        if out.sufficient:
            for c in chunks:
                relevant.setdefault(c.id, c)
        subs[i].answered = subs[i].partly = subs[i].answered or out.sufficient
        subs[i].missing = "" if out.sufficient else (out.missing.strip() or "not stated")
        report.append({"query": subs[i].query, "sufficient": out.sufficient, "missing": subs[i].missing,
                       "fallback": False})
    enough = all(s.answered for s in subs)
    ctx.emit("grade", "end", "Sources are enough" if enough else "Sources not enough", {"sub_queries": report})
    return {"sub_queries": subs, "relevant": relevant, "llm_calls": state["llm_calls"] + calls}
