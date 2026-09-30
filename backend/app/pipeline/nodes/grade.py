"""Grade node: needs coverage with verbatim evidence (replaces PRD 10.2's per-chunk relevance rule).

One fast-model call per sub-query lists the pieces of information the sub-query needs and, for each,
the chunk and the exact words that state it. CODE decides: a need is covered only if the chunk was
retrieved and the quote appears word for word in its text. Sufficient = every need covered.
The grader only steers (enough -> generate, else rewrite); it never removes evidence from the writer.
"""
from __future__ import annotations

from pydantic import BaseModel

from app.llm.structured import call_structured
from app.pipeline.core import Ctx, State

PROMPT = """You check whether passages from Malaysian cyber security law contain what a question needs.

Question: {query}

List the separate pieces of information needed to fully answer the question (1 to 4 "needs").
For each need:
- if a passage states it, set "chunk_id" to that passage's id and copy the exact words from the passage
  that state it into "quote" (verbatim, one phrase or sentence);
- if no passage states it, set "chunk_id" and "quote" to null.

Passages:
{passages}"""

MIN_QUOTE_CHARS = 8  # ponytail: shorter quotes ("the", "Act") match anything; raise if the grader games it


class Need(BaseModel):
    need: str
    chunk_id: str | None = None
    quote: str | None = None


class GradeOutput(BaseModel):
    needs: list[Need]


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def build_prompt(query: str, chunks) -> str:
    passages = "\n\n".join(f"[{c.id}] {c.label} — {c.heading}\n{c.text}" for c in chunks)
    return PROMPT.format(query=query, passages=passages)


def schema_for(chunk_ids: list[str]) -> dict:
    """GradeOutput's JSON schema with chunk_id restricted to the retrieved IDs (or null), and the prompt's
    "1 to 4 needs" enforced by decoding (the model ignored it in text: 14 needs on one question)."""
    schema = GradeOutput.model_json_schema()
    schema["$defs"]["Need"]["properties"]["chunk_id"] = {"anyOf": [{"enum": chunk_ids}, {"type": "null"}]}
    schema["properties"]["needs"].update(minItems=1, maxItems=4)
    return schema


def verify(needs: list[Need], chunks) -> list[dict]:
    """Code's verdict per need: covered only if the chunk was retrieved and the quote is in its text."""
    text = {c.id: _norm(c.text) for c in chunks}
    out = []
    for n in needs:
        q = _norm(n.quote or "")
        covered = n.chunk_id in text and len(q) >= MIN_QUOTE_CHARS and q in text[n.chunk_id]
        out.append({"need": n.need, "chunk_id": n.chunk_id, "quote": n.quote, "covered": covered})
    return out


def grade_node(state: State, ctx: Ctx) -> dict:
    ctx.emit("grade", "start", "Checking what the sources cover…")
    subs = [s.model_copy() for s in state["sub_queries"]]
    relevant = dict(state.get("relevant", {}))
    calls, report = 0, []
    for i in state["current"]:
        chunks = state["retrieved"].get(i, [])
        if not chunks:
            subs[i].missing = "no passages retrieved"
            report.append({"query": subs[i].query, "needs": [], "missing": subs[i].missing})
            continue
        # Grade against what the sub-query needs (its original wording), not the rewritten search text.
        out, n = call_structured(build_prompt(subs[i].original, chunks), GradeOutput, ctx.settings,
                                 ctx.timeout(), json_schema=schema_for([c.id for c in chunks]))
        calls += n
        if out is None or not out.needs:  # fallback: proceed as sufficient, and say so in the trace
            subs[i].answered, subs[i].missing = True, ""
            report.append({"query": subs[i].query, "needs": [], "fallback": True})
            continue
        needs = verify(out.needs, chunks)
        by_id = {c.id: c for c in chunks}
        for n_ in needs:
            if n_["covered"]:
                relevant.setdefault(n_["chunk_id"], by_id[n_["chunk_id"]])
        uncovered = [n_["need"] for n_ in needs if not n_["covered"]]
        subs[i].answered = subs[i].answered or not uncovered
        subs[i].partly = subs[i].partly or len(uncovered) < len(needs)
        subs[i].missing = "; ".join(uncovered)
        report.append({"query": subs[i].query, "needs": needs, "missing": subs[i].missing, "fallback": False})
    enough = all(s.answered for s in subs)
    ctx.emit("grade", "end", "All needs covered" if enough else "Needs still missing", {"sub_queries": report})
    return {"sub_queries": subs, "relevant": relevant, "llm_calls": state["llm_calls"] + calls}
