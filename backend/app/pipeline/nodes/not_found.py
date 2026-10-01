"""Not-found node. No LLM. Never lets the model answer from its own knowledge.

Runs when some sub-queries are still not judged sufficient after the retries. If ANY sub-query was judged
sufficient, the writer answers from that evidence and says which parts are not covered (partial). Only with
no sufficient evidence at all is the answer "not found".
"""
from __future__ import annotations

from app.pipeline.core import Ctx, State


def not_found_node(state: State, ctx: Ctx) -> dict:
    subs = state["sub_queries"]
    missing = [s.missing or s.original for s in subs if not s.answered]
    # answered: judged sufficient (or the grader's fallback accepted it); relevant: chunks of any sufficient sub-query
    if not state.get("relevant") and not any(s.answered for s in subs):
        ctx.emit("not_found", "end", "No sufficient evidence after retries", {"unanswered": missing})
        return {"status": "not_found"}
    ctx.emit("not_found", "end", "Some parts not found; answering from the evidence found", {"unanswered": missing})
    return {"status": "partial"}
