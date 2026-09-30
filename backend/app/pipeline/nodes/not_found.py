"""Not-found node (PRD 10.2). No LLM. Never lets the model answer from its own knowledge.

Runs when some needs are still uncovered after the retries. If ANY evidence was verified by a word-for-word
quote, the writer answers from it and says which parts are not covered (partial). Only with no verified
evidence at all is the answer "not found". (The grader sometimes invents extra needs it cannot verify;
before this rule, one such need threw away verified evidence, e.g. s23 for "penalty for not reporting".)
"""
from __future__ import annotations

from app.pipeline.core import Ctx, State


def not_found_node(state: State, ctx: Ctx) -> dict:
    subs = state["sub_queries"]
    missing = [s.missing or s.original for s in subs if not s.answered]
    # answered: every need covered (or the grader's fallback accepted it); relevant: any verified quote at all
    if not state.get("relevant") and not any(s.answered for s in subs):
        ctx.emit("not_found", "end", "No verified evidence after retries", {"unanswered": missing})
        return {"status": "not_found"}
    ctx.emit("not_found", "end", "Some parts not found; answering from the verified evidence", {"unanswered": missing})
    return {"status": "partial"}
