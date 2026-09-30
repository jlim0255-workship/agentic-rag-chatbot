"""Generate node (PRD 10.2): the writer. Answers from the given chunks only, citing chunk IDs.
Shared by both modes. Uses the answer model; the only node that writes text for the user."""
from __future__ import annotations

from app.llm import ollama
from app.pipeline.core import NOT_FOUND, Ctx, State
from app.retrieval.search import Retrieved

PROMPT = """You answer questions about Malaysia's Cyber Security Act 2024 and its regulations.
Use ONLY the sources below. Never use outside knowledge.

Rules:
- After every sentence that states a fact, put the ID of the source it comes from in square brackets, e.g. [act854:s23:1].
- Only use IDs that appear in the sources.
- If the sources do not answer the question, reply with exactly: {not_found}
- Be concise and plain. Quote numbers and time limits exactly as the source states them.

Sources:
{sources}
{unanswered}
Question: {question}
Answer:"""

UNANSWERED = """
These parts of the question could not be found in the documents:
{parts}
The sources DO answer the rest of the question (this was checked), so do not reply {not_found}. Answer the
rest with citations, and say in one sentence that the parts above are not covered by the provided documents.
"""


def build_prompt(question: str, chunks: list[Retrieved], unanswered: list[str] | None = None) -> str:
    sources = "\n\n".join(f"[{c.id}] {c.label} — {c.heading}\n{c.text}" for c in chunks)
    note = UNANSWERED.format(parts="\n".join(f"- {p}" for p in unanswered), not_found=NOT_FOUND) if unanswered else ""
    return PROMPT.format(not_found=NOT_FOUND, sources=sources, unanswered=note, question=question)


def generate_node(state: State, ctx: Ctx) -> dict:
    # Steer, don't delete: chunks with a verified quote first, then everything else retrieved, capped.
    # (Traditional mode has no grader, so `relevant` is empty and the writer gets what was retrieved.)
    verified = list(state["relevant"].values())
    others = [c for id_, c in state["supporting"].items() if id_ not in state["relevant"]]
    chunks = (verified + others)[:max(ctx.settings.writer_max_chunks, len(verified))]
    partial = state["status"] == "partial"
    # Name only sub-queries with no verified evidence. A partly covered one is not named: its `missing` note
    # reflects only the last attempt and can list a need an earlier attempt verified (the writer then says
    # NOT_FOUND despite holding the evidence). The writer still replies NOT_FOUND if its sources don't answer.
    unanswered = [s.original for s in state["sub_queries"] if not s.answered and not s.partly] if partial else []
    ctx.emit("generate", "start", "Writing the answer…", {"chunks": [c.id for c in chunks], "unanswered": unanswered})
    text = ollama.chat(ctx.settings.ollama_base_url, ctx.settings.answer_model,
                       build_prompt(state["question"], chunks, unanswered), ctx.timeout(),
                       ctx.settings.llm_num_ctx).strip()
    ctx.emit("generate", "end", "Answer written")
    return {"answer": text, "passed": chunks, "llm_calls": state["llm_calls"] + 1}
