"""Generate node: answer from the given chunks only, citing chunk IDs (PRD 10.2). Shared by both modes."""
from __future__ import annotations

import re

from app.config import Settings
from app.llm import ollama
from app.retrieval.search import Retrieved

NOT_FOUND = "NOT_FOUND"

PROMPT = """You answer questions about Malaysia's Cyber Security Act 2024 and its regulations.
Use ONLY the sources below. Never use outside knowledge.

Rules:
- After every sentence that states a fact, put the ID of the source it comes from in square brackets, e.g. [act854:s23:1].
- Only use IDs that appear in the sources.
- If the sources do not answer the question, reply with exactly: {not_found}
- Be concise and plain. Quote numbers and time limits exactly as the source states them.

Sources:
{sources}

Question: {question}
Answer:"""

BRACKETS = re.compile(r"\[([^\]]+)\]")
CHUNK_ID = re.compile(r"[a-z0-9_]+:[a-z0-9_]+:\d+")


def build_prompt(question: str, chunks: list[Retrieved]) -> str:
    sources = "\n\n".join(f"[{c.id}] {c.label} — {c.heading}\n{c.text}" for c in chunks)
    return PROMPT.format(not_found=NOT_FOUND, sources=sources, question=question)


def generate(question: str, chunks: list[Retrieved], settings: Settings, timeout_s: float) -> str:
    return ollama.chat(settings.ollama_base_url, settings.answer_model, build_prompt(question, chunks),
                       timeout_s, settings.llm_num_ctx).strip()


def cited_ids(answer: str) -> list[str]:
    """Chunk IDs cited in the answer, in order of first use.

    Tolerates what the small model actually writes: '[a:s1:1, a:s2:1]' and '[act854:s22:1(1)(a)]'.
    """
    ids = (i for group in BRACKETS.findall(answer) for i in CHUNK_ID.findall(group))
    return list(dict.fromkeys(ids))
