"""Streamlit UI. Talks to the backend API only (no direct DB or model access).

Chat layout: every question and answer in this browser tab stays on the page (st.session_state).
Display only: each question is still answered on its own; the backend never sees earlier turns.
"""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")
READ_TIMEOUT_S = float(os.getenv("BACKEND_READ_TIMEOUT_S", "330"))  # backend enforces its own budget


def stream_chat(question: str, mode: str):
    """Yield (event, data) from the backend's SSE stream."""
    event = None
    with httpx.stream("POST", f"{BACKEND}/chat", json={"question": question, "mode": mode},
                      timeout=httpx.Timeout(10, read=READ_TIMEOUT_S)) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if line.startswith("event: "):
                event = line[7:]
            elif line.startswith("data: "):
                yield event, json.loads(line[6:])


@st.cache_data(show_spinner=False)
def get_chunk(chunk_id: str) -> dict:
    return httpx.get(f"{BACKEND}/chunks/{chunk_id}", timeout=10).json()


def ask(question: str, mode: str) -> dict:
    """Run one question, showing each step live, and return the finished turn."""
    turn = {"question": question, "mode": mode, "answer": None, "citations": [], "metrics": None,
            "done": None, "trace": [], "error": None}
    with st.status("Starting…", expanded=True) as status:
        try:
            for event, data in stream_chat(question, mode):
                if event == "trace":
                    status.update(label=data["summary"])
                    if data["status"] == "end":
                        turn["trace"].append(data)
                        st.write(f"**{data['node']}** · {data['summary']}")
                elif event in ("answer", "citations", "metrics"):
                    turn[event] = data
                elif event == "done":
                    turn["done"] = data["status"]
                elif event == "error":
                    turn["error"], turn["done"] = data["message"], "error"
        except httpx.HTTPError as e:
            turn["error"], turn["done"] = f"Backend request failed: {e}", "error"
    return turn


def show(turn: dict) -> None:
    """Draw one finished question and its answer."""
    with st.chat_message("user"):
        st.markdown(turn["question"])
    with st.chat_message("assistant"):
        if turn["error"]:
            st.error(turn["error"])
        if turn["done"] == "timeout":
            st.warning("The question took too long and was stopped.")
        if turn["answer"]:
            st.markdown(turn["answer"]["text"])
            st.caption(turn["answer"]["label"])
        m = turn["metrics"]
        if m:
            st.caption(f"{turn['mode']} mode · {m['latency_ms'] / 1000:.1f} s · {m['llm_calls']} LLM call(s) · "
                       f"{m['retries']} retries")
        if turn["citations"]:
            with st.expander(f"Sources ({len(turn['citations'])})"):
                for c in turn["citations"]:
                    chunk = get_chunk(c["chunk_id"])
                    st.markdown(f"**{c['section']}** · page {c['page']} · {chunk['heading']} · `{c['chunk_id']}`")
                    st.text(chunk["text"])
        if turn["trace"]:
            with st.expander(f"Trace ({len(turn['trace'])} steps)"):
                for step in turn["trace"]:
                    st.markdown(f"**{step['node']}** · {step['summary']}")
                    if step["data"]:
                        st.json(step["data"], expanded=False)


st.set_page_config(page_title="Cyber Security Act 2024 assistant", page_icon="⚖️")
st.title("Cyber Security Act 2024 assistant")
st.caption("Answers are informational and cite the source text; they are not legal advice.")

if "turns" not in st.session_state:
    st.session_state.turns = []

with st.sidebar:
    mode = st.radio("Mode", ["traditional", "agentic"], help="Agentic plans, grades, retries and checks its sources.")
    if st.button("Clear conversation"):
        st.session_state.turns = []
    st.subheader("Models in use")
    try:
        health = httpx.get(f"{BACKEND}/health", timeout=5).json()
        m = health["models"]
        st.markdown(f"- Fast: `{m['fast_model']}`\n- Answer: `{m['answer_model']}`\n- Embedding: `{m['embed_model']}`")
        if health["status"] != "ok":
            st.warning("Backend degraded: check /health")
    except httpx.HTTPError:
        st.error(f"Backend unreachable at {BACKEND}")

for past in st.session_state.turns:
    show(past)

st.caption("Each question is answered on its own, without earlier turns: include the details you need.")
if question := st.chat_input("Ask about the Cyber Security Act 2024…"):
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        st.session_state.turns.append(ask(question.strip(), mode))
    st.rerun()  # redraw from history so the new turn looks like the others
