"""Streamlit UI. Talks to the backend API only (no direct DB or model access)."""
from __future__ import annotations

import json
import os

import httpx
import streamlit as st

BACKEND = os.getenv("BACKEND_URL", "http://localhost:8000")
READ_TIMEOUT_S = float(os.getenv("BACKEND_READ_TIMEOUT_S", "330"))  # backend enforces its own budget


def stream_chat(question: str, mode: str):
    """Yield (event, data) from the backend's SSE stream."""
    event = None
    # establish connection with server
    # send POST request
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


st.set_page_config(page_title="Cyber Security Act 2024 assistant", page_icon="⚖️")
st.title("Cyber Security Act 2024 assistant")
st.caption("Answers are informational and cite the source text; they are not legal advice.")

with st.sidebar:
    st.subheader("Models in use")
    try:
        health = httpx.get(f"{BACKEND}/health", timeout=5).json()
        m = health["models"]
        st.markdown(f"- Fast: `{m['fast_model']}`\n- Answer: `{m['answer_model']}`\n- Embedding: `{m['embed_model']}`")
        if health["status"] != "ok":
            st.warning("Backend degraded: check /health")
    except httpx.HTTPError:
        st.error(f"Backend unreachable at {BACKEND}")

# Form for asking a question
with st.form("ask"):
    question = st.text_area("Question", placeholder="How quickly must a cyber security incident be reported?")
    mode = st.radio("Mode", ["traditional", "agentic"], horizontal=True)
    submitted = st.form_submit_button("Ask")

# process the question if submitted
if submitted and question.strip():
    answer, citations, metrics, done = None, [], None, None
    with st.status("Starting…") as status:
        try:
            for event, data in stream_chat(question.strip(), mode):
                if event == "trace":
                    status.update(label=data["summary"])
                    st.write(f"**{data['node']}** · {data['status']} · {data['summary']}")
                elif event == "answer":
                    answer = data
                elif event == "citations":
                    citations = data
                elif event == "metrics":
                    metrics = data
                elif event == "done":
                    done = data["status"]
                elif event == "error":
                    st.error(data["message"])
                    done = "error"
        except httpx.HTTPError as e:
            st.error(f"Backend request failed: {e}")
            done = "error"
        status.update(label=f"Finished: {done}", state="error" if done in ("error", "timeout") else "complete")

    if done == "timeout":
        st.warning("The question took too long and was stopped.")
    if answer:
        st.markdown(answer["text"])
        st.caption(answer["label"])
    for c in citations:
        with st.expander(f"{c['section']} · page {c['page']}"):
            chunk = get_chunk(c["chunk_id"])
            st.caption(f"{chunk['document_title']} · {chunk['heading']} · `{c['chunk_id']}`")
            st.text(chunk["text"])
    if metrics:
        st.caption(f"{metrics['latency_ms'] / 1000:.1f} s · {metrics['llm_calls']} LLM call(s) · "
                   f"{metrics['retries']} retries · {metrics['regenerations']} regenerations")
