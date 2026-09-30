"""Tool agent node (new_prd.md): the model decides whether to call a tool, using ChatOllama.bind_tools.
Its reply (an AIMessage, with or without tool_calls) goes into `messages`; tools_condition and ToolNode
(flow.py) do the rest. No tool call = no staff lookup; the law search runs either way.
"""
from __future__ import annotations

import httpx
from langchain_ollama import ChatOllama

from app.llm.ollama import OllamaTimeout
from app.pipeline.core import Ctx, State
from app.pipeline.tools import lookup_staff

PROMPT = """A user asks about Malaysian cyber security law. If they also need to know who to contact or who is
responsible in their organisation, call lookup_staff with the matching topics. Otherwise do not call any tool.

Question: {question}"""


def tool_agent_node(state: State, ctx: Ctx) -> dict:
    ctx.emit("tool_agent", "start", "Deciding whether to look up contacts…")
    s = ctx.settings
    model = ChatOllama(model=s.fast_model, base_url=s.ollama_base_url, num_ctx=s.llm_num_ctx, reasoning=False,
                       client_kwargs={"timeout": ctx.timeout()}).bind_tools([lookup_staff])
    try:
        msg = model.invoke(PROMPT.format(question=state["question"]))
    except httpx.TimeoutException as e:
        raise OllamaTimeout(f"tool_agent: {e}") from e
    ctx.emit("tool_agent", "end",
             f"calls {', '.join(c['name'] for c in msg.tool_calls)}" if msg.tool_calls else "no tool needed",
             {"tool_calls": [{"name": c["name"], "args": c["args"]} for c in msg.tool_calls]})
    return {"messages": [msg], "llm_calls": state["llm_calls"] + 1}
