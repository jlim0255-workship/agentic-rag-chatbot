"""Tools the agent may call. Run by LangGraph's ToolNode; chosen by the tool_agent node.

lookup_staff reads FICTIONAL demo data. Its result is appended under the answer by code (flow.run);
the writer never sees it, so names and numbers cannot be made up.
"""
from __future__ import annotations

from langchain_core.tools import tool
from langgraph.config import get_stream_writer

from app.config import get_settings
from app.db.connection import connect
from app.pipeline.core import trace


@tool
def lookup_staff(topics: list[str]) -> str:
    """Find who to contact in the organisation. topics: one or more of incident_response, ncii_notification,
    legal, data_protection, communications, management."""
    settings = get_settings()
    # the docstring here is the prompt the model sees; the code below is never seen by the model, so it cannot make up names or numbers.
    with connect(settings.staff_database_url, settings.health_timeout_s) as conn:  # employee_demo, not the law DB
        # Unknown topics simply match nothing (array overlap), so no extra validation is needed.
        rows = conn.execute("SELECT name, role, email, phone, hours FROM demo_staff WHERE topics && %s ORDER BY id",
                            [topics]).fetchall()
    get_stream_writer()(trace("tools", "end", f"lookup_staff: {len(rows)} contact(s)",
                              {"topics": topics, "names": [r[0] for r in rows]}))
    if not rows:
        return ""
    return "\n".join(["**Who to contact** (fictional demo staff directory; copied from the table, not from the law):",
                      *(f"- {name}, {role} · {email} · {phone} · {hours}" for name, role, email, phone, hours in rows)])
