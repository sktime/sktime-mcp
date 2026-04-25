"""
MCP Tool: get_agent_trace

Provides a structured execution trace for an agent session.
Useful for debugging multi-step agent workflows involving MCP tools.
"""
import time
from datetime import datetime
from typing import Dict, Any, List

# Simple in-memory trace store
TRACE_STORE: Dict[str, List[Dict[str, Any]]] = {}


def record_trace(session_id: str, tool: str, input_data: dict, output_data: dict, start_time: float):
    """
    Record a step in an agent session.
    """

    end_time = time.time()

    if session_id not in TRACE_STORE:
        TRACE_STORE[session_id] = []

    step = {
        "step": len(TRACE_STORE[session_id]) + 1,
        "tool": tool,
        "input": input_data,
        "output": output_data,
        "timestamp": datetime.utcnow().isoformat(),
        "execution_time_seconds": round(end_time - start_time, 4),
    }

    TRACE_STORE[session_id].append(step)


def get_agent_trace(session_id: str) -> Dict[str, Any]:
    """
    Retrieve the full execution trace for a session.

    Parameters
    ----------
    session_id : str
        The agent session identifier.

    Returns
    -------
    dict
        Structured trace of tool calls for the session.
    """

    return {
        "session_id": session_id,
        "steps": TRACE_STORE.get(session_id, []),
    }