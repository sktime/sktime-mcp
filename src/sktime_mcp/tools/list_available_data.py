# Implements issue #5: https://github.com/sktime/sktime-mcp/issues/5
#
# Context:
# The codebase had two separate tools for discovering available data:
#   - list_datasets      -> static sktime demo datasets ("airline", "sunspots", etc.)
#   - list_data_handles  -> runtime handles loaded by the user via load_data_source
#
# Problem:
# An LLM asking "what data can I use?" had to call two tools and mentally merge
# the results. This is unnecessary friction and a poor tool UX.
#
# Solution:
# This module introduces list_available_data, a single unified tool that aggregates
# both sources into one response with clear labelling:
#   - "system_demos"    -> always-available built-in datasets
#   - "active_handles"  -> user-loaded data handles for this session
#
# The old tools are kept but marked deprecated, pointing here.
# An optional is_demo boolean lets callers filter to one category if needed.

from typing import Any, Dict, Optional

from sktime_mcp.runtime.executor import get_executor


def list_available_data_tool(is_demo: Optional[bool] = None) -> Dict[str, Any]:
    """
    List all data available for use — system demo datasets and active data handles.

    Aggregates the output of list_datasets and list_data_handles into a single
    unified response, with clear labelling for each category.

    Args:
        is_demo: Optional boolean filter.
            - True  -> return only system demo datasets
            - False -> return only active (user-loaded) data handles
            - None  -> return both (default)

    Returns:
        Dictionary with:
        - success: bool
        - system_demos: list of demo dataset name strings (e.g. ["airline", "lynx", ...])
        - active_handles: list of dicts with handle id and metadata
        - total: int — combined count of items returned
    """
    executor = get_executor()

    system_demos = []
    active_handles = []

    if is_demo is None or is_demo is True:
        system_demos = executor.list_datasets()

    if is_demo is None or is_demo is False:
        handles_result = executor.list_data_handles()
        active_handles = handles_result.get("handles", [])

    return {
        "success": True,
        "system_demos": system_demos,
        "active_handles": active_handles,
        "total": len(system_demos) + len(active_handles),
    }
