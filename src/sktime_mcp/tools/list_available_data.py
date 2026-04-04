from typing import Any, Optional

from sktime_mcp.runtime.executor import get_executor


def list_available_data_tool(is_demo: Optional[bool] = None) -> dict[str, Any]:
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
