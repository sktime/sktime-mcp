from typing import Any, Optional

from sktime_mcp.runtime.executor import get_executor


def list_available_data_tool(
    is_demo: Optional[bool] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    List all data available for use — system demo datasets and active data handles.

    Aggregates the output of list_datasets and list_data_handles into a single
    unified response, with clear labelling for each category.
    Supports pagination via limit and offset to handle large numbers of datasets.

    Args:
        is_demo: Optional boolean filter.
            - True  -> return only system demo datasets
            - False -> return only active (user-loaded) data handles
            - None  -> return both (default)
        limit: Maximum number of combined results to return (default: 50).
        offset: Number of combined results to skip for pagination (default: 0).
            Use with limit to paginate through all available data.
            Example: offset=50 returns results 51-100.

    Returns:
        Dictionary with:
        - success: bool
        - system_demos: list of demo dataset name strings (e.g. ["airline", "lynx", ...])
        - active_handles: list of dicts with handle id and metadata
        - total: int — combined count of ALL items (before pagination)
        - count: int — number of items returned in this page
        - offset: int — current offset used
        - limit: int — current limit used
        - has_more: bool — True if more results exist beyond this page

    Example:
        >>> list_available_data_tool(limit=10, offset=0)
        {
            "success": True,
            "system_demos": ["airline", "lynx", ...],
            "active_handles": [...],
            "total": 25,
            "count": 10,
            "offset": 0,
            "limit": 10,
            "has_more": True
        }
    """
    if offset < 0:
        return {
            "success": False,
            "error": "offset must be a non-negative integer.",
        }

    executor = get_executor()

    system_demos = []
    active_handles = []

    if is_demo is None or is_demo is True:
        system_demos = executor.list_datasets()

    if is_demo is None or is_demo is False:
        handles_result = executor.list_data_handles()
        active_handles = handles_result.get("handles", [])

    # Combine all items into a flat list for unified pagination
    all_demos = [{"name": d, "type": "system_demo"} for d in system_demos]
    all_handles = [{**h, "type": "active_handle"} for h in active_handles]
    all_items = all_demos + all_handles

    total = len(all_items)

    # Apply pagination
    page = all_items[offset: offset + limit]

    # Split back into categories for backward-compatible response
    paged_demos = [item["name"] for item in page if item["type"] == "system_demo"]
    paged_handles = [
        {k: v for k, v in item.items() if k != "type"}
        for item in page
        if item["type"] == "active_handle"
    ]

    return {
        "success": True,
        "system_demos": paged_demos,
        "active_handles": paged_handles,
        "total": total,
        "count": len(page),
        "offset": offset,
        "limit": limit,
        "has_more": (offset + limit) < total,
    }
