"""
list_estimators tool for sktime MCP.
Discovers estimators by task type and capability tags.
"""

from typing import Any, Optional

from sktime_mcp.registry.interface import get_registry


def list_estimators_tool(
    task: Optional[str] = None,
    tags: Optional[dict[str, Any]] = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Discover sktime estimators by task type and capability tags.

    Args:
        task: Filter by task type. Options: "forecasting", "classification",
              "regression", "transformation", "clustering"
        tags: Filter by capability tags. Example: {"capability:pred_int": True}
        limit: Maximum number of results to return (default: 50)
        offset: Number of results to skip for pagination (default: 0).
                Use with limit to paginate through all estimators.
                Example: offset=50 returns results 51-100.

    Returns:
        Dictionary with:
        - success: bool
        - estimators: List of estimator summaries
        - count: Number of results returned in this page
        - total: Total matching estimators (before limit/offset)
        - offset: Current offset (for pagination)
        - limit: Current limit (for pagination)
        - has_more: True if more results exist beyond this page

    Example:
        >>> list_estimators_tool(task="forecasting", limit=50, offset=0)
        {
            "success": True,
            "estimators": [{"name": "ARIMA", "task": "forecasting", ...}, ...],
            "count": 50,
            "total": 128,
            "offset": 0,
            "limit": 50,
            "has_more": True
        }
        >>> list_estimators_tool(task="forecasting", limit=50, offset=50)
        {
            "success": True,
            "estimators": [...],
            "count": 50,
            "total": 128,
            "offset": 50,
            "limit": 50,
            "has_more": True
        }
    """
    registry = get_registry()
    try:
        estimators = registry.get_all_estimators(task=task, tags=tags)
        total = len(estimators)

        # Validate offset
        if offset < 0:
            return {
                "success": False,
                "error": "offset must be a non-negative integer.",
            }

        # Apply offset and limit for pagination
        page = estimators[offset : offset + limit]

        # Convert to summaries
        results = [est.to_summary() for est in page]

        return {
            "success": True,
            "estimators": results,
            "count": len(results),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total,
            "task_filter": task,
            "tag_filter": tags,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def get_available_tasks() -> dict[str, Any]:
    """Get list of available task types."""
    registry = get_registry()
    return {
        "success": True,
        "tasks": registry.get_available_tasks(),
    }


def get_available_tags() -> dict[str, Any]:
    """Get list of all available capability tags."""
    registry = get_registry()
    return {
        "success": True,
        "tags": registry.get_available_tags(),
    }
