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
) -> dict[str, Any]:
    """
    Discover sktime estimators by task type and capability tags.

    Args:
        task: Filter by task type. Options: "forecasting", "classification",
              "regression", "transformation", "clustering"
        tags: Filter by capability tags. Example: {"capability:pred_int": True}
        limit: Maximum number of results to return (default: 50)

    Returns:
        Dictionary with:
        - success: bool
        - estimators: List of estimator summaries
        - count: Number of results
        - total: Total matching (before limit)

    Example:
        >>> list_estimators_tool(task="forecasting", tags={"capability:pred_int": True})
        {
            "success": True,
            "estimators": [{"name": "ARIMA", "task": "forecasting", ...}, ...],
            "count": 15,
            "total": 15
        }
    """
    registry = get_registry()

    try:
        estimators = registry.get_all_estimators(task=task, tags=tags)
        total = len(estimators)

        # Apply limit
        estimators = estimators[:limit]

        # Convert to summaries
        results = [est.to_summary() for est in estimators]

        return {
            "success": True,
            "estimators": results,
            "count": len(results),
            "total": total,
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
