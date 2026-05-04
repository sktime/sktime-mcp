"""
list_estimators tool for sktime MCP.
Discovers estimators by task type, capability tags, and/or name search.
"""

import difflib
from typing import Any

from sktime_mcp.registry.interface import get_registry


def list_estimators_tool(
    task: str | None = None,
    tags: dict[str, Any] | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Discover sktime estimators by task type, capability tags, and/or name search.

    All filters are combined: query narrows by name/docstring, then task and tags
    are applied on top.

    Args:
        task: Filter by task type. Options: "forecasting", "classification", "regression", "transformation", "clustering", "detection"
        tags: Filter by capability tags. Example: {"capability:pred_int": True}
        query: Search by name or description (substring, case-insensitive).
        limit: Maximum number of results to return (default: 50)
        offset: Number of results to skip for pagination (default: 0).

    Returns:
        Dictionary with:
        - success: bool
        - estimators: List of estimator summaries
        - count: Number of results returned in this page
        - total: Total matching estimators (before limit/offset)
        - offset: Current offset (for pagination)
        - limit: Current limit (for pagination)
        - has_more: True if more results exist beyond this page
    """
    registry = get_registry()
    try:
        # Validate task
        if task is not None:
            valid_tasks = registry.get_available_tasks()
            if task not in valid_tasks:
                suggestions = difflib.get_close_matches(task, valid_tasks, n=3, cutoff=0.6)
                return {
                    "success": False,
                    "error": f"Invalid task: '{task}'. Valid options: {valid_tasks}."
                    + (f" Did you mean: {suggestions}?" if suggestions else ""),
                }

        # Validate tag keys
        if tags is not None:
            valid_tag_keys = {t["tag"] for t in registry.get_available_tags()}
            invalid_keys = [k for k in tags if k not in valid_tag_keys]
            if invalid_keys:
                suggestions = {
                    k: difflib.get_close_matches(k, valid_tag_keys, n=1, cutoff=0.6)
                    for k in invalid_keys
                }
                return {
                    "success": False,
                    "error": f"Invalid tag key(s): {invalid_keys}. Use get_available_tags to see valid keys.",
                    "suggestions": {k: v[0] if v else None for k, v in suggestions.items()},
                }

        if query:
            estimators = registry.search_estimators(query)
            if task:
                estimators = [e for e in estimators if e.task == task]
            if tags:
                estimators = registry._filter_by_tags(estimators, tags)
        else:
            estimators = registry.get_all_estimators(task=task, tags=tags)

        total = len(estimators)

        if offset < 0:
            return {
                "success": False,
                "error": "offset must be a non-negative integer.",
            }

        if limit < 1:
            return {
                "success": False,
                "error": "limit must be a positive integer.",
            }

        page = estimators[offset : offset + limit]
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
            "query": query,
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
