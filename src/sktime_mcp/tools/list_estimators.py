"""
Discovery tools for sktime MCP.
Provides query_registry, list_estimators, and get_available_tags tools.
"""

import difflib
from typing import Any

from sktime_mcp.registry.interface import get_registry


def query_registry_tool(
    target: str = "estimators",
    task: str | None = None,
    tags: dict[str, Any] | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Unified entry point to query the sktime registry for estimators, capability tags, or performance metrics.

    Args:
        target: What registry target to search. One of: "estimators", "tags", "metrics".
        task: Filter by task type (e.g., "forecasting", "transformation", "classification",
              "regression", "clustering", "splitting", "detection", "alignment",
              "parameter_estimation", "network"). Applies to "estimators" and "metrics".
        tags: Key-value pair of capability tag filters (e.g., {"capability:pred_int": True}).
              Applies to "estimators".
        query: Substring search over name, module, or docstring. Applies to "estimators".
        limit: Maximum number of results to return (default: 50).
        offset: Number of results to skip for pagination (default: 0).

    Returns:
        Dictionary with:
        - success: bool
        - results: List of matching components or tags
        - count: Number of results in this page
        - total: Total matching results
        - offset: Current offset
        - limit: Current limit
        - has_more: True if more results exist
    """
    registry = get_registry()
    try:
        # Validate target
        valid_targets = ["estimators", "tags", "metrics"]
        if target not in valid_targets:
            return {
                "success": False,
                "error": f"Invalid target: '{target}'. Valid targets: {valid_targets}",
            }

        # Check pagination bounds
        if offset < 0:
            return {"success": False, "error": "offset must be a non-negative integer."}
        if limit < 1:
            return {"success": False, "error": "limit must be a positive integer."}

        # Handle target: tags
        if target == "tags":
            all_tags = registry.get_available_tags()
            # If query is provided, filter tags by name/description
            if query:
                q_lower = query.lower()
                all_tags = [
                    t for t in all_tags
                    if q_lower in t.get("tag", "").lower() or q_lower in t.get("description", "").lower()
                ]
            
            total = len(all_tags)
            page = all_tags[offset : offset + limit]
            return {
                "success": True,
                "results": page,
                "count": len(page),
                "total": total,
                "offset": offset,
                "limit": limit,
                "has_more": (offset + limit) < total,
            }

        # Handle target: metrics or estimators
        # Validate task if provided
        if task is not None:
            valid_tasks = registry.get_available_tasks()
            if task not in valid_tasks:
                suggestions = difflib.get_close_matches(task, valid_tasks, n=3, cutoff=0.6)
                return {
                    "success": False,
                    "error": f"Invalid task: '{task}'. Valid options: {valid_tasks}."
                    + (f" Did you mean: {suggestions}?" if suggestions else ""),
                }

        # Validate tag keys if provided
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
                    "error": f"Invalid tag key(s): {invalid_keys}. Use target='tags' to see valid keys.",
                    "suggestions": {k: v[0] if v else None for k, v in suggestions.items()},
                }

        # Fetch base list of components
        if target == "metrics":
            # Metrics are components with task == "metric"
            components = registry.get_all_estimators(task="metric")
        else:
            # Estimators are everything except metrics
            components = [e for e in registry.get_all_estimators() if e.task != "metric"]
            # Apply task filter
            if task:
                components = [e for e in components if e.task == task]
            # Apply tags filter
            if tags:
                components = registry._filter_by_tags(components, tags)

        # Apply query search if provided
        if query:
            q_lower = query.lower()
            filtered_components = []
            for node in components:
                name_lower = node.name.lower()
                module_lower = node.module.lower()
                doc_lower = node.docstring.lower() if node.docstring else ""
                if q_lower in name_lower or q_lower in module_lower or q_lower in doc_lower:
                    filtered_components.append(node)
            components = filtered_components

        # Pagination
        total = len(components)
        page = components[offset : offset + limit]
        results = [est.to_summary() for est in page]

        return {
            "success": True,
            "results": results,
            "count": len(results),
            "total": total,
            "offset": offset,
            "limit": limit,
            "has_more": (offset + limit) < total,
            "target": target,
            "task_filter": task,
            "tag_filter": tags,
            "query": query,
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# --- Backward Compatibility Wrappers ---

def list_estimators_tool(
    task: str | None = None,
    tags: dict[str, Any] | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """Deprecated: Use query_registry_tool with target='estimators' instead."""
    res = query_registry_tool(
        target="estimators",
        task=task,
        tags=tags,
        query=query,
        limit=limit,
        offset=offset,
    )
    if not res["success"]:
        return res
    return {
        "success": True,
        "estimators": res["results"],
        "count": res["count"],
        "total": res["total"],
        "offset": res["offset"],
        "limit": res["limit"],
        "has_more": res["has_more"],
        "task_filter": task,
        "tag_filter": tags,
        "query": query,
    }


def get_available_tasks() -> dict[str, Any]:
    """Get list of available task types."""
    registry = get_registry()
    return {
        "success": True,
        "tasks": registry.get_available_tasks(),
    }


def get_available_tags() -> dict[str, Any]:
    """Deprecated: Use query_registry_tool with target='tags' instead."""
    registry = get_registry()
    return {
        "success": True,
        "tags": registry.get_available_tags(),
    }
