"""
Discovery tools for sktime MCP.
Provides the query_registry tool.
"""

import difflib
import json
from typing import Any

from sktime_mcp.registry.interface import get_registry


def query_registry_tool(
    task: str | None = None,
    tags: dict[str, Any] | str | None = None,
    query: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Query the sktime registry for estimators,or capability tags.

    All filters are combined: task and tags are applied by the registry (same
    semantics with or without a query), then query narrows by name/description.

    To retrieve capability tags instead of estimators, set task="tag" (or "tags").

    Args:
        task: Filter by scitype (e.g., "forecaster", "classifier", "regressor",
              "transformer", "detector", "metric").
              Set to "tag" or "tags" to query available capability tags.
        tags: Filter estimators by capability tags. Can be a dictionary or a JSON string.
              Example JSON string: '{"capability:pred_int": true}'.
              Ignored when task="tag".
        query: Search by name/description (substring, case-insensitive).
        limit: Maximum number of results to return (default: 50). Ignored when task="tag".
        offset: Number of results to skip for pagination (default: 0). Ignored when task="tag".

    Returns:
        A dictionary with:
        - success: bool
        - results: List of matching estimators or tags
        - count: Number of results returned in this page
        - total: Total matching results (before limit/offset)
        - offset: Current offset (for pagination)
        - limit: Current limit (for pagination)
        - has_more: True if more results exist beyond this page
    """
    registry = get_registry()
    try:
        # Check pagination bounds
        if offset < 0:
            return {"success": False, "error": "offset must be a non-negative integer."}
        if limit < 1:
            return {"success": False, "error": "limit must be a positive integer."}

        # 1. Handle capability tags query
        if task in ("tag", "tags"):
            all_tags = registry.get_available_tags()
            if query:
                q_lower = query.lower()
                all_tags = [
                    t
                    for t in all_tags
                    if q_lower in t.get("tag", "").lower()
                    or q_lower in t.get("description", "").lower()
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
                "task_filter": task,
                "query": query,
            }

        # 2. Handle estimators query
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
            if isinstance(tags, str):
                try:
                    tags = json.loads(tags)
                except json.JSONDecodeError as e:
                    return {"success": False, "error": f"Invalid JSON string in 'tags': {e}"}
            if not isinstance(tags, dict):
                return {"success": False, "error": "tags must be a dictionary or a JSON string."}

            valid_tag_keys = {t["tag"] for t in registry.get_available_tags()}
            invalid_keys = [k for k in tags if k not in valid_tag_keys]
            if invalid_keys:
                suggestions = {
                    k: difflib.get_close_matches(k, valid_tag_keys, n=1, cutoff=0.6)
                    for k in invalid_keys
                }
                return {
                    "success": False,
                    "error": f"Invalid tag key(s): {invalid_keys}. Use task='tag' to see valid keys.",
                    "suggestions": {k: v[0] if v else None for k, v in suggestions.items()},
                }

            tags, tag_errors = registry.coerce_tag_filters(tags)
            if tag_errors:
                details = "; ".join(
                    f"'{e['tag']}': {e['value']!r} (expected {e['expected']})" for e in tag_errors
                )
                return {
                    "success": False,
                    "error": f"Invalid tag filter value(s): {details}. "
                    "Use task='tag' to see each tag's value_type.",
                    "invalid_tag_values": tag_errors,
                }

        # One filter path: task/tags are applied by the registry (sktime scitype
        # hierarchy, any-of lists, membership on list-valued tags) and the query
        # then narrows and ranks that list.
        if query:
            estimators = registry.search_estimators(query, task=task, tags=tags)
        else:
            estimators = registry.get_all_estimators(task=task, tags=tags)

        total = len(estimators)
        page = estimators[offset : offset + limit]
        results = [est.to_summary() for est in page]

        return {
            "success": True,
            "results": results,
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
        return {"success": False, "error": str(e)}
