"""
describe_estimator tool for sktime MCP.

Gets detailed information about an estimator's capabilities.
"""

from typing import Any

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.registry.tag_resolver import get_tag_resolver


def describe_estimator_tool(estimator: str) -> dict[str, Any]:
    """
    Get detailed information about a specific estimator.

    Args:
        estimator: Name of the estimator class (e.g., "ARIMA", "RandomForest")

    Returns:
        Dictionary with:
        - success: bool
        - name: Estimator name
        - task: Task type
        - module: Full module path
        - hyperparameters: Dict of parameter names with defaults
        - tags: Dict of capability tags
        - tag_explanations: Human-readable tag descriptions
        - docstring: First 500 chars of docstring

    Example:
        >>> describe_estimator_tool("ARIMA")
        {
            "success": True,
            "name": "ARIMA",
            "task": "forecasting",
            "hyperparameters": {"order": {"default": [1,0,0], "required": False}, ...},
            "tags": {"capability:pred_int": True, ...},
            ...
        }
    """
    registry = get_registry()
    tag_resolver = get_tag_resolver()

    node = registry.get_estimator_by_name(estimator)
    if node is None:
        # Try case-insensitive search
        all_estimators = registry.get_all_estimators()
        matches = [e for e in all_estimators if e.name.lower() == estimator.lower()]
        if matches:
            node = matches[0]
        else:
            return {
                "success": False,
                "error": f"Unknown estimator: {estimator}",
                "suggestion": "Use list_estimators to discover available estimators",
            }

    # Get tag explanations
    tag_explanations = tag_resolver.explain_tags(node.tags)

    return {
        "success": True,
        "name": node.name,
        "task": node.task,
        "module": node.module,
        "hyperparameters": node.hyperparameters,
        "tags": node.tags,
        "tag_explanations": tag_explanations,
        "docstring": node.docstring[:500] if node.docstring else None,
    }


def search_estimators_tool(query: str, limit: int = 20) -> dict[str, Any]:
    """
    Search estimators by name or description.

    Args:
        query: Search string (case-insensitive)
        limit: Maximum results

    Returns:
        Dictionary with matching estimators
    """
    registry = get_registry()

    try:
        matches = registry.search_estimators(query)[:limit]
        return {
            "success": True,
            "query": query,
            "results": [est.to_summary() for est in matches],
            "count": len(matches),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}
