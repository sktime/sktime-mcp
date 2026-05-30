"""
describe_estimator tool for sktime MCP.
Gets detailed information about an estimator's capabilities.
"""

from typing import Any

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.registry.tag_resolver import get_tag_resolver


def describe_component_tool(name: str) -> dict[str, Any]:
    """
    Get detailed information about ANY class or component in the sktime ecosystem
    (estimators, transformers, splitters, metrics, aligners).

    Args:
        name: Name of the component class (e.g., "ARIMA", "SlidingWindowSplitter", "MeanAbsolutePercentageError")

    Returns:
        Dictionary with:
        - success: bool
        - name: Component name
        - task: Task type (e.g., "forecasting", "transformation", "splitting", "metric")
        - module: Full module path
        - parameters: Dict of parameter names with defaults
        - tags: Dict of capability tags
        - tag_explanations: Human-readable tag descriptions
        - docstring: First 500 chars of docstring
    """
    registry = get_registry()
    tag_resolver = get_tag_resolver()

    node = registry.get_estimator_by_name(name)
    if node is None:
        # Try case-insensitive search
        all_estimators = registry.get_all_estimators()
        matches = [e for e in all_estimators if e.name.lower() == name.lower()]
        if matches:
            node = matches[0]
        else:
            return {
                "success": False,
                "error": f"Unknown component class: {name}",
                "suggestion": "Use query_registry to discover available component classes",
            }

    # Get tag explanations
    tag_explanations = tag_resolver.explain_tags(node.tags)

    doc = node.docstring or "No description available."

    return {
        "success": True,
        "name": node.name,
        "task": node.task,
        "module": node.module,
        "parameters": node.hyperparameters,
        # Keep hyperparameters for backward compatibility with describe_estimator_tool
        "hyperparameters": node.hyperparameters,
        "tags": node.tags,
        "tag_explanations": tag_explanations,
        "docstring": doc[:500],
    }
