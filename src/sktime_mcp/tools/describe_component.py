"""
describe_component tool for sktime MCP.
Gets detailed information about a component's capabilities, parameters, and tags.
"""

from typing import Any

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.registry.tag_resolver import get_tag_resolver


def describe_component_tool(name: str) -> dict[str, Any]:
    """
    Get detailed information about ANY class or component in the sktime ecosystem
    (estimators, transformers, splitters, metrics, aligners).

    Args:
        name: Name of the component class (e.g., "ARIMA", "SlidingWindowSplitter",
            "MeanAbsolutePercentageError") or its dotted import path
            (e.g., "sktime.forecasting.arima.ARIMA").

    Returns:
        Dictionary with:
        - success: bool
        - name: Component name
        - task: Scitype (e.g., "forecaster", "transformer", "splitter", "metric")
        - module: Full module path
        - parameters: Dict of parameter names with defaults
        - tags: Dict of capability tags
        - tag_explanations: Human-readable tag descriptions
        - docstring: First 500 chars of docstring
    """
    if not isinstance(name, str) or not name.strip():
        return {
            "success": False,
            "error": f"name must be a non-empty string naming a component class, got {name!r}",
            "suggestion": "Use query_registry to discover available component classes",
        }

    registry = get_registry()
    tag_resolver = get_tag_resolver()

    # Registry lookup (exact, then case-insensitive) or a dotted import path of a
    # BaseObject subclass; the name is never evaluated as an expression.
    node = registry.get_estimator_by_name(name)
    if node is None:
        error: dict[str, Any] = {
            "success": False,
            "error": f"Unknown component class: {name}",
            "suggestion": "Use query_registry to discover available component classes",
        }
        did_you_mean = registry.suggest_names(name)
        if did_you_mean:
            error["did_you_mean"] = did_you_mean
        return error

    # Get tag explanations
    tag_explanations = tag_resolver.explain_tags(node.tags)

    doc = node.docstring or "No description available."

    return {
        "success": True,
        "name": node.name,
        "task": node.task,
        "module": node.module,
        "parameters": node.parameters,
        # Keep hyperparameters for backward compatibility with describe_estimator_tool
        "hyperparameters": node.parameters,
        "tags": node.tags,
        "tag_explanations": tag_explanations,
        "docstring": doc[:500],
    }
