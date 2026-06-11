"""
describe_component tool for sktime MCP.

Gets detailed metadata about any sktime component class.
"""

from typing import Any

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.registry.tag_resolver import get_tag_resolver


def describe_component_tool(name: str) -> dict[str, Any]:
    """Get detailed information about any class or component in the sktime ecosystem.

    This includes estimators, transformers, splitters, metrics, and aligners.

    Parameters
    ----------
    name : str
        Name of the component class (e.g., "ARIMA", "SlidingWindowSplitter",
        "MeanAbsolutePercentageError"). Case-insensitive.

    Returns
    -------
    dict
        A dictionary containing detailed component information:
        - "success" : bool
            True if the component was found, False otherwise.
        - "name" : str
            The formal name of the component class.
        - "task" : str
            The task type of the component (e.g., "forecasting", "splitting", "metric").
        - "module" : str
            The full import path of the module containing the component.
        - "parameters" : dict
            A dictionary mapping constructor parameter names to metadata dicts
            with ``default`` and ``required`` keys.
        - "tags" : dict
            A dictionary mapping capability tag names to their values.
            Values may be bool, str, list, or null depending on the tag.
        - "tag_explanations" : dict
            A dictionary mapping capability tags to human-readable explanations.
        - "docstring" : str
            A preview of the component's docstring (first 500 characters).
        - "error" : str, optional
            Error message if "success" is False.
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
        "tags": node.tags,
        "tag_explanations": tag_explanations,
        "docstring": doc[:500],
    }
