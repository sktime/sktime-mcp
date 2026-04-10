"""
describe_estimator tool for sktime MCP.
Gets detailed information about an estimator's capabilities.
"""

import importlib.util
from typing import Any

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.registry.tag_resolver import get_tag_resolver


def _check_dependencies(tags: dict[str, Any]) -> dict[str, Any]:
    """Check whether an estimator's soft dependencies are available.

    Uses the ``python_dependencies`` tag from the estimator to determine
    which packages are required, then checks each against the current
    environment using ``importlib.util.find_spec``.

    Parameters
    ----------
    tags : dict
        Estimator tags dict, expected to contain ``python_dependencies`` key.

    Returns
    -------
    dict with keys:
        - dependencies_available : bool
            True if all soft dependencies are installed, False otherwise.
        - missing_dependencies : list of str
            Names of packages that are not installed. Empty if all present.
        - install_hint : str or None
            pip install command to resolve missing deps, or None if all present.
    """
    raw_deps = tags.get("python_dependencies", None)

    if not raw_deps:
        return {
            "dependencies_available": True,
            "missing_dependencies": [],
            "install_hint": None,
        }

    # normalize to list
    if isinstance(raw_deps, str):
        raw_deps = [raw_deps]

    missing = []
    for dep in raw_deps:
        # strip version specifiers e.g. "torch>=1.9" -> "torch"
        for sep in (">=", "<=", "==", "!=", ">", "<"):
            dep = dep.split(sep)[0]
        package_name = dep.strip()
        if importlib.util.find_spec(package_name) is None:
            missing.append(dep)

    available = len(missing) == 0
    install_hint = f"pip install {' '.join(missing)}" if missing else None

    return {
        "dependencies_available": available,
        "missing_dependencies": missing,
        "install_hint": install_hint,
    }


def describe_estimator_tool(estimator: str) -> dict[str, Any]:
    """
    Get detailed information about a specific estimator.

    Args:
        estimator: Name of the estimator class (e.g., "ARIMA", "ChronosForecaster")

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
        - dependencies_available: bool - True if all soft deps are installed
        - missing_dependencies: list of missing package names (empty if all present)
        - install_hint: pip install command to fix missing deps, or None

    Example:
        >>> describe_estimator_tool("ARIMA")
        {
            "success": True,
            "name": "ARIMA",
            "task": "forecasting",
            "dependencies_available": True,
            "missing_dependencies": [],
            "install_hint": None,
            ...
        }
        >>> describe_estimator_tool("ChronosForecaster")
        {
            "success": True,
            "name": "ChronosForecaster",
            "task": "forecasting",
            "dependencies_available": False,
            "missing_dependencies": ["torch", "transformers", "accelerate"],
            "install_hint": "pip install torch transformers accelerate",
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

    # Check soft dependency availability
    dep_info = _check_dependencies(node.tags)

    return {
        "success": True,
        "name": node.name,
        "task": node.task,
        "module": node.module,
        "hyperparameters": node.hyperparameters,
        "tags": node.tags,
        "tag_explanations": tag_explanations,
        "docstring": node.docstring[:500] if node.docstring else None,
        "dependencies_available": dep_info["dependencies_available"],
        "missing_dependencies": dep_info["missing_dependencies"],
        "install_hint": dep_info["install_hint"],
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
