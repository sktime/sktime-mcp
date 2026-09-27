"""
describe_component tool for sktime MCP.
Gets detailed information about a component's capabilities, parameters, and tags.
"""

from typing import Any

from sktime.utils.dependencies import _check_estimator_deps, _check_soft_dependencies

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.registry.tag_resolver import get_tag_resolver


def _dependency_info(cls: type) -> dict[str, Any]:
    """Return availability and installation details for a component's dependencies."""
    available = _check_estimator_deps(cls, severity="none")
    dependencies = cls.get_class_tag("python_dependencies", None)

    if available or dependencies is None:
        return {
            "dependencies_available": available,
            "missing_dependencies": [],
            "install_hint": None,
        }

    if isinstance(dependencies, str):
        dependencies = [dependencies]

    missing = []
    install_requirements = []
    for requirement in dependencies:
        if isinstance(requirement, (list, tuple)):
            requirement_available = any(
                _check_soft_dependencies(option, severity="none") for option in requirement
            )
            if not requirement_available:
                missing.append(" or ".join(requirement))
                install_requirements.append(requirement[0])
        elif not _check_soft_dependencies(requirement, severity="none"):
            missing.append(requirement)
            install_requirements.append(requirement)

    quoted_requirements = [
        f'"{requirement}"' if any(char in requirement for char in "<>=!~; ") else requirement
        for requirement in install_requirements
    ]
    install_hint = f"pip install {' '.join(quoted_requirements)}" if missing else None

    return {
        "dependencies_available": available,
        "missing_dependencies": missing,
        "install_hint": install_hint,
    }


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
        - task: Scitype (e.g., "forecaster", "transformer", "splitter", "metric")
        - module: Full module path
        - parameters: Dict of parameter names with defaults
        - tags: Dict of capability tags
        - tag_explanations: Human-readable tag descriptions
        - docstring: First 500 chars of docstring
        - dependencies_available: Whether the component can run in this environment
        - missing_dependencies: Unsatisfied package requirements
        - install_hint: pip command that installs one valid set of missing requirements
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
    dependency_info = _dependency_info(node.class_ref)

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
        **dependency_info,
    }
