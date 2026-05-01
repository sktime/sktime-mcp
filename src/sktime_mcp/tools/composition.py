"""
Composition metadata tools for sktime MCP.

Expose pipeline planning information from the composition validator.
"""

from typing import Any

from sktime_mcp.composition.validator import get_composition_validator


def get_valid_compositions_tool(estimator: str) -> dict[str, Any]:
    """
    Return task types that can precede or follow an estimator in a pipeline.

    Args:
        estimator: Name of the estimator to inspect.

    Returns:
        Dictionary with:
        - success: bool
        - estimator: Requested estimator name
        - can_precede: Task types this estimator can precede
        - can_follow: Task types that can precede this estimator
        - next_step_hint: How agents can use the returned metadata
    """
    if not isinstance(estimator, str) or not estimator.strip():
        return {"success": False, "error": "estimator must be a non-empty string"}

    estimator_name = estimator.strip()
    validator = get_composition_validator()
    compositions = validator.get_valid_compositions(estimator_name)

    if "error" in compositions:
        return {
            "success": False,
            "estimator": estimator_name,
            "error": compositions["error"],
            "can_precede": compositions.get("can_precede", []),
            "can_follow": compositions.get("can_follow", []),
        }

    return {
        "success": True,
        "estimator": estimator_name,
        "can_precede": sorted(compositions["can_precede"]),
        "can_follow": sorted(compositions["can_follow"]),
        "next_step_hint": (
            "Use list_estimators with one of the can_precede task values to find "
            "valid next pipeline components."
        ),
    }
