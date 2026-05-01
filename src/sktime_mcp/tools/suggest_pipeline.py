"""
suggest_pipeline tool for sktime MCP.

Suggests valid estimator pipelines using the existing composition validator.
"""

from typing import Any

from sktime_mcp.composition.validator import get_composition_validator


def suggest_pipeline_tool(
    task: str,
    requirements: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Suggest a valid estimator pipeline for a target task.

    Args:
        task: Target task, e.g. "forecasting" or "classification".
        requirements: Optional capability requirements to pass to the validator.

    Returns:
        Dictionary with:
        - success: bool
        - task: Requested target task
        - requirements: Requirements used for the suggestion
        - suggestions: Ordered estimator names for the suggested pipeline
        - count: Number of suggested pipeline components
    """
    if not isinstance(task, str) or not task.strip():
        return {"success": False, "error": "task must be a non-empty string"}

    if requirements is not None and not isinstance(requirements, dict):
        return {"success": False, "error": "requirements must be a dictionary"}

    validator = get_composition_validator()
    normalized_task = task.strip()
    normalized_requirements = requirements or {}
    suggestions = validator.suggest_pipeline(normalized_task, normalized_requirements)

    return {
        "success": True,
        "task": normalized_task,
        "requirements": normalized_requirements,
        "suggestions": suggestions,
        "count": len(suggestions),
    }
