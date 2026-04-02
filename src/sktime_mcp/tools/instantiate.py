"""
instantiate_estimator tool for sktime MCP.

Creates executable estimator instances and pipelines.
"""

from typing import Any, Dict, List, Optional

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager


def instantiate_estimator_tool(
    estimator: str,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create an estimator instance and return a handle.

    Args:
        estimator: Name of the estimator class (e.g., "ARIMA")
        params: Optional hyperparameters for the estimator

    Returns:
        Dictionary with:
        - success: bool
        - handle: Unique handle ID string
        - estimator: Name of the estimator
        - params: Parameters used

    Example:
        >>> instantiate_estimator_tool("ARIMA", {"order": [1, 1, 1]})
        {
            "success": True,
            "handle": "est_abc123def456",
            "estimator": "ARIMA",
            "params": {"order": [1, 1, 1]}
        }
    """
    executor = get_executor()
    return executor.instantiate(estimator, params)


def instantiate_pipeline_tool(
    components: List[str],
    params_list: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Create a pipeline from a list of components and return a handle.

    Args:
        components: List of estimator names in pipeline order
        params_list: Optional list of parameter dicts for each component

    Returns:
        Dictionary with:
        - success: bool
        - handle: Unique handle ID string
        - pipeline: Name of the pipeline
        - components: List of component names
        - params_list: Parameters used for each component

    Example:
        >>> instantiate_pipeline_tool(
        ...     ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
        ...     [{}, {}, {"order": [1, 1, 1]}]
        ... )
        {
            "success": True,
            "handle": "est_xyz789abc123",
            "pipeline": "ConditionalDeseasonalizer → Detrender → ARIMA",
            "components": ["ConditionalDeseasonalizer", "Detrender", "ARIMA"],
            "params_list": [{}, {}, {"order": [1, 1, 1]}]
        }
    """
    executor = get_executor()
    return executor.instantiate_pipeline(components, params_list)


def release_handle_tool(handle: str) -> Dict[str, Any]:
    """
    Release an estimator handle and free resources.

    Args:
        handle: The handle ID to release

    Returns:
        Dictionary with success status
    """
    handle_manager = get_handle_manager()
    released = handle_manager.release_handle(handle)
    return {
        "success": released,
        "handle": handle,
        "message": "Handle released" if released else "Handle not found",
    }


def list_handles_tool() -> Dict[str, Any]:
    """
    List all active estimator handles.

    Returns:
        Dictionary with list of active handles and their info
    """
    handle_manager = get_handle_manager()
    handles = handle_manager.list_handles()
    return {
        "success": True,
        "handles": handles,
        "count": len(handles),
    }
