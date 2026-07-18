"""
Unified instantiate tool for sktime MCP.

Creates executable estimator instances and pipelines from a sktime craft
specification string and returns a handle registered in server memory.
"""

from typing import Any

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager


def instantiate_tool(
    spec: str,
) -> dict[str, Any]:
    """Create an estimator or pipeline instance using craft and return a handle.

    Parameters
    ----------
    spec : str
        A sktime craft specification string (e.g., "ARIMA(order=(1, 1, 1))" or
        "Detrender() * ARIMA()").

    Returns
    -------
    dict
        Dictionary containing the success status and the unique handle.
    """
    if not spec or not isinstance(spec, str):
        return {
            "success": False,
            "error": "A valid 'spec' string is required.",
        }

    executor = get_executor()
    return executor.instantiate(spec)


def release_handle_tool(handle: str) -> dict[str, Any]:
    """Release an estimator handle and free resources.

    Parameters
    ----------
    handle : str
        The handle ID to release.

    Returns
    -------
    dict
        Dictionary containing success status:
        - "success" : bool
            True if the handle was successfully released, False otherwise.
        - "handle" : str
            The handle ID that was requested for release.
        - "message" : str
            Status message indicating outcome.
    """
    handle_manager = get_handle_manager()
    released = handle_manager.release_handle(handle)
    return {
        "success": released,
        "handle": handle,
        "message": "Handle released" if released else "Handle not found",
    }


def list_handles_tool() -> dict[str, Any]:
    """List all active estimator handles.

    Returns
    -------
    dict
        Dictionary containing details of active handles:
        - "success" : bool
            True if the handles were retrieved successfully.
        - "handles" : list of dict
            Details of active handles including handle ID, estimator name, and state.
        - "count" : int
            The number of active handles.
    """
    handle_manager = get_handle_manager()
    handles = handle_manager.list_handles()
    return {
        "success": True,
        "handles": handles,
        "count": len(handles),
    }


def load_model_tool(path: str) -> dict[str, Any]:
    """Load a saved model from a local path or MLflow URI and register its handle.

    Parameters
    ----------
    path : str
        Local directory path or MLflow URI to the saved model.
        Examples:
        - "/tmp/my_arima_model" (Linux/macOS) or "C:\\Temp\\my_arima_model" (Windows)
        - "runs:/<run_id>/model"
        - "mlflow-artifacts:/<run_id>/artifacts/model"
        - "models:/<model_name>/<version>"

    Returns
    -------
    dict
        Dictionary containing success status and the new handle:
        - "success" : bool
            True if the model was loaded successfully.
        - "handle" : str, optional
            The registered handle ID for the loaded model.
        - "estimator" : str, optional
            Class name of the loaded estimator.
        - "path" : str
            The path/URI from which the model was loaded.
        - "message" : str
            Status message describing outcome.
        - "error" : str, optional
            Error message if "success" is False.
    """
    try:
        from sktime.utils.mlflow_sktime import load_model
    except ImportError:
        return {
            "success": False,
            "error": (
                "The 'mlflow' package is required to load saved models. "
                "Please install it with: pip install sktime-mcp[mlflow]"
            ),
        }
    try:
        instance = load_model(path)
        estimator_name = type(instance).__name__
        handle_manager = get_handle_manager()
        handle_id = handle_manager.create_handle(
            estimator_name=estimator_name,
            instance=instance,
            params={},
            metadata={"source": "loaded", "path": path},
        )
        handle_manager.mark_fitted(handle_id)
        return {
            "success": True,
            "handle": handle_id,
            "estimator": estimator_name,
            "path": path,
            "message": f"Successfully loaded {estimator_name}",
        }
    except Exception as exc:
        return {
            "success": False,
            "error": f"Failed to load model: {str(exc)}",
            "path": path,
        }
