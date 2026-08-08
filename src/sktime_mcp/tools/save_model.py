"""
save_model tool for sktime MCP.

Saves estimator instances via sktime's MLflow integration.
"""

import os
from collections.abc import Callable
from typing import Any

from sktime_mcp.runtime.handles import get_handle_manager

# MLflow URI schemes that must be passed through untouched
_MLFLOW_URI_PREFIXES = ("runs:/", "models:/", "mlflow-artifacts:/")


def resolve_model_path(path: str) -> str:
    """Expand ~ and resolve local filesystem paths to absolute form.

    MLflow URIs (runs:/, models:/, mlflow-artifacts:/, scheme://...) are
    returned unchanged. Without this, "~/model" creates a literal "~"
    directory in the server cwd, and relative paths land in the server's
    working directory with no way for the caller to learn where.
    """
    if path.startswith(_MLFLOW_URI_PREFIXES) or "://" in path:
        return path
    return os.path.abspath(os.path.expanduser(path))


def _get_mlflow_save_model() -> Callable[..., Any]:
    """Resolve sktime MLflow save utility lazily for better runtime compatibility."""
    try:
        from sktime.utils.mlflow_sktime import save_model as mlflow_save_model
    except Exception as exc:
        raise ImportError(
            "Unable to import sktime MLflow save_model utility. "
            "Ensure sktime and MLflow dependencies are installed."
        ) from exc
    return mlflow_save_model


def save_model_tool(
    estimator_handle: str,
    path: str,
    mlflow_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Save an instantiated estimator to a local path or URI using sktime+MLflow.

    Args:
        estimator_handle: Handle ID from instantiate
        path: Local directory or URI where the model should be saved
        mlflow_params: Optional extra keyword arguments for sktime MLflow save_model

    Returns:
        Dictionary with success status and confirmation message/path.
    """
    handle_manager = get_handle_manager()

    try:
        estimator = handle_manager.get_instance(estimator_handle)
    except KeyError:
        return {"success": False, "error": f"Handle not found: {estimator_handle}"}

    if not handle_manager.is_fitted(estimator_handle):
        handle_info = handle_manager.get_info(estimator_handle)
        return {
            "success": False,
            "error": (
                f"Estimator '{handle_info.estimator_name}' (handle: {estimator_handle}) "
                "has not been fitted. Call fit before saving."
            ),
        }

    if mlflow_params is not None and not isinstance(mlflow_params, dict):
        return {"success": False, "error": "mlflow_params must be a dictionary"}

    resolved_path = resolve_model_path(path)

    try:
        save_model = _get_mlflow_save_model()
        save_model(sktime_model=estimator, path=resolved_path, **(mlflow_params or {}))
        return {
            "success": True,
            "estimator_handle": estimator_handle,
            "saved_path": resolved_path,
            "message": f"Model saved successfully to '{resolved_path}'",
        }
    except Exception as exc:
        return {"success": False, "error": str(exc)}
