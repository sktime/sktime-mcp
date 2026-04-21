"""
instantiate_estimator tool for sktime MCP.

Creates executable estimator instances and pipelines.
"""

from typing import Any, Optional

from sktime_mcp.registry.interface import get_registry
from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.handles import get_handle_manager


def _is_safe_value(value: Any) -> bool:
    """Recursively check if a value is a safe, serializable type."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return True

    if isinstance(value, (list, tuple)):
        return all(_is_safe_value(item) for item in value)

    if isinstance(value, dict):
        return all(isinstance(k, str) and _is_safe_value(v) for k, v in value.items())

    return False


def _validate_params(
    params: Any,
    estimator_name: Optional[str] = None,
) -> dict[str, Any]:
    """
    Validate params for type safety and optionally check keys.

    Args:
        params: The params argument to validate.
        estimator_name: Optional estimator name for key validation.

    Returns:
        Dictionary with valid (bool), error (str), and warnings (list).
    """
    warnings = []

    # None means "use defaults", always valid
    if params is None:
        return {"valid": True, "warnings": warnings}

    # params must be a dict
    if not isinstance(params, dict):
        return {
            "valid": False,
            "error": (
                f"'params' must be a dictionary, got {type(params).__name__}. "
                f'Example: {{"order": [1, 1, 1], "suppress_warnings": true}}'
            ),
            "warnings": warnings,
        }

    # reject unsafe value types like callables, classes, modules
    for key, value in params.items():
        if not isinstance(key, str):
            return {
                "valid": False,
                "error": (
                    f"Parameter keys must be strings, got {type(key).__name__} for key: {key!r}"
                ),
                "warnings": warnings,
            }

        if not _is_safe_value(value):
            return {
                "valid": False,
                "error": (
                    f"Unsupported type for parameter '{key}': "
                    f"{type(value).__name__}. "
                    f"Only primitive types (str, int, float, bool, list, "
                    f"tuple, dict, None) are allowed."
                ),
                "warnings": warnings,
            }

    # check if keys match known hyperparameters (warn, don't error)
    if estimator_name and params:
        registry = get_registry()
        node = registry.get_estimator_by_name(estimator_name)

        if node is not None and node.hyperparameters:
            known_keys = set(node.hyperparameters.keys())
            provided_keys = set(params.keys())
            unknown_keys = provided_keys - known_keys

            if unknown_keys:
                warnings.append(
                    f"Unknown parameter(s) for {estimator_name}: "
                    f"{sorted(unknown_keys)}. "
                    f"Known parameters: {sorted(known_keys)}"
                )

    return {"valid": True, "warnings": warnings}


def instantiate_estimator_tool(
    estimator: str,
    params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
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
        - warnings: List of any validation warnings

    Example:
        >>> instantiate_estimator_tool("ARIMA", {"order": [1, 1, 1]})
        {
            "success": True,
            "handle": "est_abc123def456",
            "estimator": "ARIMA",
            "params": {"order": [1, 1, 1]}
        }
    """
    # validate params before passing to executor
    validation = _validate_params(params, estimator_name=estimator)

    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }

    executor = get_executor()
    result = executor.instantiate(estimator, params)

    # attach any key-mismatch warnings to the response
    if validation["warnings"] and result.get("success"):
        result["warnings"] = validation["warnings"]

    return result


def instantiate_pipeline_tool(
    components: list[str],
    params_list: Optional[list[dict[str, Any]]] = None,
) -> dict[str, Any]:
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
        - warnings: List of any validation warnings

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
    all_warnings = []

    # validate each params dict in params_list
    if params_list is not None:
        if not isinstance(params_list, list):
            return {
                "success": False,
                "error": (
                    f"'params_list' must be a list of dictionaries, "
                    f"got {type(params_list).__name__}"
                ),
            }

        for i, params in enumerate(params_list):
            comp_name = components[i] if i < len(components) else None
            validation = _validate_params(params, estimator_name=comp_name)

            if not validation["valid"]:
                return {
                    "success": False,
                    "error": (
                        f"Validation failed for component {i} "
                        f"({comp_name or 'unknown'}): {validation['error']}"
                    ),
                }

            all_warnings.extend(validation["warnings"])

    executor = get_executor()
    result = executor.instantiate_pipeline(components, params_list)

    if all_warnings and result.get("success"):
        result["warnings"] = all_warnings

    return result


def release_handle_tool(handle: str) -> dict[str, Any]:
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


def list_handles_tool() -> dict[str, Any]:
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


def load_model_tool(path: str) -> dict[str, Any]:
    """
    Load a saved model from a local path or MLflow URI and register its handle.

    Args:
        path: Local directory path or MLflow URI to the saved model.
              Examples:
                - "/tmp/my_arima_model"
                - "runs:/<run_id>/model"
                - "mlflow-artifacts:/<run_id>/artifacts/model"
                - "models:/<model_name>/<version>"
    Returns:
        Dictionary with success status and the new handle.
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