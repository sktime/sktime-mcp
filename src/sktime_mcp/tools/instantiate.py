"""
Unified instantiate_estimator tool for sktime MCP.

Creates executable estimator instances and pipelines from a list of
component names.  A single-element list creates a plain estimator;
multiple elements create a validated pipeline.
"""

from typing import Any

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
    estimator_name: str | None = None,
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

    # check if keys match known parameters (warn, don't error)
    if estimator_name and params:
        registry = get_registry()
        node = registry.get_estimator_by_name(estimator_name)

        if node is not None and node.parameters:
            known_keys = set(node.parameters.keys())
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
    estimator: str | None = None,
    params: dict[str, Any] | None = None,
    components: list[str] | None = None,
    params_list: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Create an estimator or pipeline instance and return a handle.

    Accepts either:
    - `estimator` (+ optional `params`) for a single estimator, OR
    - `components` (+ optional `params_list`) for a pipeline.

    A single-element `components` list is equivalent to passing a single
    `estimator`.

    Parameters
    ----------
    estimator : str or None, default=None
        Name of a single estimator class (e.g., "ARIMA").
        Mutually exclusive with `components`.
    params : dict or None, default=None
        Optional hyperparameters for the single estimator.
    components : list of str or None, default=None
        List of estimator class names in pipeline order
        (e.g., ["Detrender", "ARIMA"]).
        Mutually exclusive with `estimator`.
    params_list : list of dict or None, default=None
        Optional list of hyperparameter dicts, one per component.

    Returns
    -------
    dict
        Dictionary containing the success status and the unique handle:
        - "success" : bool
            True if the instantiation succeeded, False otherwise.
        - "handle" : str
            The unique handle ID representing the instantiated object.
        - "estimator" : str
            Name of the estimator or pipeline description.
        - "params" : dict, optional
            Parameters used (if instantiated as a single estimator).
        - "params_list" : list of dict, optional
            List of parameters used per component (if a pipeline).
        - "warnings" : list of str, optional
            List of parameter validation or other warnings.
        - "error" : str, optional
            Error message if "success" is False.

    Examples
    --------
    Single estimator:

    >>> instantiate_estimator_tool(estimator="ARIMA", params={"order": [1, 1, 1]})

    Pipeline:

    >>> instantiate_estimator_tool(
    ...     components=["Detrender", "ARIMA"],
    ...     params_list=[{}, {"order": [1, 1, 1]}],
    ... )
    """
    Create an estimator instance and return a handle.

    Args:
        estimator: Name of the estimator class (e.g., "ARIMA")
        params: Optional parameters for the estimator

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

    if estimator is None and components is None:
        return {
            "success": False,
            "error": (
                "Either 'estimator' (single estimator name) or 'components' "
                "(list of estimator names for a pipeline) is required."
            ),
        }

    # Convert single-estimator form into the canonical list form
    if estimator is not None:
        components = [estimator]
        params_list = [params] if params is not None else None

    # ── Validate params_list ──────────────────────────────────────────
    all_warnings: list[str] = []

    if params_list is not None:
        if not isinstance(params_list, list):
            return {
                "success": False,
                "error": (
                    f"'params_list' must be a list of dictionaries, "
                    f"got {type(params_list).__name__}"
                ),
            }

        for i, p in enumerate(params_list):
            comp_name = components[i] if i < len(components) else None
            validation = _validate_params(p, estimator_name=comp_name)

            if not validation["valid"]:
                return {
                    "success": False,
                    "error": (
                        f"Validation failed for component {i} "
                        f"({comp_name or 'unknown'}): {validation['error']}"
                    ),
                }

            all_warnings.extend(validation["warnings"])
    elif len(components) == 1:
        # Single estimator with no explicit params — still validate None
        validation = _validate_params(None, estimator_name=components[0])
        all_warnings.extend(validation["warnings"])

    # ── Dispatch to executor ──────────────────────────────────────────
    executor = get_executor()

    if len(components) == 1:
        # Single estimator path
        single_params = params_list[0] if params_list else None
        result = executor.instantiate(components[0], single_params)
    else:
        # Pipeline path (includes composition validation)
        result = executor.instantiate_pipeline(components, params_list)

    # Attach any key-mismatch warnings to the response
    if all_warnings and result.get("success"):
        result["warnings"] = all_warnings

    return result


# Keep the old name as a thin alias for backward compatibility in tests
# and downstream code that hasn't been updated yet.
instantiate_pipeline_tool = instantiate_estimator_tool


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
