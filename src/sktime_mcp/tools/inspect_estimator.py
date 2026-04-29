"""
inspect_estimator tool for sktime MCP.

Inspects the current state of a managed estimator handle,
including hyperparameters and fitted attributes.
"""

import logging
from typing import Any

from sktime_mcp.runtime.handles import get_handle_manager

logger = logging.getLogger(__name__)


def _extract_fitted_params(instance: Any) -> dict[str, Any]:
    """Extract fitted parameters (attributes ending with ``_``) from an estimator.

    Only JSON-safe scalar and collection types are included; complex objects
    are converted to their string representation.
    """
    fitted_params: dict[str, Any] = {}
    for attr_name in sorted(dir(instance)):
        # Fitted attributes follow the scikit-learn convention of trailing _
        if not attr_name.endswith("_") or attr_name.startswith("__"):
            continue
        # Skip private/internal attributes
        if attr_name.startswith("_"):
            continue
        try:
            value = getattr(instance, attr_name)
        except Exception:
            continue

        # Skip methods / callables
        if callable(value):
            continue

        # Convert to JSON-safe representation
        fitted_params[attr_name] = _to_json_safe(value)

    return fitted_params


def _to_json_safe(value: Any) -> Any:
    """Best-effort conversion of a value to a JSON-serializable type."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    # NumPy scalars
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            if value.size <= 50:
                return value.tolist()
            return f"ndarray(shape={value.shape}, dtype={value.dtype})"
    except ImportError:
        pass

    # Pandas types
    try:
        import pandas as pd

        if isinstance(value, pd.Series):
            if len(value) <= 50:
                return value.tolist()
            return f"Series(length={len(value)}, dtype={value.dtype})"
        if isinstance(value, pd.DataFrame):
            return f"DataFrame(shape={value.shape})"
        if isinstance(value, pd.Timestamp):
            return value.isoformat()
        if value is pd.NaT:
            return None
    except ImportError:
        pass

    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) <= 50:
            return [_to_json_safe(item) for item in value]
        return f"list(length={len(value)})"

    # Fallback
    return str(value)


def get_estimator_params_tool(handle: str) -> dict[str, Any]:
    """Return the hyperparameters and fitted attributes of a managed estimator.

    Args:
        handle: Estimator handle ID (e.g., ``est_abc123def456``).

    Returns:
        Dictionary with:
        - success: bool
        - handle: The handle ID queried
        - estimator_name: Name of the estimator
        - is_fitted: Whether the estimator has been fitted
        - hyperparameters: Current ``get_params()`` output
        - fitted_params: Fitted attributes (only when ``is_fitted`` is True)
        - created_at: ISO timestamp of handle creation

    Example::

        >>> get_estimator_params_tool("est_abc123def456")
        {
            "success": True,
            "handle": "est_abc123def456",
            "estimator_name": "ARIMA",
            "is_fitted": True,
            "hyperparameters": {"order": [1, 1, 1], ...},
            "fitted_params": {"aic_": 450.2, ...},
            "created_at": "2026-04-29T12:00:00"
        }
    """
    handle_manager = get_handle_manager()

    # --- Validate handle ---
    if not isinstance(handle, str) or not handle.strip():
        return {
            "success": False,
            "error": "'handle' must be a non-empty string.",
        }

    try:
        info = handle_manager.get_info(handle)
    except KeyError:
        return {
            "success": False,
            "error": f"Handle not found: {handle}",
            "suggestion": "Use list_handles to see active handles.",
        }

    instance = info.instance

    # --- Extract hyperparameters via get_params() ---
    try:
        hyperparameters = instance.get_params(deep=False)
        hyperparameters = _to_json_safe(hyperparameters)
    except Exception:
        hyperparameters = info.params

    # --- Extract fitted params if fitted ---
    fitted_params = {}
    if info.fitted:
        try:
            fitted_params = _extract_fitted_params(instance)
        except Exception as exc:
            logger.warning(f"Could not extract fitted params: {exc}")
            fitted_params = {"_extraction_error": str(exc)}

    result: dict[str, Any] = {
        "success": True,
        "handle": handle,
        "estimator_name": info.estimator_name,
        "is_fitted": info.fitted,
        "hyperparameters": hyperparameters,
        "created_at": info.created_at.isoformat(),
    }

    if info.fitted:
        result["fitted_params"] = fitted_params

    return result
