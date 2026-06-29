"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def _validate_horizon(horizon: Any) -> dict[str, Any]:
    """
    Validate the horizon parameter.
    Checks if the horizon parameter is strictly integer or not
    Checks if the horizon parameter is greater than 0 or not
    """
    warnings = []
    if not isinstance(horizon, int):
        return {
            "valid": False,
            "error": (
                f"'horizon' must be an integer, got {type(horizon).__name__}. "
                f'Example: {{"horizon": 12}}'
            ),
            "warnings": warnings,
        }
    if horizon <= 0:
        return {
            "valid": False,
            "error": f"Invalid horizon={horizon}. horizon must be a positive integer greater than 0.",
            "warnings": warnings,
        }
    return {"valid": True, "warnings": warnings}





def fit_tool(
    estimator_handle: str,
    dataset: str | None = None,
    data_handle: str | None = None,
) -> dict[str, Any]:
    """
    Fit an estimator on a dataset.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset
        data_handle: Optional handle from load_data_source for custom data
    """
    executor = get_executor()
    
    if dataset and data_handle:
        return {"success": False, "error": "Provide either 'dataset' or 'data_handle', not both."}
    
    if not dataset and not data_handle:
        return {"success": False, "error": "Either 'dataset' or 'data_handle' is required."}

    if data_handle is not None:
        if data_handle not in executor._data_handles:
            return {
                "success": False,
                "error": f"Unknown data handle: {data_handle}",
            }
        data_info = executor._data_handles[data_handle]
        y = data_info["y"]
        X = data_info.get("X")
    else:
        data_result = executor.load_dataset(dataset)
        if not data_result["success"]:
            return data_result
        y = data_result["data"]
        X = data_result.get("exog")

    fit_result = executor.fit(estimator_handle, y, X=X)
    
    if fit_result.get("success") and dataset:
        try:
            handle_info = executor._handle_manager.get_info(estimator_handle)
            handle_info.metadata["training_dataset"] = dataset
        except Exception as e:
            logger.warning(f"Could not record training dataset: {e}")
            
    return fit_result


def predict_tool(
    estimator_handle: str,
    horizon: int = 12,
    mode: str = "predict",
    coverage: float | list[float] = 0.9,
    alpha: float | list[float] | None = None,
    dataset: str | None = None,
    data_handle: str | None = None,
) -> dict[str, Any]:
    """
    Generate predictions from a fitted estimator.

    Args:
        estimator_handle: Handle of a fitted estimator
        horizon: Forecast horizon
        mode: Prediction mode
        coverage: Coverage level for intervals
        alpha: Alpha values for quantiles
        dataset: Dataset for X data (for non-forecasters)
        data_handle: Data handle for X data (for non-forecasters)
    """
    validation = _validate_horizon(horizon)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }
    
    executor = get_executor()
    X = None
    
    if dataset or data_handle:
        if dataset and data_handle:
            return {"success": False, "error": "Provide either 'dataset' or 'data_handle', not both."}
        
        if data_handle is not None:
            if data_handle not in executor._data_handles:
                return {"success": False, "error": f"Unknown data handle: {data_handle}"}
            data_info = executor._data_handles[data_handle]
            # For classifiers, 'y' in data_info (the data from load) might be X
            # But the executor load_dataset for classification returns y=X, exog=y.
            # Thus, the features are in `data_info["y"]`.
            X = data_info["y"]
        else:
            data_result = executor.load_dataset(dataset)
            if not data_result["success"]:
                return data_result
            X = data_result["data"]
            
    fh = list(range(1, horizon + 1))
    return executor.predict(
        estimator_handle,
        fh=fh,
        X=X,
        mode=mode,
        coverage=coverage,
        alpha=alpha,
    )


def list_datasets_tool() -> dict[str, Any]:
    """
    List available demo datasets.

    Returns:
        Dictionary with list of dataset names
    """
    executor = get_executor()
    return {
        "success": True,
        "datasets": executor.list_datasets(),
    }





def update_tool(
    estimator_handle: str,
    dataset: str | None = None,
    data_handle: str | None = None,
) -> dict[str, Any]:
    executor = get_executor()
    
    if dataset and data_handle:
        return {"success": False, "error": "Provide either 'dataset' or 'data_handle', not both."}
    
    if not dataset and not data_handle:
        return {"success": False, "error": "Either 'dataset' or 'data_handle' is required."}

    if data_handle is not None:
        if data_handle not in executor._data_handles:
            return {"success": False, "error": f"Unknown data handle: {data_handle}"}
        data_info = executor._data_handles[data_handle]
        y = data_info["y"]
        X = data_info.get("X")
    else:
        data_result = executor.load_dataset(dataset)
        if not data_result["success"]:
            return data_result
        y = data_result["data"]
        X = data_result.get("exog")
        
    return executor.update(estimator_handle, y, X=X)


def get_fitted_params_tool(estimator_handle: str) -> dict[str, Any]:
    executor = get_executor()
    return executor.get_fitted_params(estimator_handle)
