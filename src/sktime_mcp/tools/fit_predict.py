"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

import asyncio
import logging
from typing import Any

from sktime_mcp.runtime.executor import get_executor
from sktime_mcp.runtime.jobs import get_job_manager

logger = logging.getLogger(__name__)


def _validate_horizon(horizon: int) -> dict[str, Any]:
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
            "error": (
                f"'horizon' must be greater than 0, got {horizon}. Example: {{\"horizon\": 12}}"
            ),
            "warnings": warnings,
        }
    return {"valid": True, "warnings": warnings}


def fit_predict_tool(
    estimator_handle: str,
    dataset: str | None = None,
    horizon: int = 12,
    data_handle: str | None = None,
    exog_handle: str | None = None,
) -> dict[str, Any]:
    """
    Execute a complete fit-predict workflow.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        horizon: Forecast horizon (default: 12)
        data_handle: Optional handle from load_data_source for custom data
        exog_handle: Optional handle for exogenous variables (X)

    Returns:
        Dictionary with:
        - success: bool
        - predictions: Forecast values
        - horizon: Number of steps predicted

    Example:
        >>> fit_predict_tool("est_abc123", "airline", horizon=12)
        {
            "success": True,
            "predictions": {1: 450.2, 2: 460.5, ...},
            "horizon": 12
        }
    """
    validation = _validate_horizon(horizon)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }
    if dataset and data_handle:
        return {
            "success": False,
            "error": "Provide either 'dataset' or 'data_handle', not both.",
        }

    if data_handle is None and (not dataset or not str(dataset).strip()):
        return {
            "success": False,
            "error": (
                "Either 'dataset' (e.g. 'airline') or "
                "'data_handle' (from load_data_source) is required."
            ),
        }
    executor = get_executor()
    return executor.fit_predict(
        estimator_handle,
        dataset=dataset,
        horizon=horizon,
        data_handle=data_handle,
        exog_handle=exog_handle,
    )


def predict_tool(
    estimator_handle: str,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Generate predictions from a fitted estimator.

    Args:
        estimator_handle: Handle of a fitted estimator
        horizon: Forecast horizon

    Returns:
        Dictionary with predictions
    """
    validation = _validate_horizon(horizon)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }
    executor = get_executor()
    fh = list(range(1, horizon + 1))
    return executor.predict(estimator_handle, fh=fh)


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


def fit_predict_async_tool(
    estimator_handle: str,
    dataset: str | None = None,
    data_handle: str | None = None,
    exog_handle: str | None = None,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Execute a fit-predict workflow in the background (non-blocking).

    Schedules the training as a background job and returns immediately
    with a job_id. Use check_job_status to monitor progress.

    Accepts either a demo dataset name or a data handle from
    load_data_source -- exactly one must be provided.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        data_handle: Handle from load_data_source (e.g., "data_abc123")
        exog_handle: Optional handle for exogenous variables (X)
        horizon: Forecast horizon (default: 12)

    Returns:
        Dictionary with:
        - success: bool
        - job_id: Job ID for tracking progress
        - message: Information about the job

    Example:
        >>> fit_predict_async_tool("est_abc123", dataset="airline", horizon=12)
        >>> fit_predict_async_tool("est_abc123", data_handle="data_xyz", horizon=5)
    """
    validation = _validate_horizon(horizon)
    if not validation["valid"]:
        return {
            "success": False,
            "error": validation["error"],
        }
    if dataset and data_handle:
        return {
            "success": False,
            "error": "Provide either 'dataset' or 'data_handle', not both.",
        }

    if not dataset and not data_handle:
        return {
            "success": False,
            "error": (
                "Either 'dataset' (e.g. 'airline') or "
                "'data_handle' (from load_data_source) is required."
            ),
        }

    executor = get_executor()
    job_manager = get_job_manager()

    # Get estimator info
    try:
        handle_info = executor._handle_manager.get_info(estimator_handle)
        estimator_name = handle_info.estimator_name
    except Exception as e:
        logger.warning(f"Could not get estimator name: {e}")
        estimator_name = "Unknown"

    source_name = dataset if dataset else data_handle

    # Create job
    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle=estimator_handle,
        estimator_name=estimator_name,
        dataset_name=source_name,
        horizon=horizon,
        total_steps=3,
    )

    # Schedule the async coroutine on the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    coro = executor.fit_predict_async(
        estimator_handle,
        dataset=dataset,
        data_handle=data_handle,
        exog_handle=exog_handle,
        horizon=horizon,
        job_id=job_id,
    )
    asyncio.run_coroutine_threadsafe(coro, loop)

    return {
        "success": True,
        "job_id": job_id,
        "message": (
            f"Training job started for {estimator_name} on {source_name}. "
            f"Use check_job_status('{job_id}') to monitor progress."
        ),
        "estimator": estimator_name,
        "data_source": source_name,
        "horizon": horizon,
    }
