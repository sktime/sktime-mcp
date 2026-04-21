"""
fit_predict tool for sktime MCP.

Executes complete forecasting workflows.
"""

import asyncio
import logging
from typing import Any, Optional

from sktime_mcp.runtime.executor import get_executor

logger = logging.getLogger(__name__)


def fit_predict_tool(
    estimator_handle: str,
    dataset: str,
    horizon: int = 12,
    data_handle: Optional[str] = None,
) -> dict[str, Any]:
    """
    Execute a complete fit-predict workflow.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        horizon: Forecast horizon (default: 12)
        data_handle: Optional handle from load_data_source for custom data

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
    executor = get_executor()
    return executor.fit_predict(estimator_handle, dataset, horizon, data_handle=data_handle)


def fit_tool(
    estimator_handle: str,
    dataset: str,
) -> dict[str, Any]:
    """
    Fit an estimator on a dataset.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset

    Returns:
        Dictionary with success status
    """
    executor = get_executor()
    data_result = executor.load_dataset(dataset)
    if not data_result["success"]:
        return data_result

    return executor.fit(
        estimator_handle,
        y=data_result["data"],
        X=data_result.get("exog"),
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
    dataset: str,
    horizon: int = 12,
) -> dict[str, Any]:
    """
    Execute a fit-predict workflow in the background (non-blocking).

    This tool schedules the training as a background job and returns immediately
    with a job_id. Use check_job_status to monitor progress.

    Args:
        estimator_handle: Handle from instantiate_estimator
        dataset: Name of demo dataset (e.g., "airline", "sunspots")
        horizon: Forecast horizon (default: 12)

    Returns:
        Dictionary with:
        - success: bool
        - job_id: Job ID for tracking progress
        - message: Information about the job

    Example:
        >>> fit_predict_async_tool("est_abc123", "airline", horizon=12)
        {
            "success": True,
            "job_id": "abc-123-def-456",
            "message": "Training job started. Use check_job_status to monitor progress."
        }
    """

    from sktime_mcp.runtime.jobs import get_job_manager

    executor = get_executor()
    job_manager = get_job_manager()

    # Get estimator info
    try:
        handle_info = executor._handle_manager.get_info(estimator_handle)
        estimator_name = handle_info.estimator_name
    except Exception as e:
        logger.warning(f"Could not get estimator name: {e}")
        estimator_name = "Unknown"

    # Create job
    job_id = job_manager.create_job(
        job_type="fit_predict",
        estimator_handle=estimator_handle,
        estimator_name=estimator_name,
        dataset_name=dataset,
        horizon=horizon,
        total_steps=3,
    )

    # Schedule the async coroutine on the event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop in current thread, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    # Schedule the coroutine (non-blocking!)
    coro = executor.fit_predict_async(estimator_handle, dataset, horizon, job_id)
    asyncio.run_coroutine_threadsafe(coro, loop)

    return {
        "success": True,
        "job_id": job_id,
        "message": f"Training job started for {estimator_name} on {dataset}. Use check_job_status('{job_id}') to monitor progress.",
        "estimator": estimator_name,
        "dataset": dataset,
        "horizon": horizon,
    }
